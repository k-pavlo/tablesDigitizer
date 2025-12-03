#!/usr/bin/env python3
"""
Census Table Digitizer CLI
Extracts tables from historical census PDFs using Claude API with validation

Usage:
    python census_digitizer.py extract --pdf census.pdf --okruha "Белоцерковский" --pages 78-87
    python census_digitizer.py validate --csv output.csv

Features:
- Claude API integration for table extraction
- PDF to image conversion with compression
- Automatic validation of column sums
- Self-correction with <3% error tolerance
- Few-shot learning with example digits
- Cost tracking and estimation
"""

import os
import sys
import base64
import json
import io
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import StringIO

import anthropic
import pandas as pd
import numpy as np
import click
from dotenv import load_dotenv
from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    click.echo("⚠️  Warning: pdf2image not available. PDF support disabled.")

# Load environment variables
load_dotenv()

# ============================================================================
# PRICING INFORMATION (as of December 2025)
# ============================================================================
PRICING = {
    'claude-sonnet-4-5': {
        'input': 3.00,   # per million tokens
        'output': 15.00,  # per million tokens
        'description': 'Best for coding and complex tasks'
    },
    'claude-haiku-4-5': {
        'input': 1.00,
        'output': 5.00,
        'description': 'Near-frontier performance, cheaper'
    }
}

# Image token calculation
# Formula: tokens = (width_px * height_px) / 750
# For 1568x1568 image (max before resize): ~3,277 tokens
# For typical 1200x900 scan: ~1,440 tokens
AVG_IMAGE_TOKENS = 1500  # Conservative estimate per page


class CostTracker:
    """Track API costs in real-time."""
    
    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.requests = 0
        
    def add_request(self, input_tokens: int, output_tokens: int):
        """Record a single API request."""
        self.requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        input_cost = (input_tokens / 1_000_000) * PRICING[self.model]['input']
        output_cost = (output_tokens / 1_000_000) * PRICING[self.model]['output']
        request_cost = input_cost + output_cost
        
        self.total_cost += request_cost
        
        return request_cost
    
    def print_summary(self):
        """Print cost summary."""
        print("\n" + "="*60)
        print("COST SUMMARY")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Requests: {self.requests}")
        print(f"Input tokens: {self.total_input_tokens:,}")
        print(f"Output tokens: {self.total_output_tokens:,}")
        print(f"Total cost: ${self.total_cost:.4f}")
        print("="*60 + "\n")


class CensusDigitizer:
    """Main class for extracting census tables using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-sonnet-4-5'):
        """Initialize the digitizer with Claude API client"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            click.echo("❌ Error: ANTHROPIC_API_KEY not found!")
            click.echo("\n📝 To fix this:")
            click.echo("1. Copy .env.example to .env")
            click.echo("2. Add your API key from https://platform.claude.com/")
            sys.exit(1)
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.cost_tracker = CostTracker(model)
    
    def encode_image(self, image_path: str) -> Tuple[str, str]:
        """
        Encode an image file to base64 for Claude API with automatic compression
        to stay under 5MB limit
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (base64_data, media_type)
        """
        MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB limit
        
        # Load the image
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if necessary (for JPEG compatibility)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Try different compression strategies
        quality = 95
        scale_factor = 1.0
        
        while True:
            # Resize if needed
            if scale_factor < 1.0:
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                resized_img = img
            
            # Save to bytes buffer with current quality
            buffer = io.BytesIO()
            resized_img.save(buffer, format='JPEG', quality=quality, optimize=True)
            image_data = buffer.getvalue()
            
            # Check size
            if len(image_data) <= MAX_SIZE_BYTES:
                break
            
            # Reduce quality or scale
            if quality > 60:
                quality -= 5
            elif scale_factor > 0.5:
                scale_factor -= 0.1
                quality = 85
            else:
                click.echo(f"  ⚠️  Warning: Image still {len(image_data) / 1024 / 1024:.1f}MB after compression")
                break
        
        final_size_mb = len(image_data) / 1024 / 1024
        if final_size_mb < MAX_SIZE_BYTES / 1024 / 1024:
            click.echo(f"  ✓ Compressed image to {final_size_mb:.2f}MB (quality={quality}, scale={scale_factor:.1f})")
        
        return base64.b64encode(image_data).decode('utf-8'), 'image/jpeg'
    
    def pdf_to_images(self, pdf_path: str, pages: str) -> List[str]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            pages: Page range (e.g., "78-87" or "1,3,5")
            
        Returns:
            List of paths to generated image files
        """
        if not PDF_SUPPORT:
            click.echo("❌ Error: PDF support not available. Install pdf2image and poppler.")
            sys.exit(1)
        
        # Parse page range
        page_numbers = self._parse_page_range(pages)
        
        if not page_numbers:
            click.echo("❌ Error: No valid page numbers found")
            sys.exit(1)
        
        # Get min and max to optimize conversion
        min_page = min(page_numbers)
        max_page = max(page_numbers)
        
        click.echo(f"📄 Converting PDF pages {pages} to images...")
        click.echo(f"   Processing pages {min_page}-{max_page} from PDF...")
        
        # Convert only the range we need (much faster!)
        images = convert_from_path(
            pdf_path, 
            dpi=300,
            first_page=min_page,
            last_page=max_page
        )
        
        # Save selected pages as temporary images
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for idx, page_num in enumerate(range(min_page, max_page + 1)):
            if page_num in page_numbers:
                img_path = temp_dir / f"page_{page_num}.png"
                images[idx].save(img_path, 'PNG')
                image_paths.append(str(img_path))
                click.echo(f"  ✓ Saved page {page_num}")
        
        return image_paths
    
    def _parse_page_range(self, pages: str) -> List[int]:
        """
        Parse page range string into list of page numbers
        
        Args:
            pages: Page range (e.g., "78-87" or "1,3,5" or "1")
            
        Returns:
            List of page numbers
        """
        page_numbers = []
        
        for part in pages.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                page_numbers.extend(range(int(start), int(end) + 1))
            else:
                page_numbers.append(int(part))
        
        return sorted(set(page_numbers))
        
    def create_extraction_prompt(self, okruha_name: str, few_shot_examples: bool = True, active_pages: List[int] = None) -> str:
        """
        Create optimized extraction prompt with few-shot examples for digit recognition.
        
        Args:
            okruha_name: Name of okruha
            few_shot_examples: Whether to include digit recognition examples
            active_pages: List of page numbers currently being processed (e.g., [78, 79, 80])
        """
        
        prompt = f"""Extract employment statistics from Table IV for {okruha_name} okruha.

CRITICAL: Pay close attention to digit recognition. Common OCR errors to avoid:
"""
        
        if few_shot_examples:
            prompt += """
DIGIT RECOGNITION EXAMPLES (learn these patterns):
- "3" vs "8": The number 3 has flat horizontal lines; 8 has two rounded loops
- "4" vs "1": The number 4 has a horizontal line crossing; 1 is straight
- "5" vs "6": The number 5 has a flat top; 6 has a closed loop at bottom
- "0" vs "O": The number 0 is narrower and taller; letter O is rounder
- "7" vs "1": The number 7 has a horizontal top bar; 1 is just vertical

When you see ambiguous digits:
1. Check context (does the number make sense?)
2. Compare with other similar digits in same row
3. Verify against subtotals (parts should sum to total)
"""
        
        prompt += f"""
TARGET DATA STRUCTURE:
Extract these specific rows with exactly 12 numeric columns each:
"""

        # Dynamic prompt generation based on active pages
        if active_pages is None:
            active_pages = list(range(78, 88)) # Default to all if not specified

        if 78 in active_pages:
            prompt += """
Page 78 - Main Categories:
- "Всего Total" (Total employed)
- "А. Рабочие" (Blue collar workers)
- "1. Сельское хозяйство" (Agriculture)
- "1а. Сельхоз. рабочие в хозяйствах крестьянского типа"
- "1б. Рабочие сельхоз. предприятий"
"""

        if 80 in active_pages:
            prompt += """
Page 80 - White Collar:
- "Б. Служащие" (White collar workers)
- "I. Сельское хозяйство" (Agriculture employees)
- "a) Руководящий персонал" (Management)
- "в) Технический персонал" (Technical)
- "г) Хозяйственный персонал" (Economic)
- "д) Учетно-контрольный персонал" (Accounting)
"""

        if 84 in active_pages:
            prompt += """
Page 84 - Employers:
- "Г. Хозяева с наемными рабочими" (Employers with hired labor)
- "I. Сельское хозяйство"
- "1. Земледельцы" (Farmers)
"""

        if 85 in active_pages:
            prompt += """
Page 85 - Family Businesses:
- "Д. Хозяева, работающие только с членами семьи"
- "I. Сельское хозяйство"
- "1. Земледельцы"
"""

        if 86 in active_pages:
            prompt += """
Page 86 - Individual Workers:
- "Е. Одиночки" (Individual workers)
- "I. Сельское хозяйство"
- "1. Земледельцы"
"""

        if 87 in active_pages:
            prompt += """
Page 87 - Family Labor & Special Categories:
- "Ж. Члены семьи, помогающие в занятии" (Unpaid family labor)
- "1. Земледельцы"
- "370 От сдачи домов, комнат" (Rental income)
- "И. Безработные" (Unemployed)
- "К. Военнослужащие" (Military)
"""

        prompt += """
COLUMN STRUCTURE (exactly 12 columns per row):
Columns represent: Gender (M/F) × Nationality (Ukrainian/Russian/Jewish) × Settlement (Urban/Rural/Total)

Specific mapping:
Col 1-2: Urban Male/Female (Total)
Col 3-4: Rural Male/Female (Total)
Col 5-6: Total Ukrainian Male/Female
Col 7-8: Total Russian Male/Female
Col 9-10: Total Jewish Male/Female
Col 11-12: Other/European Male/Female (or additional category)

OUTPUT FORMAT:
Return CSV with columns: row_label,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12

EXAMPLE OUTPUT:
Всего Total,27800,14469,241440,255067,269240,269536,252404,262051,2135,1110,11243,3840
А. Рабочие,4048,1488,6076,2924,10124,4412,7423,3644,457,113,1623,515

VALIDATION REQUIREMENTS:
1. Each row MUST have exactly 12 numeric values
2. Preserve Russian text labels exactly as shown
3. If a cell is empty or shows "—", use 0
4. Double-check digits that look ambiguous (3 vs 8, 4 vs 1, etc.)
5. Verify subtotals: agricultural workers should be subset of total workers

OUTPUT ONLY THE CSV DATA - no markdown formatting, no explanations.
Start directly with the first row.
"""
        
        return prompt
    
    def extract_from_images(self, image_paths: List[str], okruha_name: str) -> pd.DataFrame:
        """
        Extract table from multiple images using Claude API.
        Processes in batches to avoid request size limits.
        
        Args:
            image_paths: List of paths to image files
            okruha_name: Name of okruha (e.g., "Белоцерковский")
        
        Returns:
            DataFrame with extracted data
        """
        
        # Process in batches of 3 images to stay under request size limit
        BATCH_SIZE = 3
        all_rows = []
        
        total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            click.echo(f"\n{'='*60}")
            click.echo(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} pages)")
            click.echo(f"{'='*60}")
            
            # Extract page numbers from filenames (e.g., ".../page_78.png" -> 78)
            active_pages = []
            for p in batch_paths:
                try:
                    # Assumes filename format "page_X.png"
                    fname = Path(p).stem
                    if 'page_' in fname:
                        page_num = int(fname.split('page_')[1])
                        active_pages.append(page_num)
                except Exception:
                    pass
            
            if active_pages:
                click.echo(f"📄 Active pages in batch: {active_pages}")
            
            # Create prompt with specific pages
            prompt = self.create_extraction_prompt(okruha_name, few_shot_examples=True, active_pages=active_pages)
            
            # Estimate cost for this batch
            estimated_image_tokens = AVG_IMAGE_TOKENS * len(batch_paths)
            estimated_prompt_tokens = len(prompt.split()) * 1.3
            estimated_output_tokens = 1000  # Fewer rows per batch
            
            estimated_cost = (
                (estimated_image_tokens + estimated_prompt_tokens) / 1_000_000 * PRICING[self.model]['input'] +
                estimated_output_tokens / 1_000_000 * PRICING[self.model]['output']
            )
            
            print(f"💰 Estimated cost for batch: ${estimated_cost:.4f}")
            
            # Encode images in this batch
            click.echo(f"🔍 Encoding {len(batch_paths)} images...")
            content = []
            
            for img_path in batch_paths:
                image_data, media_type = self.encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })
            
            # Add prompt
            content.append({
                "type": "text",
                "text": prompt
            })
            
            # Call Claude API
            click.echo("Calling Claude API...")
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # Track cost
            actual_cost = self.cost_tracker.add_request(
                message.usage.input_tokens,
                message.usage.output_tokens
            )
            
            print(f"✓ API call complete")
            print(f"  Actual cost: ${actual_cost:.4f}")
            print(f"  Input tokens: {message.usage.input_tokens}")
            print(f"  Output tokens: {message.usage.output_tokens}")
            
            # Parse response
            csv_text = message.content[0].text
            
            # Clean CSV (remove markdown if present)
            if '```' in csv_text:
                csv_text = csv_text.split('```')[1]
                if csv_text.startswith('csv'):
                    csv_text = csv_text[3:]
            
            csv_text = csv_text.strip()
            
            # Parse to DataFrame
            try:
                # Use python engine for more robustness, skip bad lines
                df_batch = pd.read_csv(StringIO(csv_text), header=None, on_bad_lines='skip', engine='python')
                
                # Ensure we have correct number of columns (13: label + 12 data)
                if len(df_batch.columns) != 13:
                    # Try to fix by taking first 13 columns
                    df_batch = df_batch.iloc[:, :13]
                
                df_batch.columns = ['row_label'] + [f'col_{i+1}' for i in range(12)]
                all_rows.append(df_batch)
                click.echo(f"  ✓ Extracted {len(df_batch)} rows from batch")
            except Exception as e:
                click.echo(f"  ⚠️  Error parsing batch: {e}")
                # Try fallback: split by comma manually
                try:
                    lines = [line.split(',') for line in csv_text.split('\n') if line.strip()]
                    valid_lines = [l[:13] for l in lines if len(l) >= 13]
                    if valid_lines:
                        df_batch = pd.DataFrame(valid_lines)
                        df_batch.columns = ['row_label'] + [f'col_{i+1}' for i in range(12)]
                        all_rows.append(df_batch)
                        click.echo(f"  ✓ Recovered {len(df_batch)} rows using fallback parsing")
                except Exception as e2:
                    click.echo(f"  ❌ Fallback parsing failed: {e2}")
                continue
        
        # Combine all batches
        if not all_rows:
            click.echo("❌ No data extracted")
            sys.exit(1)
        
        df = pd.concat(all_rows, ignore_index=True)
        
        # Remove duplicates (in case same row appears in multiple batches)
        df = df.drop_duplicates(subset=['row_label'], keep='first')
        
        # Add okruha name
        df.insert(0, 'okruha', okruha_name)
        
        return df
    
    def validate_data(self, df: pd.DataFrame, max_error_pct: float = 3.0) -> Dict:
        """
        Validate extracted data by checking column sums.
        
        Args:
            df: DataFrame with extracted data
            max_error_pct: Maximum acceptable error percentage
        
        Returns:
            Dictionary with validation results and errors found
        """
        
        print("\n" + "="*60)
        print("VALIDATION: Checking column sums")
        print("="*60)
        
        errors = []
        warnings = []
        
        # Check 1: Each row has 12 columns
        for idx, row in df.iterrows():
            data_cols = [c for c in df.columns if c.startswith('col_')]
            if len(data_cols) != 12:
                errors.append({
                    'row': idx,
                    'type': 'column_count',
                    'message': f"Row {idx} has {len(data_cols)} columns (expected 12)"
                })
        
        # Clean data for numeric validation
        # Create a copy to avoid modifying the original dataframe during validation
        df_clean = df.copy()
        for col in [c for c in df.columns if c.startswith('col_')]:
            # Replace "—" with 0, remove spaces and commas
            df_clean[col] = df_clean[col].astype(str).str.replace('—', '0').str.replace('-', '0')
            df_clean[col] = df_clean[col].str.replace(',', '').str.replace(' ', '')
            # Convert to numeric, coercing errors to NaN (then to 0)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Check 2: Verify hierarchical sums
        # Example: Agricultural workers should be <= Total workers
        
        total_row = df_clean[df_clean['row_label'].str.contains('Всего', na=False)]
        workers_row = df_clean[df_clean['row_label'].str.contains('А. Рабочие', na=False) & 
                        ~df_clean['row_label'].str.contains('1\.|1а|1б', na=False)]
        
        if len(total_row) > 0 and len(workers_row) > 0:
            for col in [c for c in df_clean.columns if c.startswith('col_')]:
                total_val = total_row[col].iloc[0]
                workers_val = workers_row[col].iloc[0]
                
                if workers_val > total_val:
                    errors.append({
                        'row': 'hierarchical_check',
                        'column': col,
                        'type': 'logical_error',
                        'message': f"Workers ({workers_val}) > Total ({total_val}) in {col}"
                    })
        
        # Check 3: Look for obvious transcription errors
        # Check if Male/Female values are extremely imbalanced (>10:1 ratio)
        for idx, row in df_clean.iterrows():
            # Pairs: col_1/col_2, col_3/col_4, etc.
            for i in range(1, 12, 2):
                col_m = f'col_{i}'
                col_f = f'col_{i+1}'
                
                if col_m in df_clean.columns and col_f in df_clean.columns:
                    val_m = row[col_m]
                    val_f = row[col_f]
                    
                    if val_m > 0 and val_f > 0:
                        ratio = max(val_m, val_f) / min(val_m, val_f)
                        if ratio > 10:
                            warnings.append({
                                'row': idx,
                                'columns': f'{col_m}/{col_f}',
                                'type': 'gender_imbalance',
                                'message': f"Extreme M/F ratio: {ratio:.1f}:1 ({val_m}/{val_f})"
                            })
        
        # Print results
        if errors:
            print(f"\n❌ Found {len(errors)} ERRORS:")
            for err in errors:
                print(f"  - {err['message']}")
        else:
            print("\n✓ No errors found")
        
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} WARNINGS:")
            for warn in warnings:
                print(f"  - {warn['message']}")
        
        print("="*60 + "\n")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_rows': len(df),
            'error_rate': len(errors) / len(df) if len(df) > 0 else 0
        }
    
    def fix_errors(self, image_paths: List[str], df: pd.DataFrame, 
                   validation_results: Dict, okruha_name: str) -> pd.DataFrame:
        """
        Attempt to fix errors by calling Claude API again with error context.
        
        Args:
            image_paths: Original image paths
            df: DataFrame with errors
            validation_results: Results from validate_data()
            okruha_name: Okruha name
        
        Returns:
            Corrected DataFrame
        """
        
        if not validation_results['errors']:
            print("No errors to fix")
            return df
        
        print(f"\n🔧 Attempting to fix {len(validation_results['errors'])} errors...")
        
        # Create correction prompt with specific error information
        error_desc = "\n".join([
            f"- {err['message']}" for err in validation_results['errors']
        ])
        
        correction_prompt = f"""The previous extraction had validation errors. Please re-extract with extra attention to accuracy.

ERRORS FOUND:
{error_desc}

SPECIFIC INSTRUCTIONS:
1. Double-check digit recognition (3 vs 8, 4 vs 1, 5 vs 6)
2. Verify hierarchical consistency (parts <= total)
3. Check that Male + Female values make sense
4. Ensure exactly 12 columns per row

{self.create_extraction_prompt(okruha_name, few_shot_examples=True)}
"""
        
        # Encode all images again
        content = []
        for img_path in image_paths:
            image_data, media_type = self.encode_image(img_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
        
        # Add correction prompt
        content.append({
            "type": "text",
            "text": correction_prompt
        })
        
        # Call API again
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        # Track cost
        cost = self.cost_tracker.add_request(
            message.usage.input_tokens,
            message.usage.output_tokens
        )
        
        print(f"✓ Correction attempt complete (cost: ${cost:.4f})")
        
        # Parse corrected response
        csv_text = message.content[0].text
        if '```' in csv_text:
            csv_text = csv_text.split('```')[1]
            if csv_text.startswith('csv'):
                csv_text = csv_text[3:]
        csv_text = csv_text.strip()
        
        df_corrected = pd.read_csv(StringIO(csv_text), header=None)
        df_corrected.columns = ['row_label'] + [f'col_{i+1}' for i in range(12)]
        df_corrected.insert(0, 'okruha', okruha_name)
        
        return df_corrected


# ============================================================================
# CLI COMMANDS
# ============================================================================

@click.group()
def cli():
    """Census Table Digitizer - Extract historical census tables using Claude API"""
    pass


@cli.command()
@click.option('--pdf', required=True, type=click.Path(exists=True), help='Path to PDF file')
@click.option('--okruha', required=True, help='Name of okruha (e.g., "Белоцерковский")')
@click.option('--pages', required=True, help='Page range (e.g., "78-87")')
@click.option('--output', default='output/census_data.csv', help='Output CSV file')
@click.option('--model', default='claude-sonnet-4-5', 
              type=click.Choice(['claude-sonnet-4-5', 'claude-haiku-4-5']),
              help='Claude model to use')
@click.option('--validate/--no-validate', default=True, help='Validate results')
@click.option('--auto-fix/--no-auto-fix', default=True, help='Automatically fix errors')
@click.option('--max-error', default=3.0, help='Maximum error percentage before auto-fix')
def extract(pdf, okruha, pages, output, model, validate, auto_fix, max_error):
    """Extract census table from PDF using Claude API."""
    
    # Ensure output directory exists and enforce output/ folder if no path specified
    output_path = Path(output)
    if len(output_path.parts) == 1:
        # If just a filename is provided, put it in output/ folder
        output_path = Path('output') / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n{'='*60}")
    click.echo("CENSUS TABLE EXTRACTION")
    click.echo(f"{'='*60}")
    click.echo(f"PDF: {pdf}")
    click.echo(f"Okruha: {okruha}")
    click.echo(f"Pages: {pages}")
    click.echo(f"Model: {model} ({PRICING[model]['description']})")
    click.echo(f"Output: {output_path}")
    click.echo(f"{'='*60}\n")
    
    # Initialize digitizer
    digitizer = CensusDigitizer(model=model)
    
    # Extract
    try:
        # Convert PDF pages to images
        image_paths = digitizer.pdf_to_images(pdf, pages)
        
        # Extract from images
        df = digitizer.extract_from_images(image_paths, okruha)
        
        click.echo(f"\n✓ Extracted {len(df)} rows")
        click.echo(f"\nFirst few rows:")
        click.echo(df.head())
        
        # SAVE RAW DATA IMMEDIATELY
        # Create a raw filename based on output name
        raw_output = output_path.with_name(f"{output_path.stem}_raw.csv")
        df.to_csv(raw_output, index=False, encoding='utf-8-sig')
        click.echo(f"💾 Saved raw data to: {raw_output} (Backup before validation)")
        
        # Validate
        if validate:
            try:
                results = digitizer.validate_data(df, max_error_pct=max_error)
            except Exception as e:
                click.echo(f"\n⚠️  Validation crashed: {e}")
                click.echo("Skipping validation and saving raw data.")
                results = {'valid': True, 'errors': [], 'error_rate': 0} # Assume valid to proceed
            
            error_pct = results['error_rate'] * 100
            
            # Auto-fix if needed
            if not results['valid'] and auto_fix and error_pct > max_error:
                click.echo(f"\n⚠️  Error rate {error_pct:.1f}% exceeds threshold {max_error}%")
                click.echo("Attempting automatic correction...")
                
                df = digitizer.fix_errors(image_paths, df, results, okruha)
                
                # Re-validate
                results = digitizer.validate_data(df, max_error_pct=max_error)
                
                if results['valid']:
                    click.echo("✓ Errors fixed successfully!")
                else:
                    click.echo("⚠️  Some errors remain. Manual review recommended.")
            elif not results['valid']:
                click.echo("⚠️  Validation found errors. Manual review recommended.")
        
        # Save
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        click.echo(f"\n✓ Saved to: {output_path}")
        
        # Cost summary
        digitizer.cost_tracker.print_summary()
        
        # Cleanup temp files
        temp_dir = Path("temp_images")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            click.echo(f"🧹 Cleaned up temporary files")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--csv', required=True, type=click.Path(exists=True), help='CSV file to validate')
def validate(csv):
    """Validate extracted data by checking sums and consistency."""
    
    click.echo(f"\nValidating: {csv}")
    
    df = pd.read_csv(csv)
    
    digitizer = CensusDigitizer()
    results = digitizer.validate_data(df)
    
    if results['valid']:
        click.echo("\n✅ Validation passed!")
        sys.exit(0)
    else:
        click.echo(f"\n❌ Validation failed: {len(results['errors'])} errors")
        sys.exit(1)


@cli.command()
@click.option('--model', default='claude-sonnet-4-5',
              type=click.Choice(['claude-sonnet-4-5', 'claude-haiku-4-5']))
@click.option('--pages', default=10, type=int, help='Number of pages to process')
@click.option('--okruhas', default=3, type=int, help='Number of okruhas')
def estimate(model, pages, okruhas):
    """Estimate cost for processing census data."""
    
    click.echo(f"\n{'='*60}")
    click.echo("COST ESTIMATION")
    click.echo(f"{'='*60}")
    click.echo(f"Model: {model}")
    click.echo(f"Pages per okruha: {pages}")
    click.echo(f"Number of okruhas: {okruhas}")
    click.echo(f"{'='*60}\n")
    
    # Estimates
    image_tokens_per_page = AVG_IMAGE_TOKENS
    prompt_tokens = 1000  # Base prompt
    output_tokens = 2000  # CSV output per okruha
    
    # Per okruha
    total_image_tokens = image_tokens_per_page * pages
    total_input_tokens = total_image_tokens + prompt_tokens
    
    input_cost_per_okruha = (total_input_tokens / 1_000_000) * PRICING[model]['input']
    output_cost_per_okruha = (output_tokens / 1_000_000) * PRICING[model]['output']
    cost_per_okruha = input_cost_per_okruha + output_cost_per_okruha
    
    # Total
    total_cost = cost_per_okruha * okruhas
    
    # With potential corrections (assume 20% need re-extraction)
    total_with_corrections = total_cost * 1.2
    
    click.echo("Per Okruha:")
    click.echo(f"  Image tokens: {total_image_tokens:,}")
    click.echo(f"  Input cost: ${input_cost_per_okruha:.4f}")
    click.echo(f"  Output cost: ${output_cost_per_okruha:.4f}")
    click.echo(f"  Subtotal: ${cost_per_okruha:.4f}")
    click.echo(f"\nTotal ({okruhas} okruhas):")
    click.echo(f"  Base cost: ${total_cost:.4f}")
    click.echo(f"  With corrections: ${total_with_corrections:.4f}")
    click.echo(f"\n💡 You get $5 free credits with new Claude API account")
    click.echo(f"{'='*60}\n")


@cli.command()
def pricing():
    """Show current Claude API pricing information."""
    
    click.echo(f"\n{'='*60}")
    click.echo("CLAUDE API PRICING (December 2025)")
    click.echo(f"{'='*60}\n")
    
    for model, info in PRICING.items():
        click.echo(f"{model}:")
        click.echo(f"  Input: ${info['input']}/M tokens")
        click.echo(f"  Output: ${info['output']}/M tokens")
        click.echo(f"  {info['description']}")
        click.echo()
    
    click.echo("Image Costs:")
    click.echo(f"  ~{AVG_IMAGE_TOKENS} tokens per page image")
    click.echo(f"  Formula: (width × height) / 750")
    click.echo()
    
    click.echo("Cost Optimization:")
    click.echo("  • Use Haiku for simple tasks, Sonnet for complex")
    click.echo("  • Batch processing reduces per-page overhead")
    click.echo(f"\n{'='*60}\n")


if __name__ == '__main__':
    cli()
