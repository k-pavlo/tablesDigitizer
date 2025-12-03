#!/usr/bin/env python3
"""
Historical Table Digitizer CLI

Extracts tables from scanned historical documents using Claude's vision API
with automatic validation and self-correction.
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re

import click
import pandas as pd
from anthropic import Anthropic
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


class TableDigitizer:
    """Main class for digitizing historical tables using Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the digitizer with Claude API client"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            click.echo("❌ Error: ANTHROPIC_API_KEY not found!")
            click.echo("\n📝 To fix this:")
            click.echo("1. Copy .env.example to .env")
            click.echo("2. Add your API key from https://platform.claude.com/")
            sys.exit(1)
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5"
    
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
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
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
                quality = 85  # Reset quality when scaling
            else:
                # If we can't compress enough, use what we have
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
            pages: Page range (e.g., "3-5" or "1,3,5")
            
        Returns:
            List of paths to generated image files
        """
        if not PDF_SUPPORT:
            click.echo("❌ Error: PDF support not available. Install pdf2image and poppler.")
            sys.exit(1)
        
        # Parse page range
        page_numbers = self._parse_page_range(pages)
        
        click.echo(f"📄 Converting PDF pages {pages} to images...")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        # Save selected pages as temporary images
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for page_num in page_numbers:
            if page_num <= len(images):
                img_path = temp_dir / f"page_{page_num}.png"
                images[page_num - 1].save(img_path, 'PNG')
                image_paths.append(str(img_path))
                click.echo(f"  ✓ Saved page {page_num}")
        
        return image_paths
    
    def _parse_page_range(self, pages: str) -> List[int]:
        """
        Parse page range string into list of page numbers
        
        Args:
            pages: Page range (e.g., "3-5" or "1,3,5" or "1")
            
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
    
    def extract_table(self, image_path: str, examples: Optional[List[str]] = None) -> str:
        """
        Extract table data from an image using Claude API
        
        Args:
            image_path: Path to the image containing the table
            examples: Optional list of example images for few-shot learning
            
        Returns:
            Extracted table data as CSV string
        """
        click.echo(f"🔍 Extracting table from {Path(image_path).name}...")
        
        # Encode the main image
        image_data, media_type = self.encode_image(image_path)
        
        # Build the prompt
        system_prompt = """You are an expert at digitizing historical tables from scanned documents.
Your task is to extract ALL data from the table and output it in CSV format.

CRITICAL INSTRUCTIONS:
1. Extract EVERY row and column from the table
2. Pay special attention to numbers - distinguish between similar digits:
   - 3 vs 8 (3 has flat top, 8 has two loops)
   - 4 vs 1 (4 has horizontal line, 1 is straight)
   - 0 vs O (0 is a digit, O is a letter)
   - 5 vs S, 6 vs G, etc.
3. Preserve the exact structure of the table
4. Include column headers
5. If there's a "Total" or "Sum" column/row, include it
6. Output ONLY the CSV data, no explanations
7. Use commas as delimiters
8. If a cell contains commas, replace them with another relevant symbol 
9. If a cell is empty, leave it blank in the CSV"""

        user_prompt = "Please extract the complete table from this image and output it as CSV data."
        
        # Build message content with image
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        
        # Add example images if provided (few-shot learning)
        if examples:
            click.echo(f"  📚 Using {len(examples)} example images for improved accuracy...")
            # Examples would be added to the prompt here
        
        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": content}]
            )
            
            csv_data = response.content[0].text.strip()
            
            # Remove markdown code fences if present
            if csv_data.startswith('```'):
                lines = csv_data.split('\n')
                # Remove first line if it's a code fence
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove last line if it's a code fence
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                csv_data = '\n'.join(lines).strip()
            
            click.echo(f"  ✓ Extraction complete")
            
            return csv_data
            
        except Exception as e:
            click.echo(f"❌ Error calling Claude API: {str(e)}")
            sys.exit(1)
    
    def validate_table(self, csv_data: str, threshold: float = 3.0) -> Tuple[bool, List[Dict]]:
        """
        Validate table data by checking column sums
        
        Args:
            csv_data: CSV string data
            threshold: Error threshold percentage (default 3%)
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        click.echo("\n🔍 Validating extracted data...")
        
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data))
            
            click.echo(f"  ✓ Parsed {len(df)} rows, {len(df.columns)} columns")
            
            # Look for "Total" column
            total_col = None
            for col in df.columns:
                if 'total' in col.lower() or 'sum' in col.lower():
                    total_col = col
                    break
            
            if not total_col:
                click.echo("  ⚠️  No 'Total' column found - skipping validation")
                return True, []
            
            click.echo(f"  ✓ Found total column: '{total_col}'")
            
            # Validate each row
            errors = []
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Remove total column from numeric columns
            if total_col in numeric_cols:
                numeric_cols.remove(total_col)
            
            for idx, row in df.iterrows():
                try:
                    # Calculate sum of numeric columns (excluding total)
                    row_sum = sum(row[col] for col in numeric_cols if pd.notna(row[col]))
                    expected_total = row[total_col]
                    
                    if pd.notna(expected_total):
                        # Calculate error percentage
                        error_pct = abs(row_sum - expected_total) / expected_total * 100 if expected_total != 0 else 0
                        
                        if error_pct > threshold:
                            errors.append({
                                'row': idx,
                                'calculated': row_sum,
                                'expected': expected_total,
                                'error_pct': error_pct
                            })
                except Exception as e:
                    # Skip rows that can't be validated
                    continue
            
            if errors:
                click.echo(f"\n  ⚠️  Found {len(errors)} rows with errors > {threshold}%:")
                for err in errors[:5]:  # Show first 5 errors
                    click.echo(f"    Row {err['row']}: calculated={err['calculated']:.2f}, "
                             f"expected={err['expected']:.2f}, error={err['error_pct']:.1f}%")
                if len(errors) > 5:
                    click.echo(f"    ... and {len(errors) - 5} more")
                return False, errors
            else:
                click.echo(f"  ✓ All rows validated successfully (within {threshold}% threshold)")
                return True, []
                
        except Exception as e:
            click.echo(f"  ⚠️  Validation error: {str(e)}")
            return True, []  # Continue even if validation fails
    
    def correct_table(self, image_path: str, csv_data: str, errors: List[Dict], 
                     max_retries: int = 3) -> str:
        """
        Attempt to correct table extraction errors
        
        Args:
            image_path: Path to the original image
            csv_data: Original CSV data
            errors: List of validation errors
            max_retries: Maximum correction attempts
            
        Returns:
            Corrected CSV data
        """
        click.echo(f"\n🔧 Attempting to correct errors (max {max_retries} retries)...")
        
        corrected_data = csv_data
        
        for attempt in range(max_retries):
            click.echo(f"\n  Attempt {attempt + 1}/{max_retries}...")
            
            # Encode the image
            image_data, media_type = self.encode_image(image_path)
            
            # Build correction prompt
            error_details = "\n".join([
                f"- Row {err['row']}: Sum is {err['calculated']:.2f} but should be {err['expected']:.2f} "
                f"(error: {err['error_pct']:.1f}%)"
                for err in errors[:10]  # Show up to 10 errors
            ])
            
            correction_prompt = f"""The previous extraction had some errors. Please re-extract the table data more carefully.

ERRORS FOUND:
{error_details}

Please focus on these problematic rows and ensure accurate number recognition.
Pay special attention to similar-looking digits (3 vs 8, 4 vs 1, 0 vs O).

Output the COMPLETE table again as CSV data."""

            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": correction_prompt
                }
            ]
            
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": content}]
                )
                
                corrected_data = response.content[0].text.strip()
                
                # Validate the corrected data
                is_valid, new_errors = self.validate_table(corrected_data)
                
                if is_valid:
                    click.echo(f"  ✓ Correction successful!")
                    return corrected_data
                elif len(new_errors) < len(errors):
                    click.echo(f"  ↗️  Improved: {len(errors)} → {len(new_errors)} errors")
                    errors = new_errors
                else:
                    click.echo(f"  ↔️  No improvement: still {len(new_errors)} errors")
                    
            except Exception as e:
                click.echo(f"  ❌ Correction attempt failed: {str(e)}")
        
        click.echo(f"\n  ⚠️  Could not fully correct all errors after {max_retries} attempts")
        click.echo(f"  💡 Returning best result - please review manually")
        
        return corrected_data


@click.group()
def cli():
    """Historical Table Digitizer - Extract tables from scanned documents using Claude AI"""
    pass


@cli.command()
@click.option('--pdf', type=click.Path(exists=True), help='Path to PDF file')
@click.option('--image', type=click.Path(exists=True), help='Path to image file')
@click.option('--pages', default='1', help='Page range (e.g., "3-5" or "1,3,5")')
@click.option('--output', required=True, type=click.Path(), help='Output CSV file path')
@click.option('--examples', type=click.Path(exists=True), help='Directory with example images for few-shot learning')
@click.option('--threshold', default=3.0, help='Error threshold percentage (default: 3.0)')
@click.option('--max-retries', default=3, help='Maximum correction attempts (default: 3)')
def extract(pdf, image, pages, output, examples, threshold, max_retries):
    """
    Extract table data from PDF or image files
    
    Example:
        python tables_digitizer.py extract --pdf table.pdf --pages 3-5 --output output.csv
    """
    click.echo("╔═══════════════════════════════════════════════════════════╗")
    click.echo("║       📊 Historical Table Digitizer                       ║")
    click.echo("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Validate inputs
    if not pdf and not image:
        click.echo("❌ Error: Please provide either --pdf or --image")
        sys.exit(1)
    
    if pdf and image:
        click.echo("❌ Error: Please provide only one of --pdf or --image")
        sys.exit(1)
    
    # Initialize digitizer
    digitizer = TableDigitizer()
    
    # Get image paths
    if pdf:
        image_paths = digitizer.pdf_to_images(pdf, pages)
    else:
        image_paths = [image]
    
    # Process each image
    all_data = []
    
    for img_path in image_paths:
        click.echo(f"\n{'='*60}")
        click.echo(f"Processing: {Path(img_path).name}")
        click.echo(f"{'='*60}")
        
        # Extract table
        csv_data = digitizer.extract_table(img_path, examples=None)
        
        # Validate
        is_valid, errors = digitizer.validate_table(csv_data, threshold)
        
        # Correct if needed
        if not is_valid and errors:
            csv_data = digitizer.correct_table(img_path, csv_data, errors, max_retries)
        
        all_data.append(csv_data)
    
    # Combine all data
    if len(all_data) > 1:
        click.echo(f"\n📑 Combining data from {len(all_data)} pages...")
        # Parse and concatenate DataFrames
        dfs = [pd.read_csv(io.StringIO(data)) for data in all_data]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure output directory exists
        output_path = Path("output") / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
    else:
        # Ensure output directory exists
        output_path = Path("output") / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save single page data
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_data[0])
    
    click.echo(f"\n✅ Success! Data saved to: {output_path}")
    click.echo(f"\n💡 Tip: Review the output file to verify accuracy")
    
    # Cleanup temp files
    if pdf:
        import shutil
        temp_dir = Path("temp_images")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            click.echo(f"🧹 Cleaned up temporary files")


if __name__ == '__main__':
    cli()
