# Historical Table Digitizer CLI

A powerful command-line tool for extracting data from scanned historical tables using Claude's vision API with automatic validation and self-correction.

## ✨ Features

- 🤖 **Claude Vision API** - Uses Claude Sonnet 4.5 for accurate table extraction
- 📄 **PDF & Image Support** - Process both PDF pages and image files
- ✅ **Automatic Validation** - Checks column sums against totals
- 🔧 **Self-Correction** - Auto-fixes errors above threshold (default 3%)
- 📊 **CSV Output** - Clean, structured data ready for analysis
- 🎯 **Smart Number Recognition** - Distinguishes similar digits (3 vs 8, 4 vs 1, 0 vs O)

---

## 🚀 Installation

### 1. Clone or Download

```bash
cd tablesDigitizer
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your API key
# Get your key from: https://console.anthropic.com/
```

Your `.env` file should look like:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. (Optional) Install Poppler for PDF Support

**Windows:**
- Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add `bin` folder to PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

> **Note:** If you skip this step, you can still use the tool with image files (PNG, JPG, etc.)

---

## 📖 Usage

### Basic Usage - Extract from Image

```bash
python tables_digitizer.py extract \
    --image input/sample.png \
    --output results.csv
```

### Extract from PDF

```bash
python tables_digitizer.py extract \
    --pdf document.pdf \
    --pages 3-5 \
    --output results.csv
```

### Extract Specific Pages

```bash
# Single page
python tables_digitizer.py extract --pdf doc.pdf --pages 1 --output out.csv

# Page range
python tables_digitizer.py extract --pdf doc.pdf --pages 3-7 --output out.csv

# Specific pages
python tables_digitizer.py extract --pdf doc.pdf --pages 1,3,5,7 --output out.csv
```

### Adjust Error Threshold

```bash
# Use 5% threshold instead of default 3%
python tables_digitizer.py extract \
    --image table.png \
    --output results.csv \
    --threshold 5.0
```

### Limit Correction Attempts

```bash
# Try up to 5 correction attempts
python tables_digitizer.py extract \
    --image table.png \
    --output results.csv \
    --max-retries 5
```

---

## 🔍 How It Works

### 1. **Extraction**
- Converts PDF pages to images (if needed)
- Sends images to Claude's vision API
- Claude extracts table data and outputs CSV format

### 2. **Validation**
- Parses the CSV data
- Looks for "Total" or "Sum" columns
- Calculates sum of numeric columns
- Compares with expected totals
- Flags errors above threshold (default 3%)

### 3. **Self-Correction** (if errors found)
- Re-prompts Claude with error details
- Highlights problematic rows
- Requests more careful extraction
- Repeats up to max-retries times
- Returns best result

### 4. **Output**
- Saves clean CSV file
- Ready for Excel, pandas, or other tools

---

## 📊 Example Output

When you run the tool, you'll see:

```
╔═══════════════════════════════════════════════════════════╗
║       📊 Historical Table Digitizer                       ║
╚═══════════════════════════════════════════════════════════╝

📄 Converting PDF pages 3-5 to images...
  ✓ Saved page 3
  ✓ Saved page 4
  ✓ Saved page 5

============================================================
Processing: page_3.png
============================================================
🔍 Extracting table from page_3.png...
  ✓ Extraction complete

🔍 Validating extracted data...
  ✓ Parsed 25 rows, 8 columns
  ✓ Found total column: 'Total'
  ✓ All rows validated successfully (within 3.0% threshold)

✅ Success! Data saved to: results.csv

💡 Tip: Review the output file to verify accuracy
```

---

## 🎯 Tips for Best Results

### Image Quality
- Use high-resolution scans (300 DPI or higher)
- Ensure good contrast between text and background
- Avoid skewed or rotated images

### Table Structure
- Works best with clearly defined rows and columns
- Include column headers in the image
- If you have a "Total" column, validation will be automatic

### Handling Errors
- If validation fails repeatedly, try:
  - Increasing `--threshold` (e.g., 5.0 or 10.0)
  - Using higher quality scans
  - Manually reviewing the output CSV

### CSV Format
- Commas in table data are automatically replaced
- Empty cells are preserved as blank in CSV
- Column headers are included in output

---

## 🛠️ Troubleshooting

### "ANTHROPIC_API_KEY not found"
- Make sure you created `.env` file (not `.env.example`)
- Check that your API key is correct
- Verify the file is in the same directory as the script

### "PDF support not available"
- Install poppler (see installation section)
- Alternatively, convert PDF to images manually and use `--image`

### "Error calling Claude API"
- Check your internet connection
- Verify your API key is valid
- Check your API usage limits at https://console.anthropic.com/

### Poor Extraction Quality
- Use higher resolution images
- Ensure table is clearly visible
- Try adjusting the image contrast/brightness
- Consider pre-processing images to improve clarity

---

**Happy Digitizing! 📊✨**