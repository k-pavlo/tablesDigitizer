# Historical Table Digitizer CLI

Command-line tool for extracting historical tables from PDF files using Claude API with automatic validation and self-correction.

## Features

✅ **Claude API Integration** - Uses state-of-the-art vision models  
✅ **Automatic Validation** - Checks column sums and logical consistency  
✅ **Self-Correction** - Auto-fixes errors under 3% threshold  
✅ **Few-Shot Learning** - Teaches Claude digit patterns (3 vs 8, 4 vs 1)

---

## Installation

```bash

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac 
venv\Scripts\activate # On Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file and add API key like in .env.example
```

---

## Quick Start

### 1. Extract Table from PDF

```bash
python tables_digitizer.py extract \
    --pdf tables.pdf \
    --pages 3-5 \
    --output output.csv
```

**What it does:**
1. Uploads PDF to Claude API
2. Extracts table
3. Validates column sums
4. Auto-fixes errors >3%
5. Saves to CSV

**Output:**
```
✓ API call complete
✓ Extracted 25 rows

VALIDATION: Checking column sums
✓ No errors found

✓ Saved to: output.csv
```