# Setup Instructions

## Creating a Virtual Environment

### Using `venv` (Recommended)

1. **Create the virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python chunk1_data_structures_test.py
   ```

5. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Using `conda` (Alternative)

1. **Create a conda environment:**
   ```bash
   conda create -n maps python=3.9
   conda activate maps
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python chunk1_data_structures_test.py
```

## Dependencies

- `numpy>=1.20.0` - For numerical operations and array handling

