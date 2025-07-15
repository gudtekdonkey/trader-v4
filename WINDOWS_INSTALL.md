# Windows Installation Guide

## Quick Setup for Windows Users

### Step 1: Install TA-Lib Using Windows Executable

**Method 1: Windows Installer (Recommended)**
1. Visit: https://ta-lib.org/install/
2. Click "Download ta-lib-0.4.0-windows.exe"
3. Run the installer **as Administrator**
4. Install to the default location (`C:\ta-lib`)
5. The installer should automatically add TA-Lib to your system PATH

**Verify Installation:**
```cmd
# Open a NEW command prompt and check PATH
echo %PATH%
# You should see C:\ta-lib\c\bin in the output

# Test the C library
python -c "import ctypes; ctypes.CDLL('ta_lib'); print('TA-Lib C library OK!')"
```

### Step 2: Run Windows Setup Script
```cmd
cd C:\projects\ai-crypto-trader-v3
setup-windows.bat
```

### Step 3: Manual Python Setup (if script fails)
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install TA-Lib Python package
python -m pip install TA-Lib

# Install other requirements
python -m pip install -r requirements.txt
```

## Troubleshooting TA-Lib on Windows

### Issue 1: Installer doesn't add to PATH
If the Windows installer didn't automatically add TA-Lib to PATH:

1. **Manual PATH Addition:**
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - In "System Variables", find "Path" → Edit
   - Add these entries:
     - `C:\ta-lib\c\bin`
     - `C:\ta-lib\c\include`
   - Click OK and restart Command Prompt

2. **Verify PATH:**
   ```cmd
   echo %PATH% | findstr ta-lib
   ```

### Issue 2: Python Package Installation Fails

**Option A: Use pre-compiled wheel**
```cmd
pip install --find-links=https://download.lfd.uci.edu/pythonlibs/archived/ TA-Lib
```

**Option B: Download specific wheel**
1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Download wheel for your Python version:
   - Python 3.11: `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`
   - Python 3.10: `TA_Lib-0.4.28-cp310-cp310-win_amd64.whl`
   - Python 3.9: `TA_Lib-0.4.28-cp39-cp39-win_amd64.whl`
3. Install: `pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`

**Option C: Use conda**
```cmd
conda install -c conda-forge ta-lib
```

### Issue 3: "ta_lib.dll not found"
```cmd
# Temporarily add to current session
set PATH=%PATH%;C:\ta-lib\c\bin

# Test again
python -c "import talib; print('Success!')"
```

### Issue 4: Visual Studio Build Tools Required
If you get compilation errors:
1. Download Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install C++ build tools
3. Restart command prompt and try again

## Complete Installation Test

After following the steps above, run this complete test:

```cmd
# Test 1: Check C library
python -c "import ctypes; ctypes.CDLL('ta_lib')"

# Test 2: Check Python package
python -c "import talib; print('TA-Lib version:', talib.__version__)"

# Test 3: Check specific function
python -c "import talib; import numpy as np; print('SMA test:', talib.SMA(np.array([1,2,3,4,5], dtype=float), 3))"
```

If all three tests pass without errors, TA-Lib is properly installed!

## Alternative: Use Conda Environment

If you're still having issues, try using conda:

```cmd
# Install Miniconda if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create conda environment
conda create -n trading python=3.11
conda activate trading

# Install TA-Lib via conda
conda install -c conda-forge ta-lib

# Install other packages
pip install -r requirements.txt
```

## Directory Structure After Installation

Your TA-Lib installation should look like this:
```
C:\ta-lib\
├── c\
│   ├── bin\           <- This should be in PATH
│   │   ├── ta_lib.dll
│   │   └── ...
│   ├── include\       <- This should be in PATH
│   │   ├── ta_libc.h
│   │   └── ...
│   └── lib\
│       └── ...
└── ...
```

## Quick Commands Summary

```cmd
# 1. Download and run: ta-lib-0.4.0-windows.exe
# 2. Open NEW command prompt
cd C:\projects\ai-crypto-trader-v3
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install TA-Lib
pip install -r requirements.txt

# Test installation
python -c "import talib; print('TA-Lib installed successfully!')"

# Run the bot
python src\main.py
```

If you follow these steps exactly, TA-Lib should install without issues on Windows!
