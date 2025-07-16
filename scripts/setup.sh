#!/bin/bash

# Setup script for trader-v4
# This script installs all dependencies and sets up the trading environment

set -e  # Exit on error

echo "================================================"
echo "Trading Bot Setup Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python version $python_version is compatible"
else
    print_error "Python version $python_version is not compatible. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
print_status "pip upgraded"

# Install TA-Lib (platform-specific)
echo "Installing TA-Lib..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! command -v ta-lib-config &> /dev/null; then
        print_warning "TA-Lib not found. Installing..."
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        sudo ldconfig
        cd ..
        rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
        print_status "TA-Lib installed"
    else
        print_status "TA-Lib already installed"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    if ! command -v ta-lib-config &> /dev/null; then
        print_warning "TA-Lib not found. Installing with Homebrew..."
        brew install ta-lib
        print_status "TA-Lib installed"
    else
        print_status "TA-Lib already installed"
    fi
fi

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt
print_status "Python requirements installed"

# Create necessary directories
echo "Creating project directories..."
directories=("logs" "data" "models" "notebooks" "docker" "configs" "backtest_results")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    fi
done

# Setup environment file
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please edit .env file with your configuration"
    else
        print_error ".env.example not found"
    fi
else
    print_warning ".env file already exists"
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_status "Pre-commit hooks installed"
else
    print_warning "pre-commit not found. Skipping hook installation"
fi

# Database setup
echo "Setting up database..."
if command -v psql &> /dev/null; then
    print_status "PostgreSQL is installed"
    print_warning "Please ensure PostgreSQL is running and create the trading database"
else
    print_error "PostgreSQL not found. Please install PostgreSQL"
fi

# Redis setup
echo "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    print_status "Redis is installed"
    print_warning "Please ensure Redis is running"
else
    print_error "Redis not found. Please install Redis"
fi

# Download sample data (optional)
read -p "Do you want to download sample market data? (y/n): " download_data
if [[ $download_data == "y" ]]; then
    echo "Downloading sample data..."
    python scripts/download_sample_data.py
    print_status "Sample data downloaded"
fi

echo ""
echo "================================================"
echo "Setup completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start PostgreSQL and Redis services"
echo "3. Run 'python scripts/init_database.py' to initialize the database"
echo "4. Run 'python src/main.py' to start the trading bot"
echo ""
echo "For Docker setup, run: docker-compose up -d"