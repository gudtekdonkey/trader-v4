# Hyperliquid Trading Bot

An advanced cryptocurrency trading bot specifically designed for Hyperliquid DEX, featuring state-of-the-art machine learning models, comprehensive risk management, and multiple trading strategies.

## 🚀 Features

### Machine Learning Models
- **Temporal Fusion Transformer (TFT)** for interpretable time series forecasting
- **Attention-LSTM** with self-attention mechanisms
- **Ensemble Learning** combining deep learning and gradient boosting
- **Market Regime Detection** using Hidden Markov Models

### Trading Strategies
- **Momentum Trading** with adaptive parameters
- **Mean Reversion** using statistical arbitrage
- **Market Making** with inventory management
- **Arbitrage** (triangular, statistical, funding rate)

### Risk Management
- Maximum 2% risk per trade
- 20% maximum drawdown limit
- Dynamic position sizing (Kelly Criterion, Volatility-based)
- Correlation-aware portfolio management
- Real-time VaR and CVaR calculations

### Infrastructure
- Real-time WebSocket data streaming
- Redis for high-performance caching
- PostgreSQL for historical data
- Docker containerization
- Prometheus + Grafana monitoring

## 📊 Performance Targets

- **Monthly Returns**: 60-120% (in favorable conditions)
- **Sharpe Ratio**: > 3
- **Max Drawdown**: < 20%
- **Win Rate**: > 70%

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.11 (3.11 recommended)
- Docker & Docker Compose (optional)
- Redis
- PostgreSQL

### Step 1: Install TA-Lib

TA-Lib is required for technical analysis calculations. Choose the method for your operating system:

#### Windows Installation (Using Executable - Recommended):

1. **Download and run the Windows installer:**
   - Visit: https://ta-lib.org/install/
   - Click "Download ta-lib-0.4.0-windows.exe"
   - Run the installer as Administrator
   - Install to the default location (usually `C:\ta-lib`)

2. **Verify installation:**
   - The installer should automatically add TA-Lib to your system PATH
   - Open a new Command Prompt and verify:
   ```cmd
   echo %PATH%
   ```
   - You should see `C:\ta-lib\c\bin` in the PATH

3. **If PATH wasn't added automatically:**
   - Open Windows Settings → System → Advanced system settings
   - Click "Environment Variables"
   - In "System Variables", find and select "Path", then click "Edit"
   - Click "New" and add: `C:\ta-lib\c\bin`
   - Click "New" again and add: `C:\ta-lib\c\include`
   - Click OK to close all dialogs

#### Alternative Windows Methods:

**Option A: Manual Download (if executable doesn't work)**
1. Download `ta-lib-0.4.0-msvc.zip` from https://ta-lib.org/hdr_dw.html
2. Extract to `C:\ta-lib\`
3. Add `C:\ta-lib\c\bin` to your PATH manually

**Option B: Using conda**
```bash
conda install -c conda-forge ta-lib
```

#### Linux Installation:
```bash
# Download and compile from source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
```

#### macOS Installation:
```bash
# Using Homebrew
brew install ta-lib

# Or using MacPorts
sudo port install ta-lib
```

### Step 2: Clone Repository
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

### Step 3: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 4: Install Python Dependencies

```bash
# Test TA-Lib installation first
python -c "import ctypes; ctypes.CDLL('ta_lib')"

# If no error, install Python packages
pip install -r requirements.txt
```

**If TA-Lib Python package installation fails:**

Try these solutions in order:

1. **Install from pre-compiled wheel (Windows):**
   ```bash
   pip install --find-links=https://download.lfd.uci.edu/pythonlibs/archived/ TA-Lib
   ```

2. **Install specific wheel for your Python version:**
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - Download appropriate wheel (e.g., `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`)
   - Install: `pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`

3. **Force reinstall with correct paths:**
   ```bash
   pip install TA-Lib --force-reinstall --no-cache-dir
   ```

4. **Install with conda instead:**
   ```bash
   conda install -c conda-forge ta-lib
   ```

### Step 5: Verify TA-Lib Installation

Test that everything is working:
```python
python -c "import talib; print('TA-Lib version:', talib.__version__)"
```

You should see output like: `TA-Lib version: 0.4.28`

### Step 6: Configuration

1. **Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Update .env file:**
   ```env
   HYPERLIQUID_PRIVATE_KEY=your_private_key_here
   REDIS_URL=redis://localhost:6379
   DATABASE_URL=postgresql://user:pass@localhost/trading
   INITIAL_CAPITAL=100000
   ```

3. **Trading Configuration:**
   Edit `configs/config.yaml` to customize:
   - Trading pairs
   - Strategy parameters
   - Risk limits
   - Model settings

### Step 7: Run the Bot

**Option A: Direct Python**
```bash
python src/main.py
```

**Option B: Docker (Recommended for Production)**
```bash
docker-compose up -d
```

## 🔧 Configuration

### Trading Parameters (`configs/config.yaml`)
```yaml
trading:
  initial_capital: 100000
  max_positions: 10
  risk_per_trade: 0.02  # 2% risk per trade
  symbols:
    - BTC-USD
    - ETH-USD
    - SOL-USD

strategies:
  momentum:
    enabled: true
    weight: 0.3
  mean_reversion:
    enabled: true
    weight: 0.3
  arbitrage:
    enabled: true
    weight: 0.2
  market_making:
    enabled: true
    weight: 0.2
```

### Environment Variables (`.env`)
```env
# Required
HYPERLIQUID_PRIVATE_KEY=your_private_key_here

# Optional
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📈 Dashboard

Access the web dashboard at `http://localhost:5000` to monitor:
- Real-time P&L and performance metrics
- Open positions and trade history
- Risk metrics and drawdown
- Equity curve visualization
- Strategy performance breakdown

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_strategies.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

### Project Structure
```
crypto-trading-bot/
├── src/
│   ├── models/          # ML models (LSTM, TFT, Ensemble)
│   ├── data/           # Data collection & processing
│   ├── trading/        # Trading strategies & execution
│   │   └── strategies/ # Individual strategy implementations
│   ├── exchange/       # Hyperliquid client
│   └── utils/          # Configuration, logging, database
├── dashboard/          # Web interface
├── tests/             # Unit tests
├── configs/           # Configuration files
├── notebooks/         # Research notebooks
└── docker/            # Docker configuration
```

### Key Components

1. **Data Pipeline**: 
   - Real-time WebSocket data collection
   - Feature engineering with 100+ indicators
   - Market microstructure analysis

2. **ML Models**: 
   - Temporal Fusion Transformer for multi-horizon forecasting
   - Attention-LSTM for pattern recognition
   - Ensemble methods combining multiple approaches

3. **Risk Management**: 
   - Kelly Criterion position sizing
   - Dynamic drawdown protection
   - Correlation-aware portfolio optimization

4. **Execution Engine**: 
   - Smart order routing with TWAP/Iceberg algorithms
   - Slippage protection and latency optimization
   - Atomic execution for arbitrage opportunities

## 🚨 Troubleshooting

### Common TA-Lib Issues

1. **"Cannot open include file: 'ta_libc.h'" Error**
   - Make sure you installed the TA-Lib C library using the Windows executable
   - Verify `C:\ta-lib\c\include` is in your PATH
   - Restart your command prompt after installation

2. **"ta_lib.dll not found" Error**
   - Ensure `C:\ta-lib\c\bin` is in your system PATH
   - Try running: `set PATH=%PATH%;C:\ta-lib\c\bin`
   - Restart your command prompt

3. **Python package installation fails**
   ```bash
   # Try installing from wheel
   pip install --find-links=https://download.lfd.uci.edu/pythonlibs/archived/ TA-Lib
   
   # Or use conda
   conda install -c conda-forge ta-lib
   ```

4. **Import errors in Python**
   ```python
   # Test the C library
   import ctypes
   ctypes.CDLL('ta_lib')  # Should not raise an error
   
   # Test Python package
   import talib
   print(talib.__version__)
   ```

### Other Common Issues

1. **Database Connection Issues**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Check PostgreSQL connection
   psql -h localhost -U trader -d trading
   ```

2. **Memory Issues with Large Models**
   - Reduce batch sizes in configs
   - Use CPU instead of GPU for development
   - Enable model checkpointing

3. **WebSocket Connection Errors**
   - Check internet connection
   - Verify Hyperliquid API status
   - Implement exponential backoff retry logic

### Performance Optimization

1. **Speed up data processing:**
   ```python
   # Use vectorized operations
   df['feature'] = np.where(condition, value1, value2)
   
   # Parallel processing
   from multiprocessing import Pool
   ```

2. **Reduce memory usage:**
   ```python
   # Use efficient data types
   df['price'] = df['price'].astype('float32')
   
   # Clear unused variables
   del large_dataframe
   gc.collect()
   ```

## ⚠️ Disclaimer

**Important:** This bot is for educational and research purposes. Cryptocurrency trading carries significant risk of loss. 

- **Never risk more than you can afford to lose**
- **Always test thoroughly with small amounts first**
- **Past performance does not guarantee future results**
- **The authors are not responsible for any financial losses**

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hyperliquid](https://hyperliquid.xyz/) team for the excellent DEX infrastructure
- [TA-Lib](https://ta-lib.org/) for technical analysis functions
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- All contributors to the open-source libraries used in this project

## 📞 Support

- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Security**: Report security vulnerabilities privately via email

---

**Happy Trading! 🚀**
