# ₿ ML-prediction ₿

## Disclaimer ⚠️
This repository is provided for educational and informational purposes only. It does not constitute financial, investment, or trading advice. You should conduct your own due diligence and consult a professional advisor before making any financial decisions.

## Description 📝
This pipeline fetches BTC/USDT candlestick data across 1h, 2h, 4h, 12h and 1d timeframes, computes a variety of technical indicators (`SMA`, `RSI`, `MACD`, `ATR`, `Bollinger Bands`, `EWMA`, etc.), and builds classification models (`Logistic Regression`, `Random Forest`, `XGBoost`) to forecast whether the price will go **UP** or **DOWN** in the next 4 hours. It automatically tunes hyperparameters, evaluates fit (over-/under-fitting), and—if the chosen model is overfitted—iteratively simplifies it up to three times.

## Features ✨
- Multi-timeframe data merge  
- Extensive feature engineering with `ta` indicators  
- SMOTE balancing for imbalanced classes  
- Randomized hyperparameter search with time-series CV  
- Automated overfitting detection & model simplification  
- Final prediction output with probability and timestamp  

## Requirements 📋
- Python ≥ 3.8  
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imblearn`, `ta`, `matplotlib`, `joblib`  
- Access to Binance REST API data  

## Installation 🚀
1. Clone the repo  
   `git clone https://github.com/your-username/ML-prediction.git`  
   `cd ML-prediction`  
2. Create & activate a virtual environment  
   `python -m venv venv`  
   `source venv/bin/activate`  
3. Install dependencies  
   `pip install -r requirements.txt`
4. Binance API Key should be set in Environment

## Usage ▶️
`python main.py`  

Outputs model summaries and a 4-hour forecast at the end, e.g.:  
```
Prediction: The price is likely to go UP in the next 4 hours.  
Prediction Probability: UP 65.34%, DOWN 34.66%  
2025-06-19 13:50:02.123456
```

## License 📄
MIT © 2025

