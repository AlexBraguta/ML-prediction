from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from ta.volatility import average_true_range, bollinger_mavg, bollinger_hband, bollinger_lband
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from ta.momentum import rsi, stoch, stoch_signal
from data import display_candles, get_candles
from imblearn.over_sampling import SMOTE
from ta.trend import sma_indicator, macd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# Load your data (replace 'your_file.csv' with your actual data loading logic)
df_4h = pd.DataFrame(display_candles(get_candles('BTCUSDT', '4h', 1500)))  # 4-hour candles
df_1h = pd.DataFrame(display_candles(get_candles('BTCUSDT', '1h', 1500)))  # 1-hour candles
df_1d = pd.DataFrame(display_candles(get_candles('BTCUSDT', '1d', 1500)))  # 1-day candles
df_2h = pd.DataFrame(display_candles(get_candles('BTCUSDT', '2h', 1500)))  # 2-hour candles
df_12h = pd.DataFrame(display_candles(get_candles('BTCUSDT', '12h', 1500)))  # 12-hour candles

# Convert time columns to datetime
df_4h['time'] = pd.to_datetime(df_4h['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df_1h['time'] = pd.to_datetime(df_1h['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df_1d['time'] = pd.to_datetime(df_1d['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df_2h['time'] = pd.to_datetime(df_2h['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df_12h['time'] = pd.to_datetime(df_12h['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)

# Merge all timeframes
merged_df = df_4h.merge(df_1h, on='time', suffixes=('', '_1h'))
merged_df = merged_df.merge(df_1d, on='time', suffixes=('', '_1d'))
merged_df = merged_df.merge(df_2h, on='time', suffixes=('', '_2h'))
merged_df = merged_df.merge(df_12h, on='time', suffixes=('', '_12h'))

# Feature Engineering: Add Technical Indicators for 4-hour candles
merged_df['sma_10'] = sma_indicator(merged_df['close'], window=10)
merged_df['rsi'] = rsi(merged_df['close'], window=14)
merged_df['macd'] = macd(merged_df['close'])
merged_df['atr'] = average_true_range(merged_df['high'], merged_df['low'], merged_df['close'], window=14)
merged_df['stoch'] = stoch(merged_df['high'], merged_df['low'], merged_df['close'], window=14, smooth_window=3)
merged_df['stoch_signal'] = stoch_signal(merged_df['high'], merged_df['low'], merged_df['close'], window=14, smooth_window=3)
merged_df['bb_mavg'] = bollinger_mavg(merged_df['close'], window=20)
merged_df['bb_hband'] = bollinger_hband(merged_df['close'], window=20)
merged_df['bb_lband'] = bollinger_lband(merged_df['close'], window=20)

# Add Features from 1-Hour Timeframe
merged_df['sma_10_1h'] = sma_indicator(merged_df['close_1h'], window=10)
merged_df['rsi_1h'] = rsi(merged_df['close_1h'], window=14)
merged_df['atr_1h'] = average_true_range(merged_df['high_1h'], merged_df['low_1h'], merged_df['close_1h'], window=14)

# Add Features from 2-Hour Timeframe
merged_df['sma_10_2h'] = sma_indicator(merged_df['close_2h'], window=10)
merged_df['rsi_2h'] = rsi(merged_df['close_2h'], window=14)
merged_df['atr_2h'] = average_true_range(merged_df['high_2h'], merged_df['low_2h'], merged_df['close_2h'], window=14)

# Add Features from 12-Hour Timeframe
merged_df['sma_10_12h'] = sma_indicator(merged_df['close_12h'], window=10)
merged_df['rsi_12h'] = rsi(merged_df['close_12h'], window=14)
merged_df['atr_12h'] = average_true_range(merged_df['high_12h'], merged_df['low_12h'], merged_df['close_12h'], window=14)

# Add Features from 1-Day Timeframe
merged_df['sma_10_1d'] = sma_indicator(merged_df['close_1d'], window=10)
merged_df['rsi_1d'] = rsi(merged_df['close_1d'], window=14)
merged_df['atr_1d'] = average_true_range(merged_df['high_1d'], merged_df['low_1d'], merged_df['close_1d'], window=14)

# Interaction and Trend Reversal Features
merged_df['atr_rsi_interaction'] = merged_df['atr'] * merged_df['rsi']
merged_df['close_macd_diff'] = merged_df['close'] - merged_df['macd']
merged_df['rsi_change'] = merged_df['rsi'] - merged_df['rsi'].shift(1)
merged_df['macd_change'] = merged_df['macd'] - merged_df['macd'].shift(1)

# Add Lag Features
merged_df['lag_close_1'] = merged_df['close'].shift(1)
merged_df['lag_close_2'] = merged_df['close'].shift(2)

# Add Rolling Statistics
merged_df['rolling_mean_20'] = merged_df['close'].rolling(window=20).mean()
merged_df['rolling_std_20'] = merged_df['close'].rolling(window=20).std()
merged_df['rolling_momentum_20'] = merged_df['close'] - merged_df['close'].shift(20)

# Add Aggregated Features from Higher Timeframes
merged_df['daily_sma'] = merged_df['close'].rolling(window=6).mean()  # 6 x 4h candles = 1 day
merged_df['daily_volatility'] = merged_df['close'].rolling(window=6).std()

# Add Exponentially Weighted Moving Averages (EWMA)
merged_df['ewma_12'] = merged_df['close'].ewm(span=12, adjust=False).mean()
merged_df['ewma_26'] = merged_df['close'].ewm(span=26, adjust=False).mean()
merged_df['ewma_diff'] = merged_df['ewma_12'] - merged_df['ewma_26']

# Drop rows with NaN values caused by lag or rolling operations
merged_df = merged_df.dropna()

# Target Variable: Predict whether the price will go up (1) or down (0)
merged_df['target'] = (merged_df['close'].shift(-1) > merged_df['close']).astype(int)

# Drop rows with undefined target values
merged_df = merged_df.dropna()

# Select Important Features Based on Previous Analysis
important_features = [
    'rsi_change', 'stoch', 'stoch_signal', 'macd_change', 'atr', 'rolling_std_20',
    'rolling_momentum_20', 'rsi', 'macd', 'lag_close_2', 'daily_sma', 'daily_volatility',
    'ewma_12', 'ewma_26', 'ewma_diff',
    'sma_10_1h', 'rsi_1h', 'atr_1h',  # Features from 1-hour timeframe
    'sma_10_2h', 'rsi_2h', 'atr_2h',  # Features from 2-hour timeframe
    'sma_10_12h', 'rsi_12h', 'atr_12h',  # Features from 12-hour timeframe
    'sma_10_1d', 'rsi_1d', 'atr_1d'   # Features from 1-day timeframe
]
X = merged_df[important_features]
y = merged_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Handle Class Imbalance using SMOTE
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Hyperparameter Optimization for Random Forest
param_dist_lr = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist_lr,
                                   n_iter=100,
                                   cv=TimeSeriesSplit(n_splits=3),
                                   scoring='accuracy',
                                   random_state=42,
                                   verbose=2)
random_search.fit(X_train_balanced, y_train_balanced)
best_model = random_search.best_estimator_

# Ensemble with Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=500)
lr_model.fit(X_train_balanced, y_train_balanced)


ensemble_model = VotingClassifier(estimators=[
    ('rf', best_model),
    ('lr', lr_model)
], voting='soft')
ensemble_model.fit(X_train_balanced, y_train_balanced)

# Evaluate the ensemble model on the test set
y_pred = ensemble_model.predict(X_test)
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance from Random Forest
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.show()

# Adjust Decision Threshold
thresholds = np.arange(0.4, 0.61, 0.02)
for threshold in thresholds:
    y_pred_custom = (y_pred_proba > threshold).astype(int)
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    print(f"Threshold: {threshold:.2f}, Accuracy: {custom_accuracy:.2f}")

# Make a Prediction for the Next Candle
latest_data = X.iloc[-1:].copy()  # Get the latest row of features
next_candle_prediction = ensemble_model.predict(latest_data)
next_candle_proba = ensemble_model.predict_proba(latest_data)

if next_candle_prediction[0] == 1:
    print("Prediction: The price is likely to go UP in the next 4 hours.")
else:
    print("Prediction: The price is likely to go DOWN in the next 4 hours.")

print(f"Probability of UP: {next_candle_proba[0][1]:.2f}, Probability of DOWN: {next_candle_proba[0][0]:.2f}")

# Optional: Save the model
joblib.dump(ensemble_model, 'best_ensemble_model.pkl')
