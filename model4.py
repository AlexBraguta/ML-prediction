from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from ta.volatility import average_true_range, bollinger_mavg, bollinger_hband, bollinger_lband
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import rsi, stoch, stoch_signal
from data import display_candles, get_candles
from imblearn.over_sampling import SMOTE
from ta.trend import sma_indicator, macd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


df = pd.DataFrame(display_candles(get_candles('BTCUSDT', '4h', 1500)))
# df['time'] = pd.to_datetime(df['time'])
# df = df.drop(columns=['time'])
df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)


# Feature Engineering: Add Technical Indicators
df['sma_10'] = sma_indicator(df['close'], window=10)
df['rsi'] = rsi(df['close'], window=14)
df['macd'] = macd(df['close'])
df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=14)
df['stoch'] = stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
df['stoch_signal'] = stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
df['bb_mavg'] = bollinger_mavg(df['close'], window=20)
df['bb_hband'] = bollinger_hband(df['close'], window=20)
df['bb_lband'] = bollinger_lband(df['close'], window=20)

# Interaction and Trend Reversal Features
df['atr_rsi_interaction'] = df['atr'] * df['rsi']
df['close_macd_diff'] = df['close'] - df['macd']
df['rsi_change'] = df['rsi'] - df['rsi'].shift(1)
df['macd_change'] = df['macd'] - df['macd'].shift(1)

# Add Lag Features
df['lag_close_1'] = df['close'].shift(1)
df['lag_close_2'] = df['close'].shift(2)

# Add Rolling Statistics
df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
df['rolling_std_20'] = df['close'].rolling(window=20).std()
df['rolling_momentum_20'] = df['close'] - df['close'].shift(20)

# Drop rows with NaN values caused by lag or rolling operations
df = df.dropna()

# Target Variable: Predict whether the price will go up (1) or down (0)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop rows with undefined target values
df = df.dropna()

# Prepare Features (X) and Target (y)
X = df.drop(columns=['time', 'target'])
y = df['target']

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

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.show()

# Adjust Decision Threshold (Optional)
threshold = 0.6
y_pred_custom = (y_pred_proba > threshold).astype(int)
custom_accuracy = accuracy_score(y_test, y_pred_custom)
print(f"Custom Threshold Accuracy (0.6): {custom_accuracy:.2f}")

# Optional: Save the model
joblib.dump(best_model, 'best_random_forest_model.pkl')
