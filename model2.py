from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data import display_candles, get_candles
from ta.volatility import average_true_range
from imblearn.over_sampling import SMOTE
from ta.trend import sma_indicator
from ta.momentum import rsi
from ta.trend import macd
import pandas as pd


df = pd.DataFrame(display_candles(get_candles('BTCUSDT', '4h', 1500)))
# df['time'] = pd.to_datetime(df['time'])
# df = df.drop(columns=['time'])
df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)

# Feature Engineering: Add Technical Indicators
df['sma_10'] = sma_indicator(df['close'], window=10)  # 10-period Simple Moving Average
df['rsi'] = rsi(df['close'], window=14)  # Relative Strength Index
df['macd'] = macd(df['close'])  # MACD
df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=14)  # Average True Range

# Add Lag Features
df['lag_close_1'] = df['close'].shift(1)
df['lag_close_2'] = df['close'].shift(2)
df['lag_rsi'] = df['rsi'].shift(1)

# Add Rolling Statistics
df['rolling_mean'] = df['close'].rolling(window=5).mean()
df['rolling_std'] = df['close'].rolling(window=5).std()

# Drop rows with NaN values caused by lag or rolling operations
df = df.dropna()

# Target Variable: Predict whether the price will go up (1) or down (0)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop rows with undefined target values
df = df.dropna()

# Prepare Features (X) and Target (y)
X = df[['open', 'high', 'low', 'close', 'sma_10', 'rsi', 'macd', 'atr', 'lag_close_1', 'lag_close_2', 'lag_rsi', 'rolling_mean', 'rolling_std']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Handle Class Imbalance using SMOTE
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train_balanced, y_train_balanced)

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)

# Print Results
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
