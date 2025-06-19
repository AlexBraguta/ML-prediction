from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from ta.volatility import average_true_range, bollinger_mavg, bollinger_hband, bollinger_lband
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from ta.momentum import rsi, stoch, stoch_signal
from sklearn.preprocessing import StandardScaler
from data import display_candles, get_candles
from imblearn.over_sampling import SMOTE
from ta.trend import sma_indicator, macd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import joblib


token = 'BTCUSDT'


def load_and_merge_data():
    # Load your data (replace 'your_file.csv' with your actual data loading logic)
    df_1h = pd.DataFrame(display_candles(get_candles(token, '1h', 1500)))  # 1-hour candles
    df_2h = pd.DataFrame(display_candles(get_candles(token, '2h', 1500)))  # 2-hour candles
    df_4h = pd.DataFrame(display_candles(get_candles(token, '4h', 1500)))  # 4-hour candles
    df_12h = pd.DataFrame(display_candles(get_candles(token, '12h', 1500)))  # 12-hour candles
    df_1d = pd.DataFrame(display_candles(get_candles(token, '1d', 1500)))  # 1-day candles

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

    return merged_df


def feature_engineering(merged_df):
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
    return merged_df


def define_target(merged_df):
    # Target Variable: Predict whether the price will go up (1) or down (0)
    merged_df['target'] = (merged_df['close'].shift(-1) > merged_df['close']).astype(int)

    # Drop rows with undefined target values
    merged_df = merged_df.dropna()
    return merged_df


def select_and_scale_features(merged_df):
    # Select Important Features Based on Previous Analysis
    important_features = [
        'rsi_change', 'stoch', 'stoch_signal', 'macd_change', 'atr', 'rolling_std_20',
        'rolling_momentum_20', 'rsi', 'macd', 'lag_close_2', 'daily_sma', 'daily_volatility',
        'ewma_12', 'ewma_26', 'ewma_diff',
        'sma_10_1h', 'rsi_1h', 'atr_1h',
        'sma_10_2h', 'rsi_2h', 'atr_2h',
        'sma_10_12h', 'rsi_12h', 'atr_12h',
        'sma_10_1d', 'rsi_1d', 'atr_1d'
    ]
    x = merged_df[important_features]
    y = merged_df['target']

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y, x, scaler


def split_and_balance_data(x_scaled, y):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=False)

    # Handle Class Imbalance using SMOTE
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

    return x_train_balanced, y_train_balanced, x_test, y_test


# We'll define a global param_dist for Random Forest that can be adjusted in case of over/underfitting
param_dist_rf = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# We'll also define a param_dist_lr for Logistic Regression hyperparams (C, penalty, solver)
# We'll ignore incompatible penalty/solver combos for simplicity; some combos may fail but are caught internally.
param_dist_lr = [
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    },
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    }
]


def hyperparameter_optimization_lr(x_train_balanced, y_train_balanced):
    print("Starting hyperparameter optimization for Logistic Regression...")

    # Create a base logistic regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)

    # We'll do random search on LR hyperparams, ignoring advanced combos
    random_search_lr = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_dist_lr,
        n_iter=10,  # fewer iterations just for demonstration
        cv=TimeSeriesSplit(n_splits=3),
        scoring='accuracy',
        random_state=42,
        verbose=0
    )
    random_search_lr.fit(x_train_balanced, y_train_balanced)

    print("Logistic Regression hyperparameter optimization completed.")
    best_lr = random_search_lr.best_estimator_
    return best_lr


def hyperparameter_optimization_rf(x_train_balanced, y_train_balanced):
    global param_dist_rf
    # Print a single message before fitting
    print("Starting hyperparameter optimization (Random Forest) with current param_dist_rf...")

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist_rf,
        n_iter=50,  # We'll do fewer iterations for quicker re-runs
        cv=TimeSeriesSplit(n_splits=3),
        scoring='accuracy',
        random_state=42,
        verbose=0
    )
    random_search.fit(x_train_balanced, y_train_balanced)

    # Print a single message after fitting
    print("Random Forest hyperparameter optimization completed.")

    best_rf = random_search.best_estimator_
    return best_rf


def build_ensemble_model(best_rf, best_lr, x_train_balanced, y_train_balanced):
    # Build final ensemble using best RF and best LR
    ensemble_model = VotingClassifier(
        estimators=[('rf', best_rf), ('lr', best_lr)],
        voting='soft',
        n_jobs=-1
    )
    ensemble_model.fit(x_train_balanced, y_train_balanced)
    return ensemble_model


def evaluate_model(ensemble_model, x_train_balanced, y_train_balanced, x_test, y_test):
    # Evaluate the ensemble model on the test set
    y_pred = ensemble_model.predict(x_test)
    y_pred_proba = ensemble_model.predict_proba(x_test)[:, 1]

    # Calculate training accuracy to check for overfitting/underfitting
    train_pred = ensemble_model.predict(x_train_balanced)
    train_accuracy = accuracy_score(y_train_balanced, train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Determine if overfitted/underfitted/well fitted
    fit_status = "well fitted"
    if (train_accuracy - test_accuracy) > 0.1:
        fit_status = "overfitted"
    elif test_accuracy < 0.6:
        fit_status = "underfitted"

    return fit_status, y_pred, y_pred_proba, train_accuracy, test_accuracy


def final_prints(ensemble_model, x_test, y_test, y_pred, y_pred_proba, train_accuracy, test_accuracy, x_scaled, x):
    # If well_fitted, we print all results
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # We'll assume the first estimator is the random forest
    importances = ensemble_model.estimators_[0].feature_importances_
    indices = np.argsort(importances)[::-1]
    features = x.columns

    # Compute metrics
    accuracy = test_accuracy
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"AUC-ROC: {roc_auc:.2f}")

    # Make a Prediction for the Next Candle
    latest_data = x_scaled[-1:]  # Get the latest row of features
    next_candle_prediction = ensemble_model.predict(latest_data)
    next_candle_proba = ensemble_model.predict_proba(latest_data)

    if next_candle_prediction[0] == 1:
        print("Prediction: The price is likely to go UP in the next 4 hours.")
    else:
        print("\nPrediction: The price is likely to go DOWN in the next 4 hours.")

    print(f"\nProbability of UP: {next_candle_proba[0][1]:.2f}, Probability of DOWN: {next_candle_proba[0][0]:.2f}")
    print(f'\n {datetime.now()}')

    # Optional: Save the model
    # joblib.dump(ensemble_model, 'best_ensemble_model.pkl')


def main():
    global param_dist_rf
    merged_df = load_and_merge_data()
    merged_df = feature_engineering(merged_df)
    merged_df = define_target(merged_df)
    x_scaled, y, x, _ = select_and_scale_features(merged_df)
    x_train_balanced, y_train_balanced, x_test, y_test = split_and_balance_data(x_scaled, y)

    # First, tune logistic regression once
    best_lr = hyperparameter_optimization_lr(x_train_balanced, y_train_balanced)

    # We'll do a loop for random forest until well fitted or we tried up to 3 times
    max_attempts = 3
    attempt = 0

    while True:
        attempt += 1
        print(f"\n=== Attempt {attempt} (Random Forest) ===")
        best_rf = hyperparameter_optimization_rf(x_train_balanced, y_train_balanced)
        ensemble = build_ensemble_model(best_rf, best_lr, x_train_balanced, y_train_balanced)

        fit_status, y_pred, y_pred_proba, train_acc, test_acc = evaluate_model(
            ensemble, x_train_balanced, y_train_balanced, x_test, y_test
        )

        if fit_status == "well fitted" or attempt == max_attempts:
            # If well_fitted or we are out of attempts, print final results
            if fit_status == "overfitted":
                print("\nModel is still overfitted at final attempt. Printing current results.")
            elif fit_status == "underfitted":
                print("\nModel is still underfitted at final attempt. Printing current results.")
            final_prints(
                ensemble,
                x_test,
                y_test,
                y_pred,
                y_pred_proba,
                train_acc,
                test_acc,
                x_scaled,
                x
            )
            break
        else:
            # If overfitted, let's reduce complexity
            if fit_status == "overfitted":
                print("Overfitted detected. Adjusting param_dist_rf to reduce complexity.")
                # remove None from max_depth if present
                if None in param_dist_rf['max_depth']:
                    param_dist_rf['max_depth'].remove(None)
                # reduce n_estimators upper bound if > 100
                param_dist_rf['n_estimators'] = [n for n in param_dist_rf['n_estimators'] if n <= 100]

            # If underfitted, let's allow more complexity
            if fit_status == "underfitted":
                print("Underfitted detected. Adjusting param_dist_rf to allow more complexity.")
                if None not in param_dist_rf['max_depth']:
                    param_dist_rf['max_depth'].append(None)
                # ensure we have bigger n_estimators
                if 500 not in param_dist_rf['n_estimators']:
                    param_dist_rf['n_estimators'].append(500)
            print("Re-running with updated param_dist_rf...\n")


if __name__ == "__main__":
    main()
