"""
Code to include multiple models (Logistic Regression, Random Forest, XGBoost),
pick the best, and if that best model is overfitted, automatically simplify it
until it is well-fitted (or we reach a limit of 3 attempts).
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from ta.volatility import average_true_range, bollinger_mavg, bollinger_hband, bollinger_lband
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
import warnings
import joblib

token = 'BTCUSDT'

warnings.filterwarnings("ignore", category=UserWarning)


def load_and_merge_data():
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
    # Basic 4-hour features
    merged_df['sma_10'] = sma_indicator(merged_df['close'], window=10)
    merged_df['rsi'] = rsi(merged_df['close'], window=14)
    merged_df['macd'] = macd(merged_df['close'])
    merged_df['atr'] = average_true_range(merged_df['high'], merged_df['low'], merged_df['close'], window=14)
    merged_df['stoch'] = stoch(merged_df['high'], merged_df['low'], merged_df['close'], window=14, smooth_window=3)
    merged_df['stoch_signal'] = stoch_signal(merged_df['high'], merged_df['low'], merged_df['close'], window=14, smooth_window=3)
    merged_df['bb_mavg'] = bollinger_mavg(merged_df['close'], window=20)
    merged_df['bb_hband'] = bollinger_hband(merged_df['close'], window=20)
    merged_df['bb_lband'] = bollinger_lband(merged_df['close'], window=20)

    # Additional 4h indicators
    merged_df['rsi_change'] = merged_df['rsi'] - merged_df['rsi'].shift(1)
    merged_df['macd_change'] = merged_df['macd'] - merged_df['macd'].shift(1)
    merged_df['rolling_std_20'] = merged_df['close'].rolling(window=20).std()
    merged_df['rolling_momentum_20'] = merged_df['close'] - merged_df['close'].shift(20)
    merged_df['lag_close_2'] = merged_df['close'].shift(2)

    # daily_sma, daily_volatility (6 x 4h = 24h) or adjust to your timeframe
    merged_df['daily_sma'] = merged_df['close'].rolling(window=6).mean()
    merged_df['daily_volatility'] = merged_df['close'].rolling(window=6).std()

    # EWMA indicators
    merged_df['ewma_12'] = merged_df['close'].ewm(span=12, adjust=False).mean()
    merged_df['ewma_26'] = merged_df['close'].ewm(span=26, adjust=False).mean()
    merged_df['ewma_diff'] = merged_df['ewma_12'] - merged_df['ewma_26']

    # 1-hour timeframe features (assuming merges exist)
    if 'close_1h' in merged_df.columns:
        merged_df['sma_10_1h'] = sma_indicator(merged_df['close_1h'], window=10)
        merged_df['rsi_1h'] = rsi(merged_df['close_1h'], window=14)
        merged_df['atr_1h'] = average_true_range(merged_df['high_1h'], merged_df['low_1h'], merged_df['close_1h'], window=14)

    # 2-hour timeframe
    if 'close_2h' in merged_df.columns:
        merged_df['sma_10_2h'] = sma_indicator(merged_df['close_2h'], window=10)
        merged_df['rsi_2h'] = rsi(merged_df['close_2h'], window=14)
        merged_df['atr_2h'] = average_true_range(merged_df['high_2h'], merged_df['low_2h'], merged_df['close_2h'], window=14)

    # 12-hour timeframe
    if 'close_12h' in merged_df.columns:
        merged_df['sma_10_12h'] = sma_indicator(merged_df['close_12h'], window=10)
        merged_df['rsi_12h'] = rsi(merged_df['close_12h'], window=14)
        merged_df['atr_12h'] = average_true_range(merged_df['high_12h'], merged_df['low_12h'], merged_df['close_12h'], window=14)

    # 1-day timeframe
    if 'close_1d' in merged_df.columns:
        merged_df['sma_10_1d'] = sma_indicator(merged_df['close_1d'], window=10)
        merged_df['rsi_1d'] = rsi(merged_df['close_1d'], window=14)
        merged_df['atr_1d'] = average_true_range(merged_df['high_1d'], merged_df['low_1d'], merged_df['close_1d'], window=14)

    # Drop rows with NaN from the added features
    merged_df.dropna(inplace=True)
    return merged_df


def define_target(merged_df):
    merged_df['target'] = (merged_df['close'].shift(-1) > merged_df['close']).astype(int)
    merged_df = merged_df.dropna()
    return merged_df


def select_and_scale_features(merged_df):
    important_features = [
        'rsi_change','stoch','stoch_signal','macd_change','atr','rolling_std_20',
        'rolling_momentum_20','rsi','macd','lag_close_2','daily_sma','daily_volatility',
        'ewma_12','ewma_26','ewma_diff',
        'sma_10_1h','rsi_1h','atr_1h',
        'sma_10_2h','rsi_2h','atr_2h',
        'sma_10_12h','rsi_12h','atr_12h',
        'sma_10_1d','rsi_1d','atr_1d'
    ]
    x = merged_df[important_features]
    y = merged_df['target']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, x, scaler


def split_and_balance_data(x_scaled, y):
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, shuffle=False)
    smote = SMOTE()
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    return x_train_balanced, y_train_balanced, x_test, y_test


############################################
# Hyperparameter dists
############################################


param_dist_rf = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

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

param_dist_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}


############################################
# Model-specific hyperparam search
############################################


def hyperparam_search_rf(x_train_balanced, y_train_balanced, param_dist):
    print("\n[RandomForest] Hyperparameter Optimization...")
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='accuracy',
        random_state=42,
        verbose=0
    )
    search.fit(x_train_balanced, y_train_balanced)
    return search.best_estimator_


def hyperparam_search_lr(x_train_balanced, y_train_balanced, param_dist):
    print("\n[LogisticRegression] Hyperparameter Optimization...")
    base_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    search = RandomizedSearchCV(
        estimator=base_lr,
        param_distributions=param_dist,
        n_iter=10,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='accuracy',
        random_state=42,
        verbose=0
    )
    search.fit(x_train_balanced, y_train_balanced)
    return search.best_estimator_


def hyperparam_search_xgb(x_train_balanced, y_train_balanced, param_dist):
    print("\n[XGBoost] Hyperparameter Optimization...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='accuracy',
        random_state=42,
        verbose=0
    )
    search.fit(x_train_balanced, y_train_balanced)
    return search.best_estimator_


def evaluate_model(model, x_train_balanced, y_train_balanced, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    train_acc = accuracy_score(y_train_balanced, model.predict(x_train_balanced))
    test_acc  = accuracy_score(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred)
    test_roc  = roc_auc_score(y_test, y_pred_proba)

    # Fit status
    fit_status = 'well fitted'
    if (train_acc - test_acc) > 0.1:
        fit_status = 'overfitted'
    elif test_acc < 0.6:
        fit_status = 'underfitted'

    return {
        'model': model,
        'fit_status': fit_status,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_roc': test_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def print_model_summary(model_name, results):
    print(f"\nModel chosen: {model_name}")
    print(f"Reason: This model achieved the best performance among the candidates.")
    fs = results['fit_status']
    if fs == 'well fitted':
        print(f"Model Fittedness: {fs} (Balanced training vs. testing performance)")
    elif fs == 'overfitted':
        print(f"Model Fittedness: {fs} (The model might be memorizing training data)")
    else:
        print(f"Model Fittedness: {fs} (The model may not capture data complexity)")

    print(f"Accuracy: {results['test_acc']*100:.2f}% (Proportion of correct predictions on test)")
    print(f"F1 Score: {results['test_f1']*100:.2f}% (Harmonic mean of precision and recall)")
    print(f"AUC-ROC: {results['test_roc']*100:.2f}% (ROC area under curve)")


def final_prediction_prints(best_results, x_scaled, x):
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nDetailed Classification Report is omitted here for brevity.")

    model = best_results['model']
    latest_data = x_scaled[-1:].copy()
    next_pred = model.predict(latest_data)
    next_proba = model.predict_proba(latest_data)

    direction = "UP" if next_pred[0] == 1 else "DOWN"
    print(f"\nPrediction: The price is likely to go {direction} in the next 4 hours.")
    print(f"Prediction Probability: UP {next_proba[0][1]*100:.2f}%, DOWN {next_proba[0][0]*100:.2f}%")
    print(f"\n{datetime.now()}")


############################################
# Logic to reduce complexity if overfitted
############################################


def reduce_complexity_rf(param_dist):
    """
    For RandomForest, remove None from max_depth if present,
    reduce n_estimators upper bound to <= 100, etc.
    """
    print("[RandomForest] Reducing complexity param_dist...")
    # remove None from max_depth
    if None in param_dist['max_depth']:
        param_dist['max_depth'].remove(None)
    # reduce n_estimators to <= 100
    param_dist['n_estimators'] = [n for n in param_dist['n_estimators'] if n <= 100]
    # potentially enlarge min_samples_leaf
    if 1 in param_dist['min_samples_leaf']:
        param_dist['min_samples_leaf'].append(2)


def reduce_complexity_lr(param_dist):
    """
    For LogisticRegression, reduce 'C' range to eliminate higher values
    """
    print("[LogisticRegression] Reducing complexity param_dist...")
    for subgrid in param_dist:
        # e.g. keep only lower Cs
        subgrid['C'] = [c for c in subgrid['C'] if c < 10]
        # If we want further simplification, we might remove 'l1' penalty, etc.


def reduce_complexity_xgb(param_dist):
    """
    For XGBoost, reduce max_depth, n_estimators, etc.
    """
    print("[XGBoost] Reducing complexity param_dist...")
    # reduce max_depth if possible
    param_dist['max_depth'] = [md for md in param_dist['max_depth'] if md <= 5]
    # reduce n_estimators
    param_dist['n_estimators'] = [n for n in param_dist['n_estimators'] if n <= 100]
    # reduce learning_rate upper bound
    param_dist['learning_rate'] = [lr for lr in param_dist['learning_rate'] if lr <= 0.1]


def simplify_best_model(best_model_name, param_dist_dict, x_train_balanced, y_train_balanced, x_test, y_test):
    """
    We do up to 3 attempts. If overfitted, we reduce complexity and re-run param search.
    Then evaluate. Stop if not overfitted or attempt limit reached.
    """
    attempts = 0
    max_attempts = 3
    chosen_model = None
    final_results = None

    while attempts < max_attempts:
        attempts += 1
        print(f"\n--- Checking Overfitting for {best_model_name}, Attempt {attempts} ---")

        if best_model_name == 'RandomForest':
            chosen_model = hyperparam_search_rf(x_train_balanced, y_train_balanced, param_dist_dict['rf'])
        elif best_model_name == 'LogisticRegression':
            chosen_model = hyperparam_search_lr(x_train_balanced, y_train_balanced, param_dist_dict['lr'])
        else:  # 'XGBoost'
            chosen_model = hyperparam_search_xgb(x_train_balanced, y_train_balanced, param_dist_dict['xgb'])

        tmp_results = evaluate_model(chosen_model, x_train_balanced, y_train_balanced, x_test, y_test)
        print(f"Fit status after search: {tmp_results['fit_status']}, train_acc={tmp_results['train_acc']:.3f}, test_acc={tmp_results['test_acc']:.3f}")

        if tmp_results['fit_status'] == 'overfitted':
            # reduce complexity
            if best_model_name == 'RandomForest':
                reduce_complexity_rf(param_dist_dict['rf'])
            elif best_model_name == 'LogisticRegression':
                reduce_complexity_lr(param_dist_dict['lr'])
            else:
                reduce_complexity_xgb(param_dist_dict['xgb'])
            if attempts == max_attempts:
                print("[INFO] Still overfitted after max attempts, using it anyway.")
                final_results = tmp_results
                break
        else:
            # well fitted or underfitted => we accept it
            final_results = tmp_results
            break

    return final_results


def main():
    print("Loading and preparing data...")
    merged_df = load_and_merge_data()
    merged_df = feature_engineering(merged_df)
    merged_df = define_target(merged_df)
    x_scaled, y, x, _ = select_and_scale_features(merged_df)
    x_train_balanced, y_train_balanced, x_test, y_test = split_and_balance_data(x_scaled, y)

    # 1. Single-pass hyperparam search for each model:
    print("\n--- Initial Single-Pass Hyperparam Search for each model ---")
    lr_model = hyperparam_search_lr(x_train_balanced, y_train_balanced, param_dist_lr)
    rf_model = hyperparam_search_rf(x_train_balanced, y_train_balanced, param_dist_rf)
    xgb_model = hyperparam_search_xgb(x_train_balanced, y_train_balanced, param_dist_xgb)

    # 2. Evaluate each model
    lr_results = evaluate_model(lr_model, x_train_balanced, y_train_balanced, x_test, y_test)
    rf_results = evaluate_model(rf_model, x_train_balanced, y_train_balanced, x_test, y_test)
    xgb_results = evaluate_model(xgb_model, x_train_balanced, y_train_balanced, x_test, y_test)

    all_results = {
        'LogisticRegression': lr_results,
        'RandomForest': rf_results,
        'XGBoost': xgb_results
    }

    # Pick best by test_acc
    best_model_name = None
    best_score = -1
    best_model_results = None
    for mname, r in all_results.items():
        if r['test_acc'] > best_score:
            best_score = r['test_acc']
            best_model_name = mname
            best_model_results = r

    # If best model is overfitted, we will do iterative approach
    if best_model_results['fit_status'] == 'overfitted':
        print(f"\n[INFO] Best model is {best_model_name} but overfitted. We will reduce complexity and re-run.")
        # We'll keep param distributions in a dictionary
        param_dict = {
            'LogisticRegression': param_dist_lr,
            'RandomForest': param_dist_rf,
            'XGBoost': param_dist_xgb
        }
        final_results = simplify_best_model(
            best_model_name,
            {
                'lr': param_dist_lr,
                'rf': param_dist_rf,
                'xgb': param_dist_xgb
            },
            x_train_balanced,
            y_train_balanced,
            x_test,
            y_test
        )
        best_model_results = final_results
    else:
        print(f"\n[INFO] Best model is {best_model_name} and is not overfitted. No need to re-run.")

    # 4. Print structured result
    print_model_summary(best_model_name, best_model_results)
    final_prediction_prints(best_model_results, x_scaled, x)


if __name__ == "__main__":
    main()
