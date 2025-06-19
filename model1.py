from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from ta.volatility import average_true_range, bollinger_mavg, bollinger_hband, bollinger_lband
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from ta.momentum import rsi, stoch, stoch_signal
from sklearn.preprocessing import StandardScaler
from data import display_candles, get_candles
from imblearn.over_sampling import SMOTE
from ta.trend import sma_indicator, macd
from xgboost import XGBClassifier
from colorama import Fore, Style
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
from log import log
import pandas as pd
import numpy as np
from tg import tg
import schedule
import warnings
import joblib
import ast

data = {}

model_on = True

warnings.filterwarnings("ignore", category=UserWarning)


def run_model(token='BTCUSDT'):
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
        merged_df['stoch_signal'] = stoch_signal(merged_df['high'], merged_df['low'], merged_df['close'], window=14,
                                                 smooth_window=3)
        merged_df['bb_mavg'] = bollinger_mavg(merged_df['close'], window=20)
        merged_df['bb_hband'] = bollinger_hband(merged_df['close'], window=20)
        merged_df['bb_lband'] = bollinger_lband(merged_df['close'], window=20)

        # Add Features from 1-Hour Timeframe
        merged_df['sma_10_1h'] = sma_indicator(merged_df['close_1h'], window=10)
        merged_df['rsi_1h'] = rsi(merged_df['close_1h'], window=14)
        merged_df['atr_1h'] = average_true_range(merged_df['high_1h'], merged_df['low_1h'], merged_df['close_1h'],
                                                 window=14)

        # Add Features from 2-Hour Timeframe
        merged_df['sma_10_2h'] = sma_indicator(merged_df['close_2h'], window=10)
        merged_df['rsi_2h'] = rsi(merged_df['close_2h'], window=14)
        merged_df['atr_2h'] = average_true_range(merged_df['high_2h'], merged_df['low_2h'], merged_df['close_2h'],
                                                 window=14)

        # Add Features from 12-Hour Timeframe
        merged_df['sma_10_12h'] = sma_indicator(merged_df['close_12h'], window=10)
        merged_df['rsi_12h'] = rsi(merged_df['close_12h'], window=14)
        merged_df['atr_12h'] = average_true_range(merged_df['high_12h'], merged_df['low_12h'], merged_df['close_12h'],
                                                  window=14)

        # Add Features from 1-Day Timeframe
        merged_df['sma_10_1d'] = sma_indicator(merged_df['close_1d'], window=10)
        merged_df['rsi_1d'] = rsi(merged_df['close_1d'], window=14)
        merged_df['atr_1d'] = average_true_range(merged_df['high_1d'], merged_df['low_1d'], merged_df['close_1d'],
                                                 window=14)

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

    #####################################################
    # Hyperparameter dists for each model
    #####################################################

    param_dist_rf = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # We'll also define a param_dist_lr for Logistic Regression hyperparams (C, penalty, solver)
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

    # Basic example param_dist for xgboost
    param_dist_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    #####################################################
    # Model-specific hyperparam optimization
    #####################################################

    def hyperparam_search_rf(x_train_balanced, y_train_balanced):
        print("\n[RandomForest] Hyperparameter Optimization...")
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_dist_rf,
            n_iter=30,  # fewer for speed
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            verbose=0
        )
        random_search.fit(x_train_balanced, y_train_balanced)
        best_rf = random_search.best_estimator_
        return best_rf

    def hyperparam_search_lr(x_train_balanced, y_train_balanced):
        print("\n[LogisticRegression] Hyperparameter Optimization...")
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        random_search_lr = RandomizedSearchCV(
            estimator=lr,
            param_distributions=param_dist_lr,
            n_iter=10,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            verbose=0
        )
        random_search_lr.fit(x_train_balanced, y_train_balanced)
        best_lr = random_search_lr.best_estimator_
        return best_lr

    def hyperparam_search_xgb(x_train_balanced, y_train_balanced):
        print("\n[XGBoost] Hyperparameter Optimization...")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        random_search_xgb = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist_xgb,
            n_iter=10,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            verbose=0
        )
        random_search_xgb.fit(x_train_balanced, y_train_balanced)
        best_xgb = random_search_xgb.best_estimator_
        return best_xgb

    def evaluate_model(model, x_train_balanced, y_train_balanced, x_test, y_test):
        # Predict on test set
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]

        # Training accuracy
        y_pred_train = model.predict(x_train_balanced)
        train_acc = accuracy_score(y_train_balanced, y_pred_train)

        # Testing metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_pred_proba)

        # Fit status
        fit_status = "well fitted"
        if (train_acc - test_acc) > 0.1:
            fit_status = "overfitted"
        elif test_acc < 0.6:
            fit_status = "underfitted"

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
        # This prints a summary:
        # Model chosen: model name
        # due to results (description)
        # model fittedness: result (description)
        # accuracy 75% (description)
        # f1 75% (description)
        # etc.

        print(f"\nModel chosen: {model_name}")
        print(f"Reason: This model achieved the best performance among the candidates.")

        # Fittedness description
        fs = results['fit_status']
        if fs == 'well fitted':
            print(f"Model Fittedness: {fs} (Balanced training vs. testing performance)")
        elif fs == 'overfitted':
            print(f"Model Fittedness: {fs} (The model might be memorizing training data)")
        else:
            print(f"Model Fittedness: {fs} (The model may not capture data complexity)")

        # Metrics
        print(f"Accuracy: {results['test_acc'] * 100:.2f}% (Proportion of correct predictions on test)")
        print(f"F1 Score: {results['test_f1'] * 100:.2f}% (Harmonic mean of precision and recall)")
        print(f"AUC-ROC: {results['test_roc'] * 100:.2f}% (ROC area under curve)")

        data['Accuracy'] = f"{results['test_acc'] * 100:.2f}"
        data['F1-Score'] = f"{results['test_f1'] * 100:.2f}"
        data['AUC-ROC'] = f"{results['test_roc'] * 100:.2f}"

    def final_prediction_prints(best_results, x_scaled, x):
        # We'll do final classification report, confusion matrix, etc.
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = best_results['y_pred']
        model = best_results['model']
        # we assume x_test, y_test are still known, but let's just do the final descriptions.

        print("\nDetailed Classification Report:")
        # There's not an easy direct reference to y_test here, so let's say we skip or user must pass y_test.

        # We'll do final predictions on the latest row
        latest_data = x_scaled[-1:].copy()
        next_pred = model.predict(latest_data)
        next_proba = model.predict_proba(latest_data)

        if next_pred[0] == 1:
            data['Prediction'] = 'UP'
            data['Probability'] = f"{next_proba[0][1] * 100:.2f}"
            direction = Fore.GREEN + "UP" + Style.RESET_ALL
        else:
            data['Prediction'] = 'DOWN'
            data['Probability'] = f"{next_proba[0][0] * 100:.2f}"
            direction = Fore.RED + "DOWN" + Style.RESET_ALL

        print(f"\nPrediction: The price is likely to go {direction} in the next 4 hours.")
        print(f"Prediction Probability: UP {next_proba[0][1] * 100:.2f}%, DOWN {next_proba[0][0] * 100:.2f}%")

        print(f"\n{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

    def final():
        print("Loading and preparing data...")
        merged_df = load_and_merge_data()
        merged_df = feature_engineering(merged_df)
        merged_df = define_target(merged_df)
        x_scaled, y, x, _ = select_and_scale_features(merged_df)
        x_train_balanced, y_train_balanced, x_test, y_test = split_and_balance_data(x_scaled, y)

        # 1. Train each model with hyperparameter search
        best_lr = hyperparam_search_lr(x_train_balanced, y_train_balanced)
        best_rf = hyperparam_search_rf(x_train_balanced, y_train_balanced)
        best_xgb = hyperparam_search_xgb(x_train_balanced, y_train_balanced)

        # 2. Evaluate each model
        lr_results = evaluate_model(best_lr, x_train_balanced, y_train_balanced, x_test, y_test)
        rf_results = evaluate_model(best_rf, x_train_balanced, y_train_balanced, x_test, y_test)
        xgb_results = evaluate_model(best_xgb, x_train_balanced, y_train_balanced, x_test, y_test)

        # 3. Compare results (choose best by test_acc or any other metric)
        all_results = {
            'LogisticRegression': lr_results,
            'RandomForest': rf_results,
            'XGBoost': xgb_results
        }

        # pick the best by test_acc or maybe f1?
        best_model_name = None
        best_model_results = None
        best_score = -1.0

        for mname, r in all_results.items():
            if r['test_acc'] > best_score:
                best_score = r['test_acc']
                best_model_name = mname
                best_model_results = r

        # 4. Print a structured result for the best model
        print_model_summary(best_model_name, best_model_results)

        # 5. Provide final predictions
        final_prediction_prints(best_model_results, x_scaled, x)

    final()


def compare_data():
    file = open('log.txt')
    content = file.readlines()

    # Put last 2 entries in Strings
    second_last_line = content[-2]
    last_line = content[-1]

    # Remove timeframes from Strings
    second_last_line = second_last_line[22:]
    last_line = last_line[22:]

    # Convert Strings to Dictionaries
    s_last_data = ast.literal_eval(second_last_line)
    last_data = ast.literal_eval(last_line)

    if last_data['Prediction'] != s_last_data['Prediction']:
        tg('+---------------------------------------------')
        tg(f"| Trend changed to: {last_data['Prediction']}")
        tg(f"| Probability: {last_data['Probability']}%")
        tg(f"| Acc: {last_data['Accuracy']}%, F1: {last_data['F1-Score']}%, Auc: {last_data['AUC-ROC']}%")
        tg('+---------------------------------------------')


def main():
    run_model()

    log(data)

    compare_data()


if __name__ == '__main__':
    schedule.every(1).hour.do(main)
    main()

    while model_on:
        schedule.run_pending()
        sleep(300)
