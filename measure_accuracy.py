import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Re-using the logic from the main script for consistency
def create_lagged_features(data, window_size=3):
    X = []
    n_rows = len(data)
    if n_rows <= window_size:
        return np.array([]), np.array([])
        
    for i in range(window_size, n_rows):
        window = data.iloc[i-window_size:i].values.flatten()
        X.append(window)
        
    return np.array(X)

def backtest_accuracy(file_path, n_tests=10):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    sign_df = (df > 0).astype(int)
    
    WINDOW_SIZE = 3
    total_rows = len(sign_df)
    
    if total_rows < n_tests + WINDOW_SIZE + 10:
        print("Not enough data to backtest.")
        return

    print(f"\nEvaluating performance on the last {n_tests} actual results (Window={WINDOW_SIZE}, Model=RF)...")
    print("-" * 60)
    
    correct_predictions = 0
    start_test_idx = total_rows - n_tests
    
    for current_row_idx in range(start_test_idx, total_rows):
        # 1. Define Training Data (History up to current_row_idx - 1)
        history_df = sign_df.iloc[:current_row_idx]
        
        # 2. Define Actual Outcome
        actual_row = sign_df.iloc[current_row_idx]
        actual_positives = actual_row[actual_row == 1].index.tolist()
        
        if not actual_positives:
            continue
            
        # 3. Train Model
        X_all = create_lagged_features(history_df, WINDOW_SIZE)
        last_window = history_df.iloc[-WINDOW_SIZE:].values.flatten().reshape(1, -1)
        
        probabilities = {}
        
        from sklearn.ensemble import RandomForestClassifier
        
        for i, col_name in enumerate(history_df.columns):
            y_col = history_df.iloc[WINDOW_SIZE:, i].values
            
            unique_classes = np.unique(y_col)
            if len(unique_classes) < 2:
                probabilities[col_name] = 1.0 if unique_classes[0] == 1 else 0.0
                continue
            
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_all, y_col)
            
            prob = model.predict_proba(last_window)[0]
            if len(model.classes_) == 2:
                probabilities[col_name] = prob[1]
            else:
                probabilities[col_name] = 1.0 if model.classes_[0] == 1 else 0.0
        
        # 4. Check Prediction
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_predicted_col = sorted_probs[0][0]
        
        is_correct = top_predicted_col in actual_positives
        if is_correct:
            correct_predictions += 1
            result_str = "SUCCESS"
        else:
            result_str = "FAIL"
            
        print(f"Row {current_row_idx+1}: Pred: {top_predicted_col} | Actual: {actual_positives} | Result: {result_str}")

    accuracy = (correct_predictions / n_tests) * 100
    print("-" * 60)
    print(f"Accuracy Rate: {correct_predictions}/{n_tests} ({accuracy:.2f}%)")
    print("-" * 60)

if __name__ == "__main__":
    file_path = r'c:\laragon\www\excel\USDJPY_patterns_rectangular.csv'
    backtest_accuracy(file_path, n_tests=10)
