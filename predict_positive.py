import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def create_lagged_features(data, window_size=3):
    """
    Creates lagged features from the dataframe.
    X[i] will contain flattened data from i-window_size to i-1.
    """
    X = []
    # We can only predict starting from `window_size`
    # We want to enable prediction for the very next step after the dataset ends.
    # So we prepare training data up to the last available row.
    
    n_rows = len(data)
    if n_rows <= window_size:
        return np.array([]), np.array([])
        
    for i in range(window_size, n_rows):
        window = data.iloc[i-window_size:i].values.flatten()
        X.append(window)
        
    return np.array(X)

def predict_next_positive_column(file_path):
    # 1. Loading Data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Using only 0 and 1
    sign_df = (df > 0).astype(int)
    
    # Configuration
    WINDOW_SIZE = 3  # Look back 3 steps
    
    if len(sign_df) < WINDOW_SIZE + 1:
        print("Not enough data rows to build history.")
        return None

    # 2. Prepare Training Data
    # X_train: inputs from history
    # y_train: target values (each column will be a target)
    
    X_all = create_lagged_features(sign_df, WINDOW_SIZE)
    # X_all corresponds to targets starting at index `WINDOW_SIZE`
    
    # We want to predict for the specific NEXT row (which doesn't exist yet)
    # The input for that prediction is the LAST `WINDOW_SIZE` rows of the dataset.
    last_window = sign_df.iloc[-WINDOW_SIZE:].values.flatten().reshape(1, -1)
    
    probabilities = {}
    print(f"Analyzing patterns using a {WINDOW_SIZE}-step window on {len(df)} rows...")
    print("Training improved Random Forest models...")

    for i, col_name in enumerate(sign_df.columns):
        # Target: The value at the step corresponding to the X features
        # X_all[k] was formed from rows [k:k+WINDOW_SIZE]. 
        # The target for X_all[k] is row [k+WINDOW_SIZE].
        
        y_col = sign_df.iloc[WINDOW_SIZE:, i].values
        
        # Ensure sizes match
        # X_all length: N - WINDOW_SIZE
        # y_col length: N - WINDOW_SIZE
        # Perfect.
        
        # Check class diversity
        unique_classes = np.unique(y_col)
        if len(unique_classes) < 2:
            probabilities[col_name] = 1.0 if unique_classes[0] == 1 else 0.0
            continue
            
        # Model Training - Increased complexity
        model = RandomForestClassifier(
            n_estimators=200,      # More trees
            max_depth=10,          # Prevent over-fitting on small data
            min_samples_split=5,   # Regularization
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_all, y_col)
        
        # 3. Predict Next Step
        prob = model.predict_proba(last_row)[0] if 'last_row' in locals() else model.predict_proba(last_window)[0]
        
        if len(model.classes_) == 2:
            probabilities[col_name] = prob[1] # Probability of class 1 (Positive)
        else:
            # Should be handled by unique_classes check, but just in case
            probabilities[col_name] = 1.0 if model.classes_[0] == 1 else 0.0

    # 4. Sort results
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs


if __name__ == "__main__":
    file_path = r'c:\laragon\www\excel\USDJPY_patterns_rectangular.csv'
    
    try:
        results = predict_next_positive_column(file_path)

        if results:
            print("\n" + "="*55)
            print("--- Top 5 Columns Predicted to be Positive Next ---")
            print("="*55)
            for i in range(min(5, len(results))):
                col_name, prob = results[i]
                print(f"Rank {i+1}: Column ({col_name}) | Probability: {prob*100:.2f}%")

            top_col = results[0][0]
            top_prob = results[0][1]
            print("\n" + "-"*55)
            print(f"Final Decision: Column [{top_col}] is most likely at {top_prob*100:.2f}%")
            print("-"*55)

    except Exception as e:
        print(f"Error during execution: {e}")
