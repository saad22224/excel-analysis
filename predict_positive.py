import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def predict_next_positive_column(file_path):
    # 1. تحميل البيانات
    if not os.path.exists(file_path):
        print(f"الملف غير موجود في المسار: {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    
    # تحويل القيم إلى إشارات (Positive = 1, Negative/Zero = 0)
    sign_df = (df > 0).astype(int)
    
    num_columns = sign_df.shape[1]
    probabilities = {}

    print("Analyzing historical patterns for 84 columns, please wait...")

    # 2. Prepare data for training
    # We use the current row to predict the next row
    X = sign_df.iloc[:-1].values  
    
    for i in range(num_columns):
        y = sign_df.iloc[1:, i].values
        
        # Check if there is diversity in the data for the current column
        if len(np.unique(y)) < 2:
            # If the column is always positive or always negative, give probability based on its only state
            probabilities[df.columns[i]] = 1.0 if y[0] == 1 else 0.0
            continue

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # 3. Predict probability based on the last row
        last_row = sign_df.iloc[-1:].values
        prob = model.predict_proba(last_row)[0]
        
        if len(model.classes_) == 2:
            probabilities[df.columns[i]] = prob[1]
        else:
            probabilities[df.columns[i]] = 1.0 if model.classes_[0] == 1 else 0.0

    # 4. Sort results
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs

if __name__ == "__main__":
    file_path = r'c:\laragon\www\usdt\USDJPY_patterns_rectangular.csv'
    
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
