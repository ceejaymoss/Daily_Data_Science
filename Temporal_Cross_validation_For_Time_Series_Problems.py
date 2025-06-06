import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def time_series_cv(X, y, n_splits=5, test_size=30, gap=5, min_train_size=100):
    """
    Time series cross-validation with gap to prevent data leakage
    
    Parameters:
    - X, y: Features and target (must be time-ordered)
    - n_splits: Number of CV folds
    - test_size: Size of each test set
    - gap: Gap between train/test to prevent leakage
    - min_train_size: Minimum training set size
    """
    n_samples = len(X)
    scores = []
    fold_info = []
    
    # Calculate fold boundaries
    max_test_start = n_samples - test_size
    min_test_start = min_train_size + gap
    
    test_starts = np.linspace(min_test_start, max_test_start, n_splits, dtype=int)
    
    for i, test_start in enumerate(test_starts):
        # Define splits with gap
        train_end = test_start - gap
        test_end = test_start + test_size
        
        if train_end < min_train_size:
            continue
            
        # Create train/test indices
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n_samples))
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model and predict
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate score
        score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
        scores.append(score)
        
        fold_info.append({
            'fold': i+1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_period': (train_idx[0], train_idx[-1]),
            'test_period': (test_idx[0], test_idx[-1]),
            'rmse': score
        })
    
    return scores, fold_info

# Example usage with sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')

# Create sample time series with trend and seasonality
trend = np.linspace(0, 10, 500)
seasonal = 3 * np.sin(2 * np.pi * np.arange(500) / 365.25)
noise = np.random.normal(0, 1, 500)
target = trend + seasonal + noise

# Create lagged features
df = pd.DataFrame({
    'date': dates,
    'value': target,
    'lag_1': np.roll(target, 1),
    'lag_7': np.roll(target, 7),
    'lag_30': np.roll(target, 30),
    'rolling_mean_7': pd.Series(target).rolling(7).mean(),
    'rolling_std_7': pd.Series(target).rolling(7).std()
})

# Remove rows with NaN from rolling features
df = df.dropna().reset_index(drop=True)
X = df[['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']]
y = df['value']

# Perform time series cross-validation
scores, fold_info = time_series_cv(X, y, n_splits=5, test_size=30, gap=5)

# Display results
print("Time Series Cross-Validation Results:")
print(f"Average RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
print("\nFold Details:")
for info in fold_info:
    print(f"Fold {info['fold']}: Train size={info['train_size']}, "
          f"Test size={info['test_size']}, RMSE={info['rmse']:.3f}")

# Visualize the CV splits
plt.figure(figsize=(12, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, info in enumerate(fold_info[:3]):  # Show first 3 folds
    train_start, train_end = info['train_period']
    test_start, test_end = info['test_period']
    
    plt.barh(i, train_end - train_start, left=train_start, 
             color=colors[i], alpha=0.3, label=f'Fold {i+1} Train')
    plt.barh(i, test_end - test_start, left=test_start, 
             color=colors[i], alpha=0.8, label=f'Fold {i+1} Test')

plt.xlabel('Time Index')
plt.ylabel('CV Fold')
plt.title('Time Series Cross-Validation Splits (First 3 Folds)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Compare with naive random CV (BAD practice)
from sklearn.model_selection import cross_val_score
naive_scores = cross_val_score(LinearRegression(), X, y, cv=5, 
                              scoring='neg_mean_squared_error')
naive_rmse = np.sqrt(-naive_scores)

print(f"\n⚠️  Comparison with Random CV (WRONG for time series):")
print(f"Random CV RMSE: {np.mean(naive_rmse):.3f} ± {np.std(naive_rmse):.3f}")
print(f"Time Series CV RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
print(f"Difference: {np.mean(naive_rmse) - np.mean(scores):.3f}")