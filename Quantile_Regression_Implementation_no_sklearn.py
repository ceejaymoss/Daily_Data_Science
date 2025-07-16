# Pure Numpy implementation (works without sklearn)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings.filterwarnings('ignore')

# Linear Regression implementation
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: (X'X)^(-1)X'y
        params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        return self

        def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Polynomial features
def create_polynomial_features(X, degree=2):
    """Create polynomial features up to given degree"""
    n_samples, n_features = X.shape
    if degree == 1:
        return X
    
    # For simplicity, just do degree 2 with single feature
    if n_features == 1:
        return np.column_stack([X, X**2])
    else:
        return X  # Return original if multiple features

class QuantileRegressor:
    def __init__(self, quantile=0.5):
        self.quantile = quantile
        self.coef_ = None
        self.intercept_ = None
    
    def _check_function(self, residuals):
        """Quantile loss function (check function)"""
        return np.sum(np.maximum(self.quantile * residuals, (self.quantile - 1) * residuals))
    
    def fit(self, X, y):
        """Fit quantile regression"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initial guess
        initial_params = np.zeros(X_with_intercept.shape[1])
        
        # Minimize quantile loss
        def objective(params):
            predictions = X_with_intercept @ params
            residuals = y - predictions
            return self._check_function(residuals)
        
        result = minimize(objective, initial_params, method='BFGS')
        
        # Store results
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.coef_ + self.intercept_

# Generate sample data with heteroscedastic noise (varying variance)
np.random.seed(42)
n_samples = 200
X = np.linspace(0, 10, n_samples).reshape(-1, 1)

# Create heteroscedastic noise (variance increases with X)
noise_std = 0.1 + 0.5 * X.flatten()
y = 2 * X.flatten() + np.sin(X.flatten()) + np.random.normal(0, noise_std)

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Define quantiles to predict
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_models = {}

# Train quantile regression models
print("Training Quantile Regression Models...")
for q in quantiles:
    model = QuantileRegressor(quantile=q)
    # Create polynomial features
    X_train_poly = create_polynomial_features(X_train, degree=2)
    model.fit(X_train_poly, y_train)
    quantile_models[q] = model

# Train standard linear regression for comparison
linear_model = LinearRegression()
X_train_poly = create_polynomial_features(X_train, degree=2)
linear_model.fit(X_train_poly, y_train)

# Generate predictions
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
X_plot_poly = create_polynomial_features(X_plot, degree=2)
predictions = {}

for q in quantiles:
    predictions[q] = quantile_models[q].predict(X_plot_poly)

linear_pred = linear_model.predict(X_plot_poly)

# Plotting
plt.figure(figsize=(12, 8))

# Plot original data
plt.scatter(X_train, y_train, alpha=0.5, color='blue', label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, color='red', label='Test Data')

# Plot quantile predictions
colors = ['purple', 'orange', 'green', 'orange', 'purple']
labels = ['10th percentile', '25th percentile', 'Median (50th)', '75th percentile', '90th percentile']

for i, q in enumerate(quantiles):
    plt.plot(X_plot, predictions[q], color=colors[i], label=labels[i], linewidth=2)

# Plot linear regression
plt.plot(X_plot, linear_pred, color='black', linestyle='--', label='Linear Regression (Mean)', linewidth=2)

# Fill between quantiles to show prediction intervals
plt.fill_between(X_plot.flatten(), predictions[0.1], predictions[0.9], 
                 alpha=0.2, color='gray', label='80% Prediction Interval')
plt.fill_between(X_plot.flatten(), predictions[0.25], predictions[0.75], 
                 alpha=0.3, color='lightblue', label='50% Prediction Interval')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quantile Regression vs Linear Regression\n(Notice how quantile bands capture varying uncertainty)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Evaluate on test set
print("\nTest Set Evaluation:")
print("="*50)

# Calculate quantile loss (check function)
def quantile_loss(y_true, y_pred, quantile):
    residuals = y_true - y_pred
    return np.mean(np.maximum(quantile * residuals, (quantile - 1) * residuals))

X_test_poly = create_polynomial_features(X_test, degree=2)

for q in quantiles:
    test_pred = quantile_models[q].predict(X_test_poly)
    loss = quantile_loss(y_test, test_pred, q)
    print(f"Quantile {q:.1f}: Loss = {loss:.4f}")

# Calculate coverage (what % of actual values fall within prediction intervals)
test_pred_10 = quantile_models[0.1].predict(X_test_poly)
test_pred_90 = quantile_models[0.9].predict(X_test_poly)
coverage_80 = np.mean((y_test >= test_pred_10) & (y_test <= test_pred_90))

test_pred_25 = quantile_models[0.25].predict(X_test_poly)
test_pred_75 = quantile_models[0.75].predict(X_test_poly)
coverage_50 = np.mean((y_test >= test_pred_25) & (y_test <= test_pred_75))

print(f"\nPrediction Interval Coverage:")
print(f"80% interval covers {coverage_80:.1%} of test points")
print(f"50% interval covers {coverage_50:.1%} of test points")

# Example: Risk assessment for a specific prediction
x_new = np.array([[7.5]])
x_new_poly = create_polynomial_features(x_new, degree=2)
print(f"\nRisk Assessment for X = {x_new[0,0]}:")
print("="*30)
for q in quantiles:
    pred = quantile_models[q].predict(x_new_poly)[0]
    print(f"{int(q*100)}th percentile: {pred:.2f}")

print(f"\nInterpretation:")
print(f"- 10% chance Y will be below {quantile_models[0.1].predict(x_new_poly)[0]:.2f}")
print(f"- 50% chance Y will be below {quantile_models[0.5].predict(x_new_poly)[0]:.2f}")
print(f"- 90% chance Y will be below {quantile_models[0.9].predict(x_new_poly)[0]:.2f}")       