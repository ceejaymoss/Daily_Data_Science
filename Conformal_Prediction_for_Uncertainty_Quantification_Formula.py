import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simple Conformal Prediction Class
class ConformalPredictor:
    def __init__(self, base_model, alpha=0.1):
        """
        alpha: desired miscoverage rate (0.1 for 90% coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.conformity_scores = None
        self.quantile = None
    
    def fit(self, X_train, y_train, X_cal, y_cal):
        """
        X_train, y_train: training data
        X_cal, y_cal: calibration data (held-out from training)
        """
        # Train the base model
        self.base_model.fit(X_train, y_train)
        
        # Calculate conformity scores on calibration set
        y_cal_pred = self.base_model.predict(X_cal)
        
        # Conformity score = absolute residual (can be customized)
        self.conformity_scores = np.abs(y_cal - y_cal_pred)
        
        # Calculate the quantile for prediction intervals
        n = len(self.conformity_scores)
        self.quantile = np.quantile(
            self.conformity_scores, 
            (1 - self.alpha) * (n + 1) / n  # Finite sample correction
        )
        
        return self
    
    def predict(self, X_test, return_intervals=True):
        """Predict with conformal intervals"""
        point_predictions = self.base_model.predict(X_test)
        
        if not return_intervals:
            return point_predictions
        
        # Create prediction intervals
        lower = point_predictions - self.quantile
        upper = point_predictions + self.quantile
        
        return point_predictions, lower, upper

# Example usage with different base models
class SimpleLinearModel:
    """Simple linear regression for demonstration"""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Generate synthetic data
np.random.seed(42)
n_total = 1000

# Create non-linear relationship with heteroscedastic noise
X = np.linspace(0, 4*np.pi, n_total).reshape(-1, 1)
true_function = lambda x: 2 * np.sin(x) + 0.5 * x
noise_std = 0.3 + 0.2 * np.abs(np.sin(X.flatten()))  # Varying noise
y = true_function(X.flatten()) + np.random.normal(0, noise_std)

# Split data: train / calibration / test
train_size = int(0.6 * n_total)
cal_size = int(0.2 * n_total)

X_train = X[:train_size]
y_train = y[:train_size]
X_cal = X[train_size:train_size + cal_size]
y_cal = y[train_size:train_size + cal_size]
X_test = X[train_size + cal_size:]
y_test = y[train_size + cal_size:]

print(f"Data splits - Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

# Demo with different coverage levels
coverage_levels = [0.8, 0.9, 0.95]
colors = ['red', 'blue', 'green']

plt.figure(figsize=(15, 10))

for i, coverage in enumerate(coverage_levels):
    plt.subplot(2, 2, i + 1)
    
    # Create conformal predictor
    alpha = 1 - coverage
    base_model = SimpleLinearModel()  # Deliberately simple/wrong model
    conformal_pred = ConformalPredictor(base_model, alpha=alpha)
    conformal_pred.fit(X_train, y_train, X_cal, y_cal)
    
    # Generate predictions
    X_plot = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
    point_pred, lower, upper = conformal_pred.predict(X_plot)
    
    # Plot results
    plt.scatter(X_test, y_test, alpha=0.6, s=20, color='black', label='Test Data')
    plt.plot(X_plot, true_function(X_plot.flatten()), 'g--', linewidth=2, label='True Function')
    plt.plot(X_plot, point_pred, color=colors[i], linewidth=2, label='Model Prediction')
    plt.fill_between(X_plot.flatten(), lower, upper, alpha=0.3, color=colors[i], 
                     label=f'{coverage:.0%} Conformal Interval')
    
    # Calculate actual coverage on test set
    test_pred, test_lower, test_upper = conformal_pred.predict(X_test)
    actual_coverage = np.mean((y_test >= test_lower) & (y_test <= test_upper))
    
    plt.title(f'{coverage:.0%} Coverage\nActual: {actual_coverage:.1%} '
              f'(Target: {coverage:.0%})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Compare different conformity scores
plt.subplot(2, 2, 4)

# Custom conformity score (normalized residual)
class ConformalPredictorNormalized(ConformalPredictor):
    def fit(self, X_train, y_train, X_cal, y_cal):
        self.base_model.fit(X_train, y_train)
        y_cal_pred = self.base_model.predict(X_cal)
        
        # Normalized conformity score
        residuals = np.abs(y_cal - y_cal_pred)
        self.conformity_scores = residuals / (np.abs(y_cal_pred) + 1e-8)  # Avoid division by 0
        
        n = len(self.conformity_scores)
        self.quantile = np.quantile(self.conformity_scores, (1 - self.alpha) * (n + 1) / n)
        return self
    
    def predict(self, X_test, return_intervals=True):
        point_predictions = self.base_model.predict(X_test)
        if not return_intervals:
            return point_predictions
        
        # Apply normalized intervals
        interval_width = self.quantile * (np.abs(point_predictions) + 1e-8)
        lower = point_predictions - interval_width
        upper = point_predictions + interval_width
        return point_predictions, lower, upper

# Compare standard vs normalized conformity scores
base_model1 = SimpleLinearModel()
base_model2 = SimpleLinearModel()

standard_cp = ConformalPredictor(base_model1, alpha=0.1)
normalized_cp = ConformalPredictorNormalized(base_model2, alpha=0.1)

standard_cp.fit(X_train, y_train, X_cal, y_cal)
normalized_cp.fit(X_train, y_train, X_cal, y_cal)

X_plot = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
_, std_lower, std_upper = standard_cp.predict(X_plot)
_, norm_lower, norm_upper = normalized_cp.predict(X_plot)

plt.scatter(X_test, y_test, alpha=0.6, s=20, color='black', label='Test Data')
plt.fill_between(X_plot.flatten(), std_lower, std_upper, alpha=0.3, color='blue', 
                 label='Standard Conformity')
plt.fill_between(X_plot.flatten(), norm_lower, norm_upper, alpha=0.3, color='red', 
                 label='Normalized Conformity')

plt.title('Different Conformity Scores')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Performance summary
print("\nConformal Prediction Results:")
print("="*50)

for coverage in coverage_levels:
    alpha = 1 - coverage
    cp = ConformalPredictor(SimpleLinearModel(), alpha=alpha)
    cp.fit(X_train, y_train, X_cal, y_cal)
    
    _, test_lower, test_upper = cp.predict(X_test)
    actual_coverage = np.mean((y_test >= test_lower) & (y_test <= test_upper))
    avg_width = np.mean(test_upper - test_lower)
    
    print(f"{coverage:.0%} Target Coverage:")
    print(f"  Actual Coverage: {actual_coverage:.1%}")
    print(f"  Average Width: {avg_width:.3f}")
    print(f"  Width Efficiency: {abs(actual_coverage - coverage):.3f} from target")
    print()

print("Key Insight: Even with a deliberately wrong model (linear for non-linear data),")
print("conformal prediction still provides valid coverage guarantees!")
