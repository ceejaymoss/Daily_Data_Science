# Import tools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
#

#
# Create synthetic housing dataset
np.random.seed(42)
n_samples = 1000
#

#
# Base Features
sqft = np.random.normal(1500, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
neighborhood = np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples)
year_built = np.random.randint(1950, 2020, n_samples)
#

#
# Create price with interactions
# Note how we explicitly create price with feature interactions
base_price = 100000 + 100 * sqft + 15000 * bedrooms + 20000 * bathrooms
neighborhood_factor = np.where(neighborhood == 'Downtown', 1.5, 
                              np.where(neighborhood == 'Suburbs', 1.2, 0.8))
age_factor = 1 - 0.005 * (2023 - year_built)

# The key interaction: sqft is worth more in Downtown and newer houses
sqft_premium = sqft * neighborhood_factor * age_factor * 50

# Final price with noise
price = (base_price + sqft_premium) * np.random.normal(1, 0.1, n_samples)
#

# Create dataframe
data = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'neighborhood': neighborhood,
    'year_built': year_built,
    'price': price
})
#

#
# Split Data
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#

#
# Function to evaluate and print model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: ${mse:.2f}, R²: {r2:.4f}")
    return mse, r2, y_pred
#

#
# Preprocessing pipeline
numeric_features = ['sqft', 'bedrooms', 'bathrooms', 'year_built']
categorical_features = ['neighborhood']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
#

#
# Model 1 : Linear model without feature crossing
basic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
basic_pipeline.fit(X_train, y_train)
basic_mse, basic_r2, basic_pred = evaluate_model(basic_pipeline, X_test, y_test, "Basic Linear Model")
#

#
# Model 2: Linear model with manual feature crossing
# Create age feature (easier to understand than year_built for crossing)
X_train_cross = X_train.copy()
X_test_cross = X_test.copy()
X_train_cross['age'] = 2023 - X_train_cross['year_built']
X_test_cross['age'] = 2023 - X_test_cross['year_built']

# Create crossed features
X_train_cross['sqft_x_neighborhood'] = X_train_cross['sqft'] * (X_train_cross['neighborhood'] == 'Downtown').astype(int) * 2 + \
                                      X_train_cross['sqft'] * (X_train_cross['neighborhood'] == 'Suburbs').astype(int) * 1.5
X_test_cross['sqft_x_neighborhood'] = X_test_cross['sqft'] * (X_test_cross['neighborhood'] == 'Downtown').astype(int) * 2 + \
                                     X_test_cross['sqft'] * (X_test_cross['neighborhood'] == 'Suburbs').astype(int) * 1.5

X_train_cross['sqft_x_age'] = X_train_cross['sqft'] * (1 / (X_train_cross['age'] + 1))
X_test_cross['sqft_x_age'] = X_test_cross['sqft'] * (1 / (X_test_cross['age'] + 1))

# Update preprocessor for crossed features
manual_cross_features = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 
                      'sqft_x_neighborhood', 'sqft_x_age']

manual_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), manual_cross_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

manual_cross_pipeline = Pipeline(steps=[
    ('preprocessor', manual_preprocessor),
    ('regressor', LinearRegression())
])
manual_cross_pipeline.fit(X_train_cross, y_train)
manual_mse, manual_r2, manual_pred = evaluate_model(manual_cross_pipeline, X_test_cross, y_test, 
                                                "Linear Model with Manual Feature Crossing")
#

#
# Model 3: Automated polynomial feature crossing
poly_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

poly_pipeline = Pipeline(steps=[
    ('preprocessor', poly_preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('regressor', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
poly_mse, poly_r2, poly_pred = evaluate_model(poly_pipeline, X_test, y_test, 
                                          "Linear Model with PolynomialFeatures")
#

#
# Plot results
plt.figure(figsize=(12, 6))

# Plot residuals
plt.subplot(1, 2, 1)
plt.scatter(y_test, basic_pred - y_test, alpha=0.5, label='Basic Model')
plt.scatter(y_test, manual_pred - y_test, alpha=0.5, label='Manual Crossing')
plt.scatter(y_test, poly_pred - y_test, alpha=0.5, label='Polynomial Features')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.title('Residual Plots')
plt.legend()

# Plot model comparison
plt.subplot(1, 2, 2)
models = ['Basic Linear', 'Manual Crossing', 'Polynomial']
r2_scores = [basic_r2, manual_r2, poly_r2]
mse_scores = [basic_mse/1e8, manual_mse/1e8, poly_mse/1e8]  # Scale down for better visualization

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score')
bars2 = ax2.bar(x + width/2, mse_scores, width, label='MSE (× 10⁸)', color='orange')

ax1.set_ylabel('R² Score')
ax2.set_ylabel('MSE (× 10⁸)')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
#

#
# Feature importance for manual crossing model
manual_model = manual_cross_pipeline.named_steps['regressor']
feature_names = (numeric_features + ['sqft_x_neighborhood', 'sqft_x_age'] + 
                ['neighborhood_' + cat for cat in ['Suburbs', 'Rural']])

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': manual_model.coef_
})
print("\nFeature Importance (Coefficients):")
print(coef_df.sort_values('Coefficient', key=abs, ascending=False))
#