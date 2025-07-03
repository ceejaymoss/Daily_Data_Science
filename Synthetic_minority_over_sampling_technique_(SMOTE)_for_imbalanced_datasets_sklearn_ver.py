import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, weights=[0.95, 0.05], 
                          flip_y=0.01, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=y)

print(f"Original training set distribution:")
print(f"Class 0: {sum(y_train==0)} samples")
print(f"Class 1: {sum(y_train==1)} samples")

# Apply SMOTE
smote = SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Class 0: {sum(y_train_balanced==0)} samples")
print(f"Class 1: {sum(y_train_balanced==1)} samples")

# Train models for comparison
# Model 1: Without SMOTE
clf_original = RandomForestClassifier(random_state=42)
clf_original.fit(X_train, y_train)

# Model 2: With SMOTE
clf_balanced = RandomForestClassifier(random_state=42)
clf_balanced.fit(X_train_balanced, y_train_balanced)

# Evaluate both models
print("\n--- Model Performance WITHOUT SMOTE ---")
print(classification_report(y_test, clf_original.predict(X_test)))

print("\n--- Model Performance WITH SMOTE ---")
print(classification_report(y_test, clf_balanced.predict(X_test)))