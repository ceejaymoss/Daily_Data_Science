import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

# Set random seed for reproducibility
np.random.seed(42)

class EnsembleFeatureSelector:
    """
    Ensemble Feature Selection combining multiple feature selection methods
    """
    
    def __init__(self, methods=None, n_features=10):
        """
        Initialize ensemble feature selector
        
        Parameters:
        methods: dict of method_name -> (selector_class, params)
        n_features: number of top features to select
        """
        self.n_features = n_features
        self.feature_rankings_ = {}
        self.feature_scores_ = {}
        self.ensemble_scores_ = None
        self.selected_features_ = None
        
        if methods is None:
            # Default methods
            self.methods = {
                'f_classif': (SelectKBest, {'score_func': f_classif, 'k': n_features}),
                'mutual_info': (SelectKBest, {'score_func': mutual_info_classif, 'k': n_features}),
                'lasso': (SelectFromModel, {'estimator': LassoCV(cv=3, random_state=42), 'max_features': n_features}),
                'rf_importance': (SelectFromModel, {'estimator': RandomForestClassifier(n_estimators=100, random_state=42), 'max_features': n_features}),
                'rfe_logistic': (RFE, {'estimator': LogisticRegression(random_state=42, max_iter=1000), 'n_features_to_select': n_features})
            }
        else:
            self.methods = methods
    
    def fit(self, X, y):
        """
        Fit all feature selection methods and compute ensemble scores
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Apply each feature selection method
        for method_name, (selector_class, params) in self.methods.items():
            print(f"Applying {method_name}...")
            
            # Create and fit selector
            selector = selector_class(**params)
            selector.fit(X_array, y)
            
            # Get feature scores/rankings
            if hasattr(selector, 'scores_'):
                # For methods that provide scores (like SelectKBest)
                scores = selector.scores_
                # Convert to rankings (1 = best)
                rankings = rankdata(-scores, method='ordinal')
            elif hasattr(selector, 'ranking_'):
                # For RFE which provides rankings directly
                rankings = selector.ranking_
                # Convert rankings to scores (higher = better)
                scores = len(rankings) - rankings + 1
            elif hasattr(selector, 'estimator_'):
                # For SelectFromModel
                if hasattr(selector.estimator_, 'feature_importances_'):
                    scores = selector.estimator_.feature_importances_
                elif hasattr(selector.estimator_, 'coef_'):
                    scores = np.abs(selector.estimator_.coef_.flatten())
                else:
                    # Fallback: use selection mask
                    scores = selector.get_support().astype(float)
                rankings = rankdata(-scores, method='ordinal')
            else:
                # Fallback: use selection mask
                scores = selector.get_support().astype(float)
                rankings = rankdata(-scores, method='ordinal')
            
            self.feature_scores_[method_name] = scores
            self.feature_rankings_[method_name] = rankings
        
        # Compute ensemble scores using multiple aggregation methods
        self._compute_ensemble_scores(X_array.shape[1])
        
        return self
    
    def _compute_ensemble_scores(self, n_total_features):
        """Compute ensemble scores using different aggregation methods"""
        
        # Method 1: Borda Count (rank-based)
        borda_scores = np.zeros(n_total_features)
        for method_name, rankings in self.feature_rankings_.items():
            borda_scores += (n_total_features - rankings + 1)
        
        # Method 2: Normalized Score Average
        normalized_scores = np.zeros(n_total_features)
        for method_name, scores in self.feature_scores_.items():
            # Min-max normalization
            if np.max(scores) > np.min(scores):
                norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                norm_scores = np.ones_like(scores)
            normalized_scores += norm_scores
        normalized_scores /= len(self.methods)
        
        # Method 3: Voting (top-k from each method)
        voting_scores = np.zeros(n_total_features)
        k_vote = min(self.n_features, n_total_features // 2)
        for method_name, rankings in self.feature_rankings_.items():
            top_k_mask = rankings <= k_vote
            voting_scores += top_k_mask.astype(float)
        
        # Combine all methods
        self.ensemble_scores_ = {
            'borda': borda_scores,
            'normalized_avg': normalized_scores,
            'voting': voting_scores,
            'combined': (borda_scores / np.max(borda_scores) + 
                        normalized_scores + 
                        voting_scores / np.max(voting_scores)) / 3
        }
        
        # Select features based on combined score
        combined_rankings = rankdata(-self.ensemble_scores_['combined'], method='ordinal')
        self.selected_features_ = np.where(combined_rankings <= self.n_features)[0]
        
        return self
    
    def transform(self, X):
        """Transform X by selecting ensemble-chosen features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit selector before transform")
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        else:
            return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit selector and transform X"""
        return self.fit(X, y).transform(X)
    
    def get_feature_ranking_comparison(self):
        """Get detailed comparison of feature rankings across methods"""
        df_rankings = pd.DataFrame(self.feature_rankings_, index=self.feature_names_)
        df_scores = pd.DataFrame(self.feature_scores_, index=self.feature_names_)
        
        # Add ensemble results
        for ensemble_method, scores in self.ensemble_scores_.items():
            df_scores[f'ensemble_{ensemble_method}'] = scores
            df_rankings[f'ensemble_{ensemble_method}'] = rankdata(-scores, method='ordinal')
        
        return df_rankings, df_scores
    
    def plot_feature_comparison(self, top_n=15):
        """Plot comparison of feature selection methods"""
        df_rankings, df_scores = self.get_feature_ranking_comparison()
        
        # Select top features based on combined ensemble score
        top_features_idx = np.argsort(-self.ensemble_scores_['combined'])[:top_n]
        top_features = [self.feature_names_[i] for i in top_features_idx]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Heatmap of rankings
        rankings_subset = df_rankings.loc[top_features, list(self.methods.keys())]
        sns.heatmap(rankings_subset, annot=True, fmt='d', cmap='RdYlGn_r', 
                   ax=axes[0,0], cbar_kws={'label': 'Rank'})
        axes[0,0].set_title('Feature Rankings by Method\n(Lower = Better)')
        axes[0,0].set_xlabel('Selection Method')
        
        # Plot 2: Normalized scores heatmap  
        scores_subset = df_scores.loc[top_features, list(self.methods.keys())]
        # Normalize each column for better visualization
        scores_normalized = scores_subset.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x)
        sns.heatmap(scores_normalized, annot=True, fmt='.2f', cmap='viridis', 
                   ax=axes[0,1], cbar_kws={'label': 'Normalized Score'})
        axes[0,1].set_title('Normalized Feature Scores by Method\n(Higher = Better)')
        axes[0,1].set_xlabel('Selection Method')
        
        # Plot 3: Ensemble scores comparison
        ensemble_methods = ['borda', 'normalized_avg', 'voting', 'combined']
        ensemble_scores_subset = df_scores.loc[top_features, 
                                             [f'ensemble_{method}' for method in ensemble_methods]]
        ensemble_scores_subset.columns = ensemble_methods
        
        # Normalize for visualization
        ensemble_normalized = ensemble_scores_subset.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x)
        sns.heatmap(ensemble_normalized, annot=True, fmt='.2f', cmap='plasma', 
                   ax=axes[1,0], cbar_kws={'label': 'Normalized Score'})
        axes[1,0].set_title('Ensemble Method Comparison')
        axes[1,0].set_xlabel('Ensemble Method')
        
        # Plot 4: Bar plot of final combined scores
        final_scores = self.ensemble_scores_['combined'][top_features_idx]
        axes[1,1].barh(range(len(top_features)), final_scores)
        axes[1,1].set_yticks(range(len(top_features)))
        axes[1,1].set_yticklabels(top_features)
        axes[1,1].set_xlabel('Combined Ensemble Score')
        axes[1,1].set_title('Final Feature Ranking (Combined Score)')
        axes[1,1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig

def compare_with_individual_methods(X, y, n_features=10):
    """Compare ensemble selection with individual methods"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize results storage
    results = {}
    
    # Test individual methods
    individual_methods = {
        'f_classif': SelectKBest(score_func=f_classif, k=n_features),
        'mutual_info': SelectKBest(score_func=mutual_info_classif, k=n_features),
        'lasso': SelectFromModel(LassoCV(cv=3, random_state=42), max_features=n_features),
        'rf_importance': SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=n_features),
        'rfe_logistic': RFE(LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=n_features)
    }
    
    # Evaluate individual methods
    for method_name, selector in individual_methods.items():
        # Select features
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(clf, X_train_selected, y_train, cv=5).mean()
        
        results[method_name] = {
            'test_accuracy': accuracy,
            'cv_accuracy': cv_score,
            'n_features': X_train_selected.shape[1]
        }
    
    # Test ensemble method
    ensemble_selector = EnsembleFeatureSelector(n_features=n_features)
    X_train_ensemble = ensemble_selector.fit_transform(X_train_scaled, y_train)
    X_test_ensemble = ensemble_selector.transform(X_test_scaled)
    
    # Train classifier on ensemble-selected features
    clf_ensemble = LogisticRegression(random_state=42, max_iter=1000)
    clf_ensemble.fit(X_train_ensemble, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = clf_ensemble.predict(X_test_ensemble)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    cv_score_ensemble = cross_val_score(clf_ensemble, X_train_ensemble, y_train, cv=5).mean()
    
    results['ensemble'] = {
        'test_accuracy': accuracy_ensemble,
        'cv_accuracy': cv_score_ensemble,
        'n_features': X_train_ensemble.shape[1]
    }
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results).T
    print("Performance Comparison:")
    print(results_df.round(4))
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test accuracy comparison
    test_acc = results_df['test_accuracy']
    colors = ['lightblue'] * (len(test_acc) - 1) + ['red']  # Highlight ensemble
    ax1.bar(range(len(test_acc)), test_acc, color=colors)
    ax1.set_xticks(range(len(test_acc)))
    ax1.set_xticklabels(test_acc.index, rotation=45)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylim([test_acc.min() - 0.02, test_acc.max() + 0.02])
    
    # Add value labels on bars
    for i, v in enumerate(test_acc):
        ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
    
    # CV accuracy comparison
    cv_acc = results_df['cv_accuracy']
    ax2.bar(range(len(cv_acc)), cv_acc, color=colors)
    ax2.set_xticks(range(len(cv_acc)))
    ax2.set_xticklabels(cv_acc.index, rotation=45)
    ax2.set_ylabel('CV Accuracy')
    ax2.set_title('Cross-Validation Accuracy Comparison')
    ax2.set_ylim([cv_acc.min() - 0.02, cv_acc.max() + 0.02])
    
    # Add value labels on bars
    for i, v in enumerate(cv_acc):
        ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results_df, ensemble_selector

def main():
    """Main demonstration of ensemble feature selection"""
    print("Ensemble Feature Selection Demo")
    print("=" * 50)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {len(np.unique(y))}")
    
    # Create DataFrame for easier handling
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Apply ensemble feature selection
    print("\nApplying Ensemble Feature Selection...")
    ensemble_selector = EnsembleFeatureSelector(n_features=10)
    ensemble_selector.fit(X_df, y)
    
    # Plot feature comparison
    print("\nGenerating feature comparison plots...")
    ensemble_selector.plot_feature_comparison(top_n=15)
    
    # Show selected features
    selected_feature_names = [feature_names[i] for i in ensemble_selector.selected_features_]
    print(f"\nSelected Features ({len(selected_feature_names)}):")
    for i, feature in enumerate(selected_feature_names, 1):
        print(f"{i:2d}. {feature}")
    
    # Compare with individual methods
    print("\nComparing performance with individual methods...")
    results_df, _ = compare_with_individual_methods(X, y, n_features=10)
    
    print("\nEnsemble Feature Selection Demo Complete!")

if __name__ == "__main__":
    main()