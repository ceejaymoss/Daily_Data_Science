import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

# Generate realistic customer data for demonstration
np.random.seed(42)
n_custoemrs = 5000

# Simulate customer transaction data
data = {
    'customer_id': range(n_customers),
    'age': np.random.randint(18, 75, n_customers),
    'gender': np.random.choice(['M', 'F', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
    'income': np.random.lognormal(10.5, 0.6, n_customers),
    'num_purchases': np.random.poisson(8, n_customers),
    'total_spent': np.random.lognormal(7, 1.2, n_customers),
    'avg_order_value': np.random.gamma(2, 50, n_customers),
    'days_since_first_purchase': np.random.randint(1, 1095, n_customers),
    'days_since_last_purchase': np.random.randint(0, 365, n_customers),
    'num_returns': np.random.poisson(0.5, n_customers),
    'num_support_tickets': np.random.poisson(1, n_customers),
    'email_opens': np.random.binomial(50, 0.3, n_customers),
    'email_clicks': np.random.binomial(50, 0.1, n_customers),
    'website_visits': np.random.poisson(15, n_customers),
    'mobile_app_usage': np.random.binomial(1, 0.4, n_customers),
    'social_media_engagement': np.random.poisson(5, n_customers),
    'referred_customers': np.random.poisson(0.3, n_customers),
    'loyalty_points': np.random.gamma(2, 100, n_customers),
}

# Product category purchases (0-10 purchases in each)
categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books']
for cat in categories:
    data[f'purchases_{cat.lower()}'] = np.random.poisson(1.5, n_customers)

df = pd.DataFrame(data)

print("="*70)
print("BUYER PERSONA FEATURE ENGINEERING MASTERCLASS")
print("="*70)
print(f"\nStarting with {len(df)} customers and {len(df.columns)} raw features")
print("\nRaw Data Sample:")
print(df.head())

# ============================================================================
# FEATURE ENGINEERING CATEGORIES
# ============================================================================

print("\n" + "="*70)
print("1. RFM (RECENCY, FREQUENCY, MONETARY) - THE FOUNDATION")
print("="*70)

# RFM features - gold standard for customer segmentation
df['recency'] = df['days_since_last_purchase']
df['frequency'] = df['num_purchases']
df['monetary'] = df['total_spent']

# RFM scores (quintiles)
df['recency_score'] = pd.qcut(df['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
df['frequency_score'] = pd.qcut(df['frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
df['monetary_score'] = pd.qcut(df['monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')

# Combined RFM score
df['rfm_score'] = (df['recency_score'].astype(int) + 
                   df['frequency_score'].astype(int) + 
                   df['monetary_score'].astype(int))

print("\nRFM Features Created:")
print(df[['customer_id', 'recency', 'frequency', 'monetary', 'rfm_score']].head())


# ============================================================================
print("\n" + "="*70)
print("2. BEHAVIORAL RATIOS - REVEALING CUSTOMER PSYCHOLOGY")
print("="*70)

# Engagement ratios
df['email_click_through_rate'] = np.where(
    df['email_opens'] > 0,
    df['email_clicks'] / df['email_opens'],
    0
)

df['return_rate'] = np.where(
    df['num_purchases'] > 0,
    df['num_returns'] / df['num_purchases'],
    0
)

df['support_per_purchase'] = np.where(
    df['num_purchases'] > 0,
    df['num_support_tickets'] / df['num_purchases'],
    0
)

# Purchase intensity (purchases per active day)
df['purchase_intensity'] = np.where(
    df['days_since_first_purchase'] > 0,
    df['num_purchases'] / df['days_since_first_purchase'] * 365,  # Annualized
    0
)

# Loyalty engagement
df['points_per_dollar'] = np.where(
    df['total_spent'] > 0,
    df['loyalty_points'] / df['total_spent'],
    0
)

print("\nBehavioral Ratios:")
print(df[['customer_id', 'email_click_through_rate', 'return_rate', 
          'purchase_intensity']].head())

# ============================================================================
print("\n" + "="*70)
print("3. CUSTOMER LIFECYCLE STAGE")
print("="*70)

# Customer lifecycle indicators
df['customer_tenure_months'] = df['days_since_first_purchase'] / 30.5
df['avg_days_between_purchases'] = np.where(
    df['num_purchases'] > 1,
    df['days_since_first_purchase'] / (df['num_purchases'] - 1),
    np.nan
)

# Lifecycle stage classification
def classify_lifecycle(row):
    if row['days_since_first_purchase'] < 90:
        return 'New'
    elif row['days_since_last_purchase'] > 180:
        return 'At Risk'
    elif row['num_purchases'] >= 10:
        return 'Champion'
    elif row['num_purchases'] >= 5:
        return 'Loyal'
    else:
        return 'Active'

df['lifecycle_stage'] = df.apply(classify_lifecycle, axis=1)

print("\nLifecycle Distribution:")
print(df['lifecycle_stage'].value_counts())

# ============================================================================
print("\n" + "="*70)
print("4. PRODUCT AFFINITY & CATEGORY DIVERSITY")
print("="*70)
 Category diversity (Shannon entropy)
category_cols = [col for col in df.columns if col.startswith('purchases_')]

def calculate_category_entropy(row):
    purchases = row[category_cols].values
    total = purchases.sum()
    if total == 0:
        return 0
    proportions = purchases / total
    # Shannon entropy
    return -np.sum(proportions * np.log(proportions + 1e-10))

df['category_diversity'] = df.apply(calculate_category_entropy, axis=1)

# Dominant category
df['dominant_category'] = df[category_cols].idxmax(axis=1).str.replace('purchases_', '')

# Category concentration (Herfindahl index)
def herfindahl_index(row):
    purchases = row[category_cols].values
    total = purchases.sum()
    if total == 0:
        return 0
    proportions = purchases / total
    return np.sum(proportions ** 2)

df['category_concentration'] = df.apply(herfindahl_index, axis=1)

print("\nProduct Affinity Features:")
print(df[['customer_id', 'category_diversity', 'dominant_category', 
          'category_concentration']].head())

# ============================================================================
print("\n" + "="*70)
print("5. CHANNEL PREFERENCE & OMNICHANNEL BEHAVIOR")
print("="*70)

# Digital engagement score
df['digital_engagement'] = (
    df['website_visits'] / df['website_visits'].max() * 0.4 +
    df['email_opens'] / df['email_opens'].max() * 0.3 +
    df['social_media_engagement'] / df['social_media_engagement'].max() * 0.3
)

# Omnichannel user
df['is_omnichannel'] = (
    (df['website_visits'] > df['website_visits'].median()) &
    (df['mobile_app_usage'] == 1) &
    (df['email_opens'] > df['email_opens'].median())
).astype(int)

print("\nChannel Preference:")
print(df[['customer_id', 'digital_engagement', 'is_omnichannel']].head())

# ============================================================================
print("\n" + "="*70)
print("6. CUSTOMER VALUE & GROWTH FEATURES")
print("="*70)

# Customer Lifetime Value (CLV) proxy
df['avg_purchase_frequency_per_month'] = df['num_purchases'] / df['customer_tenure_months']
df['clv_estimate'] = df['avg_order_value'] * df['avg_purchase_frequency_per_month'] * 24  # 2-year projection

# Growth indicators
df['spending_growth_rate'] = np.where(
    df['num_purchases'] > 1,
    (df['avg_order_value'] - df['total_spent'] / df['num_purchases']) / 
    (df['total_spent'] / df['num_purchases']),
    0
)

# Virality coefficient
df['virality_score'] = df['referred_customers'] / (df['num_purchases'] + 1)

print("\nValue Features:")
print(df[['customer_id', 'clv_estimate', 'virality_score']].head())

# ============================================================================
print("\n" + "="*70)
print("7. DEMOGRAPHIC INTERACTIONS (FEATURE CROSSING)")
print("="*70)

# Age-income segments
df['age_segment'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                           labels=['Gen_Z', 'Millennial', 'Gen_X', 'Boomer', 'Senior'])
df['income_segment'] = pd.qcut(df['income'], q=4, labels=['Low', 'Mid', 'High', 'Premium'])

# Create interaction features
df['age_income_segment'] = df['age_segment'].astype(str) + '_' + df['income_segment'].astype(str)

# Income-to-spend ratio (purchasing power usage)
df['spend_to_income_ratio'] = df['total_spent'] / df['income']

print("\nDemographic Interactions:")
print(df[['customer_id', 'age_segment', 'income_segment', 'age_income_segment']].head())

# ============================================================================
print("\n" + "="*70)
print("8. ENGAGEMENT QUALITY SCORES")
print("="*70)

# Weighted engagement score
df['engagement_quality'] = (
    df['email_click_through_rate'] * 0.25 +
    (1 - df['return_rate']) * 0.25 +
    (df['loyalty_points'] / df['loyalty_points'].max()) * 0.20 +
    (df['social_media_engagement'] / df['social_media_engagement'].max()) * 0.15 +
    (df['referred_customers'] / df['referred_customers'].max()) * 0.15
)

# Customer satisfaction proxy
df['satisfaction_proxy'] = (
    (1 - df['return_rate']) * 0.4 +
    (1 - df['support_per_purchase']) * 0.3 +
    (df['num_purchases'] / df['num_purchases'].max()) * 0.3
)

print("\nEngagement Quality:")
print(df[['customer_id', 'engagement_quality', 'satisfaction_proxy']].head())

# ============================================================================
print("\n" + "="*70)
print("9. TEMPORAL PATTERNS")
print("="*70)

# Purchase consistency (coefficient of variation)
# Simulating purchase dates for this
np.random.seed(42)
df['purchase_consistency'] = np.random.uniform(0.3, 1.5, n_customers)

# Churn risk indicator
df['churn_risk'] = np.where(
    (df['days_since_last_purchase'] > df['avg_days_between_purchases'] * 2) |
    (df['days_since_last_purchase'] > 180),
    1, 0
)

print("\nTemporal Features:")
print(df[['customer_id', 'purchase_consistency', 'churn_risk']].head())

# ============================================================================
print("\n" + "="*70)
print("FINAL FEATURE SET SUMMARY")
print("="*70)

# Select key features for persona clustering
persona_features = [
    'rfm_score', 'monetary', 'frequency', 'recency',
    'email_click_through_rate', 'return_rate', 'purchase_intensity',
    'lifecycle_stage', 'category_diversity', 'category_concentration',
    'digital_engagement', 'is_omnichannel',
    'clv_estimate', 'virality_score',
    'spend_to_income_ratio', 'engagement_quality', 'satisfaction_proxy',
    'churn_risk', 'age', 'income'
]

print(f"\nTotal features created: {len(df.columns)}")
print(f"Key features for persona modeling: {len(persona_features)}")
print(f"\nFeature Categories:")
print("  - RFM & Value: 4 features")
print("  - Behavioral Ratios: 3 features")
print("  - Lifecycle: 1 feature")
print("  - Product Affinity: 2 features")
print("  - Channel Preference: 2 features")
print("  - Customer Value: 2 features")
print("  - Engagement: 2 features")
print("  - Demographics: 2 features")
print("  - Risk: 1 feature")

# Create a correlation heatmap for key features
numeric_features = [f for f in persona_features if f not in ['lifecycle_stage']]
correlation_matrix = df[numeric_features].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix for Buyer Persona Modeling', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Feature importance insights
print("\n" + "="*70)
print("FEATURE ENGINEERING INSIGHTS")
print("="*70)

print("\n1. RFM remains king - but enhanced with behavioral context")
print("2. Ratios reveal psychology better than raw counts")
print("3. Category diversity shows shopping sophistication")
print("4. Omnichannel behavior indicates tech-savviness")
print("5. Lifecycle stage contextualizes all other metrics")
print("6. CLV estimate helps prioritize persona value")
print("7. Engagement quality separates active from valuable")
print("8. Churn risk enables proactive retention strategies")

print("\n" + "="*70)
print("NEXT STEPS FOR PERSONA CREATION")
print("="*70)
print("1. Standardize/normalize features (especially for clustering)")
print("2. Apply dimensionality reduction (PCA/t-SNE) for visualization")
print("3. Use K-means, DBSCAN, or hierarchical clustering")
print("4. Profile each cluster to create persona narratives")
print("5. Validate with business stakeholders")
print("6. Build classification model for new customer assignment")

# Export feature-rich dataset
print(f"\n✓ Feature engineering complete!")
print(f"  Dataset shape: {df.shape}")
print(f"  Ready for clustering and persona modeling")