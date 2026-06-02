import pandas as pd
import numpy as np

# Generate 1000 realistic dummy records with churn correlated to financial_autonomy_idx and avg_balance
np.random.seed(42)

n_samples = 1000

# Generate base features
customer_id = range(1, n_samples + 1)
age = np.random.randint(18, 80, n_samples)
financial_autonomy_idx = np.random.uniform(1.0, 10.0, n_samples)
avg_balance = np.random.exponential(5000, n_samples)

# Generate churn with strong correlation to financial_autonomy_idx and avg_balance
# Churn probability = inverse function of financial_autonomy_idx and avg_balance
# Normalize features to [0, 1] for probability calculation
fa_normalized = (financial_autonomy_idx - financial_autonomy_idx.min()) / \
    (financial_autonomy_idx.max() - financial_autonomy_idx.min())
balance_normalized = (avg_balance - avg_balance.min()) / \
    (avg_balance.max() - avg_balance.min())

# Churn probability: high when both financial_autonomy_idx and avg_balance are low
# Use inverse: (1 - fa_normalized) * (1 - balance_normalized)
churn_prob = (1 - fa_normalized) * (1 - balance_normalized)

# Add some randomness to make it realistic (scale down the probability a bit)
churn_prob = churn_prob * 0.7  # Scale to get reasonable churn rates

# Generate churn based on probability
churned = np.array([np.random.binomial(1, p) for p in churn_prob])

data = {
    'customer_id': customer_id,
    'age': age,
    'financial_autonomy_idx': financial_autonomy_idx,
    'avg_balance': avg_balance,
    'churned': churned
}

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

print("dataset.csv generated successfully!")
print(f"\nDataset shape: {df.shape}")
print(f"Churn rate: {df['churned'].mean():.2%}")
print(f"\nChurn distribution:")
print(df['churned'].value_counts())
print(f"\nSample of generated data:")
print(df.head(10))

# Show correlation between features and churn
print(f"\nCorrelation with churn:")
print(f"  age: {df[['age', 'churned']].corr().iloc[0, 1]:.4f}")
print(
    f"  financial_autonomy_idx: {df[['financial_autonomy_idx', 'churned']].corr().iloc[0, 1]:.4f}")
print(f"  avg_balance: {df[['avg_balance', 'churned']].corr().iloc[0, 1]:.4f}")
