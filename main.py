import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

num_members = 5000 
members_df = pd.DataFrame({
    'member_id': range(1, num_members + 1),
    'age': np.random.randint(50, 85, size=num_members),
    'num_chronic_conditions': np.random.randint(1, 6, size=num_members)
})

# Simulate pharmacy claims data for a chronic medication
claims = []
start_date = pd.to_datetime('2024-01-01')
for _, member in members_df.iterrows():
    # Adherent members have small, regular gaps between refills
    # Non-adherent members have large, irregular gaps
    is_adherent_profile = np.random.rand() > (0.1 + member['num_chronic_conditions'] * 0.05)
    
    current_date = start_date + timedelta(days=np.random.randint(1, 30))
    for _ in range(np.random.randint(5, 12)): # Number of refills in the year
        days_supply = 30
        claims.append([member['member_id'], current_date, days_supply])
        
        if is_adherent_profile:
            # Small, regular gaps
            gap_days = np.random.randint(28, 35)
        else:
            # Large, irregular gaps
            gap_days = np.random.randint(30, 90)
        current_date += timedelta(days=gap_days)

claims_df = pd.DataFrame(claims, columns=['member_id', 'fill_date', 'days_supply'])
print("--- Simulated Claims Data Head ---")
print(claims_df.head())


# PDC (Proportion of Days Covered) is the key metric
def calculate_pdc_and_features(df, measurement_start, measurement_end):
    """Calculates PDC and other features for each member within a time window."""
    features_list = []
    total_days_in_period = (measurement_end - measurement_start).days + 1
    
    for member_id, group in df.groupby('member_id'):
        group = group.sort_values('fill_date')
        
        # Filter claims within the measurement period
        period_claims = group[(group['fill_date'] >= measurement_start) & (group['fill_date'] <= measurement_end)]
        if period_claims.empty:
            continue
            
        # Calculate days covered
        covered_days = set()
        for _, row in period_claims.iterrows():
            for i in range(row['days_supply']):
                covered_days.add(row['fill_date'] + timedelta(days=i))
        
        pdc = len(covered_days) / total_days_in_period
        
        # Other features
        refill_gaps = period_claims['fill_date'].diff().dt.days.dropna()
        
        features_list.append({
            'member_id': member_id,
            'historical_pdc': pdc,
            'refill_count': len(period_claims),
            'avg_refill_gap': refill_gaps.mean() if not refill_gaps.empty else 0,
            'std_refill_gap': refill_gaps.std() if not refill_gaps.empty else 0,
        })
        
    return pd.DataFrame(features_list)

# Define periods: Use 2024 data to predict adherence in early 2025
train_start = pd.to_datetime('2024-01-01')
train_end = pd.to_datetime('2024-12-31')
predict_start = pd.to_datetime('2025-01-01')
predict_end = pd.to_datetime('2025-03-31')

# Create historical features (X)
X_features = calculate_pdc_and_features(claims_df, train_start, train_end)

# Create future target (y)
y_target = calculate_pdc_and_features(claims_df, predict_start, predict_end)
y_target['is_adherent_future'] = (y_target['historical_pdc'] >= 0.8).astype(int)

# Merge features and target
model_data = pd.merge(X_features, y_target[['member_id', 'is_adherent_future']], on='member_id', how='inner')
model_data = pd.merge(model_data, members_df, on='member_id', how='inner')
model_data = model_data.fillna(0) # Fill NaNs from std calc for members with 1 refill

print("\n--- Final Modeling Data Head ---")
print(model_data.head())

features = ['historical_pdc', 'refill_count', 'avg_refill_gap', 'std_refill_gap', 'age', 'num_chronic_conditions']
target = 'is_adherent_future'

X = model_data[features]
y = model_data[target]

class_counts = y.value_counts()
print("\n--- Class Distribution ---")
print(class_counts)

if class_counts.min() < 2:
    print("\nError: The smallest class has fewer than 2 members, which prevents stratified splitting.")
    print("This can happen with simulated data. Try increasing 'num_members' at the top of the script.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate performance
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n--- Model Evaluation ---")
print(f"Model AUC Score: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

print("\n--- Feature Importances ---")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance for Predicting Medication Non-Adherence')
plt.tight_layout()
plt.show()

# 6. Generate Output for Outreach Team
# The score is the probability of being NON-ADHERENT, so we take 1 - P(adherent)
model_data['risk_of_non_adherence_score'] = 1 - model.predict_proba(X)[:, 1]
output_df = model_data[['member_id', 'age', 'num_chronic_conditions', 'risk_of_non_adherence_score']].sort_values('risk_of_non_adherence_score', ascending=False)

output_df.to_csv('member_adherence_risk_list.csv', index=False)
print("\n--- Outreach List Generated ---")
print("Saved 'member_adherence_risk_list.csv' for the clinical outreach team.")
print(output_df.head())


