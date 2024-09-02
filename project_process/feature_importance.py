# Prepare the data
X = data_assumption_3[['Goal', 'duration_days', 'Pledged'] + 
                      [col for col in data_assumption_3.columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country')]]
y = data_assumption_3['State']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title('Top 10 Feature Importance - Weight')
plt.show()

plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)
plt.title('Top 10 Feature Importance - Gain')
plt.show()

plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, importance_type='cover', max_num_features=10)
plt.title('Top 10 Feature Importance - Cover')
plt.show()

# Top 5 Feature Importance
def feature_importance_rank(clf, X_train):
    feature_scores_dtc = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).nlargest(5)
    print(feature_scores_dtc)
