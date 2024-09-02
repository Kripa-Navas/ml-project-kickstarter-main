#Hypertuning Baseline Model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Prepare the data
X = data_assumption_3[['Goal', 'duration_days', 'Pledged'] + 
                      [col for col in data_assumption_3.columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country')]]
y = data_assumption_3['State']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = LogisticRegression(random_state=42, max_iter=1000)

# Define the parameter grid 
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}


# Grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate on the test data
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test_scaled)
accuracy_test = accuracy_score(y_test, y_test_pred)
report_test = classification_report(y_test, y_test_pred)

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)
print("Test Data Accuracy:", accuracy_test)
print(report_test)

#learning curve for baseline model
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate learning curves
train_sizes, train_scores, test_scores = learning_curve(xgb_model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.plot(train_sizes, train_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_mean, 'o-', label="Cross-Validation Score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend(loc="best")
plt.show()



