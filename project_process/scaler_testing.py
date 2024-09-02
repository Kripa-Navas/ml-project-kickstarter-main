from sklearn.preprocessing import MinMaxScaler
# Prepare and train the logistic regression model with MinMaxScaler
def prepare_and_train_minmax(data, feature_to_use):
    X = data[['Goal', 'duration_days', feature_to_use] + [col for col in data.columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country')]]
    y = data['State']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, matrix

# Test the model with MinMaxScaler under Assumption 3 with Pledged
accuracy_pledged_3_minmax, report_pledged_3_minmax, matrix_pledged_3_minmax = prepare_and_train_minmax(data_assumption_3, 'Pledged')

# Print results
print("Assumption 3 - Pledged with MinMaxScaler:", accuracy_pledged_3_minmax)
print(report_pledged_3_minmax)


