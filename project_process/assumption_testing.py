

# Assumption 1: Treat duplicates as unique campaigns if all have failed
def handle_assumption_1(data):
    # Identify duplicate names
    duplicates = data[data.duplicated(subset='Name', keep=False)]
    
    # Filter duplicates where all campaigns failed
    failed_duplicates = duplicates.groupby('Name').filter(lambda x: x['State'].sum() == 0)
    
    # Combine non-duplicates with the filtered duplicates
    non_duplicates = data.drop_duplicates(subset='Name', keep=False)
    result = pd.concat([non_duplicates, failed_duplicates])
    
    return result

# Assumption 2: Aggregate duplicates (mean for numerical features, max for binary)
def handle_assumption_2_adjusted(data):
    current_columns = data.columns.tolist()
    columns_to_aggregate = [
        'Goal', 'duration_days', 'Pledged', 'Backers', 'State'
    ] + [col for col in current_columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country') or col == 'name_counter']

    aggregated_data = data.groupby('Name').agg({col: 'mean' if col != 'State' else 'max' for col in columns_to_aggregate}).reset_index()
    return aggregated_data

# Assumption 3: Take the most recent campaign
def handle_assumption_3(data):
    return data.sort_values(by='Launched', ascending=False).drop_duplicates(subset='Name', keep='first')

# Apply the assumptions
data_assumption_1 = handle_assumption_1(kickstarter_data)
data_assumption_2_adjusted = handle_assumption_2_adjusted(kickstarter_data)
data_assumption_3 = handle_assumption_3(kickstarter_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare and train the logistic regression model
def prepare_and_train(data, feature_to_use):
    X = data[['Goal', 'duration_days', feature_to_use] + [col for col in data.columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country')]]
    y = data['State']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, matrix


# Test models for each assumption
accuracy_pledged_1, report_pledged_1, matrix_pledged_1 = prepare_and_train(data_assumption_1, 'Pledged')
accuracy_backers_1, report_backers_1, matrix_backers_1 = prepare_and_train(data_assumption_1, 'Backers')
accuracy_pledged_2, report_pledged_2, matrix_pledged_2 = prepare_and_train(data_assumption_2_adjusted, 'Pledged')
accuracy_backers_2, report_backers_2, matrix_backers_2 = prepare_and_train(data_assumption_2_adjusted, 'Backers')
accuracy_pledged_3, report_pledged_3, matrix_pledged_3 = prepare_and_train(data_assumption_3, 'Pledged')
accuracy_backers_3, report_backers_3, matrix_backers_3 = prepare_and_train(data_assumption_3, 'Backers')


print("Assumption 1 - Pledged:", accuracy_pledged_1)
print(report_pledged_1)
print("Assumption 1 - Backers:", accuracy_backers_1)
print(report_backers_1)

print("Assumption 2 - Pledged:", accuracy_pledged_2)
print(report_pledged_2)
print("Assumption 2 - Backers:", accuracy_backers_2)
print(report_backers_2)

print("Assumption 3 - Pledged:", accuracy_pledged_3)
print(report_pledged_3)
print("Assumption 3 - Backers:", accuracy_backers_3)
print(report_backers_3)

# Since assumption 1 and backers performed poorly, testing aassumption 2 and 3 with Pledged 
def prepare_and_train(data, feature_to_use):
    X = data[['Goal', 'duration_days', feature_to_use] + [col for col in data.columns if col.startswith('Category') or col.startswith('Subcategory') or col.startswith('Country')]]
    y = data['State']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, matrix

# Test models for Assumption 2 and 3 with Pledged
accuracy_pledged_2, report_pledged_2, matrix_pledged_2 = prepare_and_train(data_assumption_2_adjusted, 'Pledged')
accuracy_pledged_3, report_pledged_3, matrix_pledged_3 = prepare_and_train(data_assumption_3, 'Pledged')

# Print results
print("Assumption 2 - Pledged:", accuracy_pledged_2)
print(report_pledged_2)

print("Assumption 3 - Pledged:", accuracy_pledged_3)
print(report_pledged_3)

