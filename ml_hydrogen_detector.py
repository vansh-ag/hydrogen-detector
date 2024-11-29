import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'hydrogen_leakage_data.csv'
hydrogen_data = pd.read_csv(file_path)

# Drop the 'Timestamp' column
hydrogen_data_cleaned = hydrogen_data.drop(columns=['Timestamp'])

# Encode the 'Leakage Status' column
label_encoder = LabelEncoder()
hydrogen_data_cleaned['Leakage Status'] = label_encoder.fit_transform(hydrogen_data_cleaned['Leakage Status'])

# Separate features and target variable
X = hydrogen_data_cleaned.drop(columns=['Leakage Status'])
y = hydrogen_data_cleaned['Leakage Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report_output)
