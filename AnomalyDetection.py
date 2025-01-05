import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data_path = "dataset.csv"  #file path
data = pd.read_csv(data_path)  # Read the dataset from the CSV file

# Step 1: Inspect the dataset
print("Dataset Shape:", data.shape)  # Output the shape of the dataset (rows, columns)
print(data.info())  # Output dataset information including data types and missing values
print(data.head())  # Display the first few rows of the dataset to inspect the data

# Step 2: Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'Time', 'Unnamed: 37', 'Unnamed: 38', 'snort_alert_type',
                   'AL_Payload']  # List columns that are unnecessary (based on your dataset)
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])  # Drop these columns if they exist

# Step 3: Handle missing values
# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns  # Select categorical columns

# Fill missing values for numeric columns (using mean)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns (using mode for the most frequent value)
for col in categorical_cols:
    if not data[col].isnull().all():  # Check if the column has missing values
        data[col] = data[col].fillna(data[col].mode()[0])  # Replace NaNs with the most frequent value (mode)

# Step 4: Encode categorical variables
label_encoders = {}  # Dictionary to store label encoders for each categorical column
for col in categorical_cols:
    le = LabelEncoder()  # Create a new label encoder
    data[col] = le.fit_transform(data[col])  # Encode the categorical data into numerical values
    label_encoders[col] = le  # Store the encoder for future use

# Step 5: Multiclass Classification
if 'attack_type' in data.columns:
    print("\n### Multiclass Classification ###")

    # Features (X) and target (y)
    X = data.drop(columns=['attack_type'], errors='ignore')  # Features are all columns except the target column 'attack_type'
    y = data['attack_type']  # Target is the 'attack_type' column

    # Redefine numeric_cols based on updated X (after dropping 'attack_type')
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Ensure no NaNs in X before scaling
    X[numeric_cols] = X[numeric_cols].fillna(0)  # Replace NaNs with 0 or any suitable value

    # Scale numeric features using StandardScaler
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])  # Normalize the numeric columns

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the RandomForestClassifier model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)  # Fit the model on the training data

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    accuracy = (cm.diagonal().sum()) / cm.sum()  # Calculate accuracy from confusion matrix
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy percentage

    # Define target names (custom labels for your 'attack_type' classes)
    target_names = ['UC1', 'UC2', 'UC3', 'UC4']

    # Print classification report with custom labels
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Plot confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Step 6: Anomaly Detection with Isolation Forest
print("\n### Anomaly Detection ###")

# Normalize the data (standardize features)
X_scaled = scaler.fit_transform(X)

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_scaled)  # Fit the model on the scaled features

# Predict anomalies
anomaly_scores = iso_forest.decision_function(X_scaled)  # Get the anomaly scores
anomaly_labels = iso_forest.predict(X_scaled)  # Predict the anomaly labels (-1 for anomalies, 1 for normal)

# Convert anomaly labels: -1 -> anomaly, 1 -> normal
anomalies_detected = (anomaly_labels == -1).sum()  # Count the number of anomalies
normal_detected = (anomaly_labels == 1).sum()  # Count the number of normal points
total_samples = X.shape[0]  # Get the total number of samples

print(f"Total samples: {total_samples}")
print(f"Anomalies detected: {anomalies_detected}")
print(f"Normal points detected: {normal_detected}")
print(f"Percentage of anomalies: {anomalies_detected / total_samples * 100:.2f}%")
print(f"Percentage of normal points: {normal_detected / total_samples * 100:.2f}%")

# Pie chart for anomalies vs normal points
plt.figure(figsize=(8, 8))
labels = ['Anomalies', 'Normal Points']
sizes = [anomalies_detected, normal_detected]
colors = ['red', 'green']
explode = (0.1, 0)  # explode the anomalies slice slightly
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Anomalies vs Normal Points')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
