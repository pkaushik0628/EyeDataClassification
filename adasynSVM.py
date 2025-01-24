import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the processed data
data = pd.read_csv('processedEyeData.csv')

# Print available columns for debugging
print("Available columns in the data:")
print(data.columns.tolist())

# Select relevant features containing 'Saccades' or 'time' in their names
saccades_features = data.filter(like='Saccades')
error_features = data.filter(like='_error')  # Filter for columns that contain '_time'
pe_features = data.filter(like='_time')
# Combine the features
features = pd.concat([saccades_features, error_features, pe_features], axis=1)

# Check if features is empty
if features.empty:
    print("Warning: No features found containing 'Saccades' or 'time'. Proceeding with an empty feature set.")
else:
    print("Filtered features:")
    print(features.head())
    print(f"Number of features selected: {features.shape[1]}")

target = data['Type']  # Adjust this based on your target variable column

# Check for NaN values and handle them (e.g., drop or fill)
features = features.fillna(0)  # Fill NaNs with 0; adjust this as needed
target = target.dropna()

# Make sure the target variable aligns with features
features = features.loc[target.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features if features are not empty
if not features.empty:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply ADASYN to balance the training data
    adasyn = ADASYN()
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)

    # Create and train the RUSBoostClassifier
    model = RUSBoostClassifier()
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Print the classification report
    print(classification_report(y_test, y_pred, zero_division=0))
else:
    print("No valid features to train the model.")

















