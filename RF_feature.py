
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def preprocess():
    # Load dataset
    data = pd.read_csv("dataset.csv")
    data[' Label'] = data[' Label'].apply({
            'DoS':
            'Anormal',
            'BENIGN':
            'Normal',
            'DDoS':
            'Anormal',
            'PortScan':
            'Anormal'
        }.get)

    # Drop rows with NaN or infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Separate features and labels
    X = data.drop(columns=[' Label'])
    y = data[' Label']
    # Standardize numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Encode non-numerical columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
        
    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Fit the model to the data
    rf_classifier.fit(X, y)

    # Get feature importances from the model
    feature_importances = rf_classifier.feature_importances_


    # Create a DataFrame to store feature names and their corresponding importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print the top 10 important features
    top_10_features = feature_importance_df.head(10)
    print(top_10_features)
    # 37            Bwd Packets/s    0.073150
    # 13    Bwd Packet Length Std    0.072391
    # 39        Max Packet Length    0.061680
    # 40       Packet Length Mean    0.053955
    # 42   Packet Length Variance    0.053221
    # 54     Avg Bwd Segment Size    0.045470
    # 10    Bwd Packet Length Max    0.041707
    # 0          Destination Port    0.039743
    # 52      Average Packet Size    0.031108
    # 12   Bwd Packet Length Mean    0.030731

    # Sort features by importance and select top 10 features
    top_10_indices = feature_importances.argsort()[-10:][::-1]
    top_10_features = X.columns[top_10_indices]
    print(top_10_features)
    # Keep only the top 10 features in the dataset
    X_top_10 = X[top_10_features]

    # Now X_top_10 contains only the top 10 features according to their importance in the Random Forest model
    print(X_top_10.head())
    return X_top_10,y