"""
Script to retrain the stress level model with the current scikit-learn version.
This will create a new stresslevel.pkl file compatible with scikit-learn 1.5.2
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading training data...")
# Load the training data
train_df = pd.read_csv('dreaddit-train.csv', encoding='ISO-8859-1')

print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")

# Prepare features and target
# Remove text and identifier columns, keep only numeric features
X = train_df.drop(['text', 'post_id', 'sentence_range', 'id', 'social_timestamp', 'label'], axis=1, errors='ignore')
y = train_df['label']

# Check if there are categorical columns that need encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"Encoding categorical columns: {categorical_cols}")
    # Simple label encoding for categorical columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes

print(f"\nFeatures shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining model...")
# You can choose different models - uncomment the one you want to use

# Option 1: Random Forest (generally good performance)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Option 2: Logistic Regression (simpler, faster)
# model = LogisticRegression(max_iter=1000, random_state=42)

# Option 3: Decision Tree
# model = DecisionTreeClassifier(random_state=42)

# Option 4: K-Nearest Neighbors
# model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nModel: {type(model).__name__}")
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save the model
print("\nSaving model to stresslevel.pkl...")
with open('stresslevel.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")
print(f"Compatible with scikit-learn version: {__import__('sklearn').__version__}")

# Test loading the model
print("\nTesting model loading...")
with open('stresslevel.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("✅ Model loaded successfully!")

# Make a test prediction
sample_prediction = loaded_model.predict(X_test[:5])
print(f"\nSample predictions: {sample_prediction}")
