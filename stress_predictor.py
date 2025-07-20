import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# ✅ Generate simulated training data
np.random.seed(42)  # Makes results reproducible
data = {
    'mood_score': np.random.uniform(1, 5, 1000),
    'journal_sentiment': np.random.uniform(-1, 1, 1000),
    'app_usage': np.random.randint(0, 10, 1000),
    'stress_level': np.random.choice(['Low', 'Moderate', 'High'], 1000)
}
df = pd.DataFrame(data)

# ✅ Encode stress levels into numeric labels
df['label'] = df['stress_level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

# ✅ Prepare features and target variable
X = df[['mood_score', 'journal_sentiment', 'app_usage']]
y = df['label']

# ✅ Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# ✅ Save the model to the backend folder
model_path = os.path.join("..", "server", "stress_predictor.pkl")
pickle.dump(clf, open(model_path, "wb"))

print("✅ Model trained and saved to:", model_path)
