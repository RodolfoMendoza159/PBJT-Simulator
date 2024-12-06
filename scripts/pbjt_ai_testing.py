import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle

# Import the functions for PBJT simulation
from pbjt_simulation import generate_dataset

# Load the trained AI model
with open("pbjt_large_dataset.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Generate 10 test samples (5 male, 5 female)
test_dataset = generate_dataset(num_samples=10)
test_dataset.to_csv("pbjt_test_dataset.csv", index=False)
print("Test dataset saved to 'pbjt_test_dataset.csv'")

# Load features and labels
X_test = test_dataset.drop(columns=["label"])
y_test = test_dataset["label"]

# Predict with the AI model
predictions = model.predict(X_test)

# Display results
for idx, pred in enumerate(predictions):
    print(f"Test Sample {idx + 1}: Predicted={pred}, Actual={y_test.iloc[idx]}")

# Calculate accuracy
accuracy = sum(predictions == y_test) / len(y_test)
print(f"AI Test Accuracy: {accuracy * 100:.2f}%")
