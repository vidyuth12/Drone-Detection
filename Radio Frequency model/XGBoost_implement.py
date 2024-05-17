import numpy as np
import pickle

# Load the saved model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function for real-time drone detection


def detect_drone(input_data):
    # Preprocess the input data (assuming it's similar to the training data)
    input_data = np.transpose(input_data)

    # Make predictions using the loaded model
    y_pred = model.predict(input_data)
    predictions = [round(value) for value in y_pred]

    # Return the predictions
    return predictions


# Example usage
# Replace 'input_data' with the actual RF signal data you want to classify
input_data = np.loadtxt('path/to/input_data.csv', delimiter=',')
predictions = detect_drone(input_data)
print(predictions)
