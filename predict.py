# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris_data = load_iris()

# Split the data into features (X) and target (y)
X = iris_data.data
y = iris_data.target

# import pandas as pd
# data = pd.DataFrame(iris_data.data)
# print(data.head(5))
# # print(data.dim)

# # print(X.head(5))
# # print(iris_data.data[0][:])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier and train it on the training data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Save the model to a file
joblib.dump(model, 'iris_model.pkl')
