
my 1st AI model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Example dataset
data = {
    'Age': [25, 30, 35, 40, 45, 50],
    'Glucose': [85, 90, 95, 100, 105, 110],
    'Diabetic': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Split the data
X = df[['Age', 'Glucose']]
y = df['Diabetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict diabetes based on new user input
def predict_diabetes(age, glucose):
    input_data = pd.DataFrame([[age, glucose]], columns=['Age', 'Glucose'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Example prediction
new_age = 37
new_glucose = 97
result = predict_diabetes(new_age, new_glucose)
print("Predicted Outcome for age 37 and glucose 97:", result)
