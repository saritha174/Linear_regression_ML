import joblib

model = joblib.load("model/height_predictor.pkl")

weight = (input("Enter weight in kg: "))
weight = (int(weight))
predicted_height = model.predict([[weight]])

print("Predicted Height:", predicted_height[0])
