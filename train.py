import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib

#step1 : Load Dataset
df = pd.read_csv("data/height-weight.csv")

#step2 : Select features
X = df[['Weight']]  #independent
y = df['Height']    #dependent

#step3 : split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#step4 : Train model
model = LinearRegression()
model.fit(X_train,y_train)

#step5 :Evaluate
print("Model Coefficient (Slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)

score = model.score(X_test,y_test)
print("Accuracy (R^2 Score):",score)

#step6: Save the model
joblib.dump(model,"model/height_predictor.pkl")
print("Model saved successfully!")

#step7: Visualize regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("Height vs Weught Linear Regression")
plt.savefig("model/plot.png")
plt.show()