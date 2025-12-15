📌 What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to predict a continuous output based on one or more input variables.

Example:
✔ Predict height using weight
✔ Predict house price using area
✔ Predict marks using study hours

The idea is simple:
🔹 Find the "best-fitting straight line" that explains the relationship between input (x) and output (y).

Mathematically:
y^ = mx + b

Where:
x = input (weight)
y = output (height)
m = slope (how much y changes when x increases)
b = intercept (value of y when x = 0)
ŷ = predicted output

📌 Mathematical Intuition Behind Linear Regression
The goal is to find the best values of m and b such that the line fits the data points as closely as possible.
⭐ Step 1 — Prediction
For any weight x, predicted height:
y^ = mx + b

⭐ Step 2 — Error (Residual)
Difference between actual and predicted:
error = y - y^

⭐ Step 3 — Cost Function (Loss Function)
We use Mean Squared Error (MSE):
J(m,b) = 1/n ∑(y-y^)^2
This measures how bad the model is.

Goal of learning:
👉 Minimize the cost function → find best m and b.


📌 How Do We Find m and b? (Gradient Descent Intuition)
Gradient descent is an optimization algorithm that moves m and b step-by-step to reduce error.

Think of it like:
👉 You are walking downhill until you reach the lowest point (minimum error).

Update rules:
𝑚 := 𝑚 −𝛼 ∂𝐽/∂𝑚
b := b-𝛼 ∂b/∂J
Where:
α = learning rate (step size)

This process repeats until the error stops decreasing.

📌 Why “Linear” Regression?
Because the relationship between x and y is represented by a straight line.

If data is non-linear (curved), linear regression will NOT fit well.

📌 Assumptions of Linear Regression

1️⃣ Linear relationship between input and output
2️⃣ Errors are normally distributed
3️⃣ No or minimal multicollinearity
4️⃣ Homoscedasticity (equal variance of errors)
5️⃣ Independent observations


📌 Performance Metrics for Linear Regression

Important metrics:
⭐ 1. Mean Squared Error (MSE)
MSE = 1/n ∑(y-y^)^2
Lower = better.

⭐ 2. Root Mean Squared Error (RMSE)
RMSE = SQRT(MSE)
Also lower = better.

⭐ 3. Mean Absolute Error (MAE)
MAE = 1/n ∑|y-y^|
Less sensitive to outliers.

⭐ 4. R² Score (Coefficient of Determination)
𝑅**2 = 1-SS(res)/SS(tot)

Where:
SS(res) = error of model
SS(tot) = total variation

R² tells you how much of the variation in Y your model explains.

Values:
1 → perfect model
0 → model explains nothing
Negative → horrible model

📌 When to Use Linear Regression
✔ Predicting continuous values
✔ Relationship is approximately linear
✔ Small to medium datasets
✔ Need simple, interpretable model

📌 When NOT to Use Linear Regression
❌ Data shows a curved pattern
❌ Many outliers
❌ Strong multicollinearity
❌ Features interact non-linearly
❌ You need high accuracy for complex problems

📌 Summary:
Linear Regression finds the best-fitting straight line between input and output variables. It works by minimizing the Mean Squared Error using optimization methods like Gradient Descent. The performance is evaluated using metrics like MSE, RMSE, MAE, and R².

🎤 Interview Answer (Use This Exactly)
“Linear Regression is a supervised ML algorithm used to predict continuous values by modeling a linear relationship between the input and target variable. For example, if I want to predict height using weight, Linear Regression will try to draw the best-fit straight line that minimizes prediction error.
The equation is ŷ = mx + b, where m is slope and b is intercept.
The model is trained by minimizing the Mean Squared Error using optimization techniques like gradient descent.
We evaluate performance using MSE, RMSE, MAE, and R² score.
Linear Regression works well when the relationship is roughly linear, there are no major outliers, and variance is stable. It is simple, interpretable, and widely used as a baseline model.”

📌 Linear Regression: Predict Height from Weight
This project builds a linear regression model that predicts a person's height based on their weight.

🔥 Dataset
Weight → Independent variable
Height → Dependent variable

Stored in:
data/height_weight.csv

🧠 Tech Stack
Python
Pandas
Scikit-learn
Matplotlib
Joblib

🚀 How to Run
pip install -r requirements.txt
python train.py
python app.py

📊 Output
Trained model saved in /model/height_predictor.pkl
Regression plot saved as /model/plot.png

