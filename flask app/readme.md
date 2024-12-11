# Uber Price Prediction Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Data](#data)
   - [Source](#source)
   - [Preprocessing](#preprocessing)
5. [Model](#model)
6. [Results](#results)
   - [Key Observations](#key-observations)
   - [Example Test Case](#example-test-case)
7. [Application](#application)
   - [How to Run](#how-to-run)
   - [Input Details](#input-details)
8. [Challenges](#challenges)
9. [Conclusions](#conclusions)
10. [Future Work](#future-work)
11. [Acknowledgements](#acknowledgements)
12. [Code Snippets](#code-snippets)
   - [Importing Libraries](#importing-libraries)
   - [Loading the Dataset](#loading-the-dataset)
   - [Preprocessing](#preprocessing-1)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
   - [Saving the Model](#saving-the-model)

## Project Overview
This project aims to predict Uber ride prices based on various factors such as time of day, cab type, source, destination, and other variables. Utilizing a supervised machine learning model, we provide predictions that closely align with real-world Uber app prices, particularly for the UberX category.

The project comprises a Flask-based web application that allows users to input trip details and receive a predicted price. The backend leverages a trained machine learning model, and the application design is structured to ensure an intuitive user experience.

---

## Features
1. **Price Prediction for Uber Rides**:
   - Supports multiple inputs such as time, day, cab type, and location.
   - The model predicts UberX fares accurately, but for other cab types, there's an average error of $2–$3.

2. **Real-Time Comparison**:
   - Predictions can be compared with live Uber app prices.

3. **Interactive Frontend**:
   - Designed with `HTML`, `CSS`, and Flask templates.

---

## File Structure

├── app.py               # Flask application code
├── analysis_new_Final_2.ipynb # Jupyter notebook for data analysis and model training
├── rideshare_kaggle.csv.zip   # Dataset used for training and evaluation
├── static/
│   └── style.css        # Stylesheet for the web app
├── templates/
│   └── index.html       # Frontend HTML template
├── README.md            # Project documentation

---

## Data
### Source
The dataset is sourced from Kaggle and contains details about ride-sharing trips, including:
- **Ride information**: cab type, time, distance, and fare.
- **Location**: source and destination areas.

### Preprocessing
- Data cleaning included handling missing values and feature engineering for time bins (e.g., Morning, Night).
- Cab types not covered in the analysis were excluded to focus on UberX predictions.

---

## Model
The predictive model was built using a supervised learning approach:
1. **Training**:
   - Data split into training and testing sets.
   - A regression model was trained to predict fares.

2. **Evaluation**:
   - Metrics like Mean Absolute Error (MAE) and R² were used to assess performance.

---

## Results
### Key Observations
- The model predicts UberX fares accurately but for other cab types there's an average error of $2–$3.
- Predictions for other cab types (e.g., Lux, Black SUV) showed higher variance, likely due to limited data representation.
- Predicted prices closely matched real-time Uber prices during testing.

### Example Test Case
| Feature          | Value                        |
|------------------|------------------------------|
| Hour             | 14                           |
| Cab Type         | UberX                        |
| Hour Bin         | Night                        |
| Is Weekend       | No                           |
| Source           | Boston University            |
| Destination      | Northeastern University      |

**Predicted Price**: $9.00 ± $0.02  
**Actual Uber App Price**: $9.02  

---

## Application
### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```bash
   python app.py
   ```

3. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in a browser.

### Input Details
- Provide ride details in the web app.
- Click Predict Price to generate predictions.

### Challenges
- Discrepancy in predictions for non-UberX cab types due to data imbalance.
- Difficulty in accurately capturing dynamic pricing changes.

### Conclusions
- The project demonstrates the feasibility of Uber price prediction using machine learning.
- While predictions for UberX are reliable, expanding the dataset and model could improve accuracy for other cab types.

### Future Work
- Incorporate real-time data for dynamic pricing.
- Enhance the model to predict fares for premium cab types.
- Deploy the app on a cloud platform for broader accessibility.

### Acknowledgements
- Kaggle for the dataset.
- Flask, Pandas, and Scikit-learn libraries for enabling this project.

---

## Code Snippets

### Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
```

### Loading the Dataset
```python
# Load the dataset
df = pd.read_csv('rideshare_kaggle.csv.zip')
```

### Preprocessing
```python
# Data cleaning and preprocessing
df.dropna(inplace=True)
df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df = df[df['cab_type'] == 'UberX']
```

### Training the Model
```python
# Splitting the data
X = df[['hour', 'distance', 'source', 'destination', 'hour_bin', 'is_weekend']]
y = df['fare']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)
```

### Evaluating the Model
```python
# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
```

### Saving the Model
```python
# Save the model
joblib.dump(model, 'uber_price_model.pkl')
````