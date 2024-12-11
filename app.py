from flask import Flask, request, render_template, session, redirect, url_for
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Replace with a secure secret key

# Load the model
model = joblib.load('flask app/model/best_random_forest_model.pkl')

# Define the distance matrix
distance_matrix = {
    ('Back Bay', 'Beacon Hill'): 1.2,
    ('Back Bay', 'Boston University'): 2.5,
    ('Back Bay', 'Fenway'): 1.0,
    ('Back Bay', 'Financial District'): 1.8,
    ('Back Bay', 'Haymarket Square'): 1.5,
    ('Back Bay', 'North End'): 2.0,
    ('Back Bay', 'North Station'): 1.7,
    ('Back Bay', 'Northeastern University'): 1.2,
    ('Back Bay', 'South Station'): 1.5,
    ('Back Bay', 'Theatre District'): 0.8,
    ('Back Bay', 'West End'): 1.6,
    ('Beacon Hill', 'Boston University'): 2.7,
    ('Beacon Hill', 'Fenway'): 1.5,
    ('Beacon Hill', 'Financial District'): 1.0,
    ('Beacon Hill', 'Haymarket Square'): 0.5,
    ('Beacon Hill', 'North End'): 1.0,
    ('Beacon Hill', 'North Station'): 0.7,
    ('Beacon Hill', 'Northeastern University'): 1.7,
    ('Beacon Hill', 'South Station'): 0.8,
    ('Beacon Hill', 'Theatre District'): 0.5,
    ('Beacon Hill', 'West End'): 0.6,
    ('Boston University', 'Fenway'): 1.2,
    ('Boston University', 'Financial District'): 3.0,
    ('Boston University', 'Haymarket Square'): 2.8,
    ('Boston University', 'North End'): 3.3,
    ('Boston University', 'North Station'): 3.0,
    ('Boston University', 'Northeastern University'): 1.5,
    ('Boston University', 'South Station'): 2.7,
    ('Boston University', 'Theatre District'): 2.5,
    ('Boston University', 'West End'): 2.9,
    ('Fenway', 'Financial District'): 2.0,
    ('Fenway', 'Haymarket Square'): 1.8,
    ('Fenway', 'North End'): 2.3,
    ('Fenway', 'North Station'): 2.0,
    ('Fenway', 'Northeastern University'): 0.5,
    ('Fenway', 'South Station'): 1.7,
    ('Fenway', 'Theatre District'): 1.5,
    ('Fenway', 'West End'): 1.9,
    ('Financial District', 'Haymarket Square'): 0.7,
    ('Financial District', 'North End'): 1.2,
    ('Financial District', 'North Station'): 0.9,
    ('Financial District', 'Northeastern University'): 2.2,
    ('Financial District', 'South Station'): 0.5,
    ('Financial District', 'Theatre District'): 1.0,
    ('Financial District', 'West End'): 0.8,
    ('Haymarket Square', 'North End'): 0.5,
    ('Haymarket Square', 'North Station'): 0.3,
    ('Haymarket Square', 'Northeastern University'): 2.0,
    ('Haymarket Square', 'South Station'): 0.8,
    ('Haymarket Square', 'Theatre District'): 0.7,
    ('Haymarket Square', 'West End'): 0.4,
    ('North End', 'North Station'): 0.5,
    ('North End', 'Northeastern University'): 2.5,
    ('North End', 'South Station'): 1.3,
    ('North End', 'Theatre District'): 1.2,
    ('North End', 'West End'): 0.9,
    ('North Station', 'Northeastern University'): 2.2,
    ('North Station', 'South Station'): 1.0,
    ('North Station', 'Theatre District'): 0.9,
    ('North Station', 'West End'): 0.6,
    ('Northeastern University', 'South Station'): 1.9,
    ('Northeastern University', 'Theatre District'): 1.7,
    ('Northeastern University', 'West End'): 2.1,
    ('South Station', 'Theatre District'): 0.7,
    ('South Station', 'West End'): 0.9,
    ('Theatre District', 'West End'): 0.8,
}

# Adding reverse pairs for symmetry
for (start, end), distance in list(distance_matrix.items()):
    distance_matrix[(end, start)] = distance

# Define the predict_price function
def predict_price(hour, cab_name, hour_bin, is_weekend, source, destination, day_of_week=None):
    # Mapping categorical values to numerical values
    cab_name_mapping = {'Black SUV': 0, 'Lux': 1, 'Shared': 2, 'Taxi': 3, 'UberPool': 4, 'UberX': 5}
    hour_bin_mapping = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
    source_mapping = {'Back Bay': 0, 'Beacon Hill': 1, 'Boston University': 2, 'Fenway': 3, 'Financial District': 4, 'Haymarket Square': 5, 'North End': 6, 'North Station': 7, 'Northeastern University': 8, 'South Station': 9, 'Theatre District': 10, 'West End': 11}
    
    # Calculate distance from the distance matrix
    distance = distance_matrix.get((source, destination), 0)  # Default to 0 if not found
    
    # Calculate fare per mile (assuming a base fare per mile, you can adjust this as needed)
    fare_per_mile = 6.0  # Example base fare per mile

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'distance': [distance],
        'fare_per_mile': [fare_per_mile],
        'hour': [hour],
        'cab_type': [cab_name_mapping[cab_name]],
        'hour_bin': [hour_bin_mapping[hour_bin]],
        'is_weekend': [is_weekend],
        'source': [source_mapping[source]],
        'destination': [source_mapping[destination]]
    })
    
    if day_of_week:
        input_data['day_of_week'] = [day_of_week]
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    return predicted_price[0], fare_per_mile

@app.route('/')
def home():
    prediction_text = session.pop('prediction_text', None)
    form_data = session.get('form_data', None)
    return render_template('index.html', prediction_text=prediction_text, form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    hour = int(request.form['hour'])
    cab_name = request.form['cab_name']
    hour_bin = request.form['hour_bin']
    is_weekend = int(request.form['is_weekend'])
    source = request.form['source']
    destination = request.form['destination']
    day_of_week = request.form.get('day_of_week', None)
    
    if day_of_week:
        day_of_week = day_of_week
    
    predicted_price, fare_per_mile = predict_price(hour, cab_name, hour_bin, is_weekend, source, destination, day_of_week)
    
    # Calculate the error margin (standard deviation of cross-validation RMSE scores)
    cv_rmse_scores = np.array([0.08667241, 0.11472886, 0.0741422, 0.07151226, 0.07850577]) 
    error_margin = cv_rmse_scores.std()
    
    session['prediction_text'] = f'Predicted Price: ${predicted_price:.2f} +/- ${error_margin:.2f}, Fare per Mile: ${fare_per_mile:.2f}'
    session['form_data'] = request.form.to_dict()
    
    return redirect(url_for('home'))

@app.route('/clear', methods=['POST'])
def clear():
    session.pop('prediction_text', None)
    session.pop('form_data', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)