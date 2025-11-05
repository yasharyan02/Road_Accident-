import json
import joblib
import pandas as pd
import warnings
from flask import Flask, request, jsonify, render_template

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Columns ---
# We do this once when the app starts
try:
    # Suppress version warnings when loading the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = joblib.load('accident_model.joblib')
        
    model_columns = json.load(open('model_columns.json'))
    print("--- Model and columns loaded successfully. ---")
except Exception as e:
    print(f"XXX Error loading model or columns: {e} XXX")
    print("--- Server is running, but /predict WILL FAIL. ---")
    model = None
    model_columns = []

# --- Helper Function: Map prediction number to a label ---
def get_severity_label(prediction):
    """Converts the numeric prediction (1, 2, 3) to a readable string."""
    if prediction == 1:
        return 'Slight'
    elif prediction == 2:
        return 'Serious'
    elif prediction == 3:
        return 'Fatal'
    return 'Unknown' # Fallback

# --- Route 1: The Home Page ---
# This serves your index.html file
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- Route 2: The Prediction API ---
# This is where your HTML form sends its data
@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, processes it, and returns a prediction."""
    
    # Check if model loaded correctly
    if model is None or not model_columns:
        return jsonify({'error': 'Model is not loaded properly. Check server logs.'}), 500
    
    try:
        # 1. Get data from the form
        data = request.get_json()
        
        # 2. Initialize the "feature row"
        # Create a dictionary with all 40+ of your model's columns, all set to 0
        features = {col: 0 for col in model_columns}

        # 3. Update the dictionary with values from the form
        
        # --- Continuous/Numeric features ---
        features['Speed_limit'] = float(data['Speed_limit'])
        features['Hours'] = float(data['Hours'])
        features['Minute'] = float(data['Minute'])
        features['Latitude'] = float(data['Latitude'])
        features['Longitude'] = float(data['Longitude'])
        features['Number_of_Vehicles'] = int(data['Number_of_Vehicles'])

        # --- One-Hot Encoded (OHE) features ---
        # For each dropdown, we find the matching column name and set it to 1
        
        # Day of Week (e.g., DoW_Monday)
        day_key = f"DoW_{data['Day_of_Week']}"
        if day_key in features:
            features[day_key] = 1
        
        # Weather Conditions (e.g., Weather_Fog or mist, High_Wind)
        weather_val = data['Weather_Conditions']
        weather_key = ""
        if weather_val == "Fog or mist":
            weather_key = "Weather_Fog or mist"
        elif weather_val == "High_Wind":
            weather_key = "High_Wind" # This was a separate column
        elif weather_val in ["Raining", "Snowing", "Other"]:
            weather_key = f"Weather_{weather_val}"
        # 'Fine' means all weather flags are 0, so we do nothing.
        
        if weather_key in features:
            features[weather_key] = 1

        # Road Type (e.g., RT_Single_carriageway)
        road_val = data['Road_Type']
        road_key = f"RT_{road_val.replace(' ', '_')}"
        if road_key in features:
            features[road_key] = 1

        # Light Conditions (e.g., LC_Darkness_lights_lit)
        light_val = data['Light_Conditions']
        light_key = f"LC_{light_val}"
        if light_key in features:
            features[light_key] = 1
            
        # Junction Detail (e.g., JD_T_or_staggered_junction)
        junction_val = data['Junction_Detail']
        junction_key = f"JD_{junction_val}"
        if junction_key in features:
            features[junction_key] = 1
            
        # Vehicle Group (e.g., vg_Motorcycle)
        vehicle_val = data['Vehicle_Group']
        vehicle_key = f"vg_{vehicle_val}"
        if vehicle_key in features:
            features[vehicle_key] = 1
        # 'Car' is likely the default (all 0s), so we do nothing.

        # Carriageway Hazard (e.g., CH_Other object on road)
        hazard_val = data['Carriageway_Hazard']
        hazard_key = ""
        if hazard_val == "No Hazard":
            hazard_key = "CH_No Hazard"
        elif hazard_val == "Pedestrian in carriageway - not injured":
            hazard_key = "CH_Pedestrian in carriageway - not injured"
        elif hazard_val in ["Other object on road", "Previous accident", "Vehicle load on road"]:
            hazard_key = f"CH_{hazard_val.replace(' ', '_')}"
            
        if hazard_key in features:
            features[hazard_key] = 1

        # 4. Create the final DataFrame
        # Convert the dictionary into a pandas DataFrame
        # We pass `columns=model_columns` to ensure the column order is
        # exactly what the model was trained on.
        df = pd.DataFrame([features], columns=model_columns)
        
        # 5. Make the prediction
        prediction_val = model.predict(df)
        
        # 6. Format the output
        prediction_num = int(prediction_val[0])
        prediction_label = get_severity_label(prediction_num)

        # 7. Send the JSON response back to the HTML
        return jsonify({
            'prediction': prediction_num,
            'prediction_label': prediction_label
        })

    except Exception as e:
        # Handle any errors during the process
        print(f"XXX Error during prediction: {e} XXX")
        return jsonify({'error': f"Prediction error: {str(e)}"}), 400

# --- This runs the app ---
if __name__ == '__main__':
    # debug=True means the server will auto-reload when you save the file
    app.run(debug=True, port=5000)