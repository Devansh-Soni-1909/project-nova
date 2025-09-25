import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Load the pre-trained model
try:
    # The path needs to go up one level from 'app' to the root folder
    with open('baseline_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'baseline_model.pkl' not found. Please run the notebook to train and save the model first.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on the server'}), 500

    try:
        # Get the 5 features from the simple HTML form
        data = request.get_json()
        input_df = pd.DataFrame(data, index=[0])

        # --- Smart Feature Engineering ---
        # This is the crucial part. We create all the other features the model expects.
        print("Received input:", data)

        # 1. Add missing behavioral features with average/default values
        input_df['safety_score'] = 85.0
        input_df['grab_pay_usage_rate'] = 0.5
        input_df['peak_hour_percentage'] = 0.5
        input_df['acceptance_rate'] = 0.95
        input_df['cancellation_rate'] = 0.05
        input_df['earnings_stability_score'] = 2000

        # 2. Create derived features, just like in the notebook
        input_df['earnings_per_trip'] = input_df['avg_weekly_earnings'] / input_df['avg_weekly_trips']
        input_df['rating_x_tenure'] = input_df['avg_customer_rating'] * input_df['tenure_months']

        # 3. One-hot encode the 'city_district' feature
        input_processed = pd.get_dummies(input_df, columns=['city_district'], drop_first=True)

        # 4. Align all columns perfectly with the model's training data
        training_cols = model.get_booster().feature_names
        for col in training_cols:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Ensure the column order is identical
        input_processed = input_processed[training_cols]
        
        print(f"Processed {len(input_processed.columns)} features for the model.")

        # --- Make Prediction and Calculate Nova Score ---
        repayment_prob = model.predict_proba(input_processed)[:, 0]
        nova_score = 300 + (repayment_prob[0] * 550)

        # Return the final score
        return jsonify({
            'nova_score': int(nova_score),
            'repayment_probability': f"{repayment_prob[0]:.2%}",
        })

    except Exception as e:
        # Provide a more detailed error for debugging
        print(f"❌ An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # We run on port 5000, which the HTML file expects
    app.run(port=5000, debug=True)
# ```

### Final Step: Relaunch and Test

# Now that you've updated `app.py`, just restart the backend server.

# 1.  In your VS Code terminal, run the command again:
#     ```bash
#     python app/app.py
    

