import pandas as pd
import numpy as np

# --- Configuration ---
NUM_PARTNERS = 5000 # The number of partners to simulate

# --- Data Generation Function ---
def generate_equitable_dataset(num_records):
    """
    Generates a rich, simulated dataset designed to test and showcase fairness mitigation.
    """
    
    # --- 1. Partner Profile & Sensitive Attributes ---
    data = {
        'partner_id': range(1, num_records + 1),
        'tenure_months': np.random.randint(1, 60, size=num_records),
        'city_district': np.random.choice(['North', 'South', 'East', 'West'], size=num_records, p=[0.25, 0.3, 0.2, 0.25]),
    }
    df = pd.DataFrame(data)

    # --- 2. Core Performance & Behavioral Metrics ---
    df['avg_customer_rating'] = np.random.uniform(4.5, 5.0, size=num_records).round(2)
    df['safety_score'] = np.random.uniform(70, 100, size=num_records).round(1)
    df['avg_weekly_trips'] = np.random.randint(20, 150, size=num_records)
    df['grab_pay_usage_rate'] = np.random.uniform(0.1, 0.9, size=num_records).round(2)
    
    # --- 3. Advanced Reliability & Financial Metrics ---
    df['acceptance_rate'] = np.random.uniform(0.85, 1.0, size=num_records).round(2)
    df['cancellation_rate'] = np.random.uniform(0.01, 0.15, size=num_records).round(2)
    df['peak_hour_percentage'] = np.random.uniform(0.2, 0.8, size=num_records).round(2)
    
    # Simulate earnings with realistic random noise
    base_earnings = 10000
    df['avg_weekly_earnings'] = (
        base_earnings + 
        (df['avg_weekly_trips'] * 50) + 
        ((df['avg_customer_rating'] - 4.5) * 1000) +
        ((df['peak_hour_percentage'] - 0.5) * 2000)
    ).round(2)
    
    # A moderate amount of randomness to reflect real-world variance
    df['avg_weekly_earnings'] += np.random.normal(0, 1500, size=num_records)
    
    # Create the 'earnings_stability_score'
    earnings_variance = np.random.normal(loc=2000, scale=1000, size=num_records).round(2)
    df['earnings_stability_score'] = np.maximum(0, earnings_variance)

    # --- 4. Simulate the Target Variable: loan_default ---
    # This logic creates a balanced signal for the model to learn from.
    score = (
        (df['avg_customer_rating'] - 4.7) * 8 +
        (df['tenure_months'] / 24.0) * 1.5 +
        (df['safety_score'] - 85) / 5 +
        (df['acceptance_rate'] - 0.9) * 15 -
        (df['cancellation_rate'] - 0.05) * 30 -
        (df['earnings_stability_score'] / 1500) * 1.5 -
        (df['peak_hour_percentage'] - 0.5) * 3 +
        (df['grab_pay_usage_rate'] - 0.5) * 2
    )
    
    # Introduce a moderate, realistic bias for the model to find and mitigate
    score[df['city_district'] == 'South'] -= 0.75

    prob_repay = 1 / (1 + np.exp(-score))
    df['loan_default'] = 1 - np.random.binomial(1, prob_repay, size=num_records)

    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    equitable_df = generate_equitable_dataset(NUM_PARTNERS)
    
    # Save the final dataset
    output_path = 'data/simulated_partner_data.csv'
    equitable_df.to_csv(output_path, index=False)
    
    print("✅ Equitable data generation complete.")
    print(f"Generated {len(equitable_df.columns)} columns for {len(equitable_df)} partners.")
    print(f"Data saved to: {output_path}")
    print("\nColumns in the dataset:")
    print(equitable_df.columns.tolist())

