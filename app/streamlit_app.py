import streamlit as st
import requests
import json
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Nova | GrabHack",
    page_icon="🚀",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main {background-color: #F8FAFC;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .score-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .score-card:hover {transform: translateY(-5px);}
    .metric-label {font-size: 1.2rem; color: #374151; font-weight: 600;}
    .nova-header {font-size: 2.5rem; font-weight: bold; color: #00B14F;}
    .nova-subheader {font-size: 1.1rem; color: #6B7280;}
            
    
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='nova-header'>✨ Project Nova</h1>", unsafe_allow_html=True)
st.markdown("<p class='nova-subheader'>Interactive & Equitable Credit Scoring Engine - GrabHack 2025</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("🧩 Partner Profile Simulator")

# Profile
st.sidebar.subheader("🏙️ Profile")
city_district = st.sidebar.selectbox("Primary City District", ['East', 'North', 'South', 'West'])
tenure_months = st.sidebar.slider("⏳ Tenure (Months)", 1, 60, 24)

# Performance
st.sidebar.subheader("⭐ Performance")
avg_customer_rating = st.sidebar.slider("Avg. Customer Rating", 4.5, 5.0, 4.8, 0.01)
safety_score = st.sidebar.slider("Safety Score", 70.0, 100.0, 85.0, 0.1)
earnings_stability_score = st.sidebar.slider("Earnings Stability Score", 0, 4000, 2000, 50)

# Financials
st.sidebar.subheader("💰 Financials")
avg_weekly_earnings = st.sidebar.slider("Avg. Weekly Earnings (INR)", 5000, 25000, 15000, 100)
avg_weekly_trips = st.sidebar.slider("Avg. Weekly Trips", 20, 150, 80)

# Engagement & Reliability
st.sidebar.subheader("📊 Engagement & Reliability")
acceptance_rate_percent = st.sidebar.slider("Acceptance Rate (%)", 85, 100, 95)
cancellation_rate_percent = st.sidebar.slider("Cancellation Rate (%)", 1, 15, 5)
peak_hour_percentage_percent = st.sidebar.slider("Peak Hour Driving (%)", 10, 100, 50)
grab_pay_usage_rate_percent = st.sidebar.slider("GrabPay Usage Rate (%)", 10, 90, 60)

# Convert percentages to 0-1 for API
acceptance_rate = acceptance_rate_percent / 100
cancellation_rate = cancellation_rate_percent / 100
peak_hour_percentage = peak_hour_percentage_percent / 100
grab_pay_usage_rate = grab_pay_usage_rate_percent / 100

# --- Input Data for API ---
input_data = {
    "tenure_months": tenure_months,
    "city_district": city_district,
    "avg_customer_rating": avg_customer_rating,
    "safety_score": safety_score,
    "avg_weekly_earnings": avg_weekly_earnings,
    "avg_weekly_trips": avg_weekly_trips,
    "acceptance_rate": acceptance_rate,
    "cancellation_rate": cancellation_rate,
    "peak_hour_percentage": peak_hour_percentage,
    "grab_pay_usage_rate": grab_pay_usage_rate,
    "earnings_stability_score": earnings_stability_score
}

# --- Main Dashboard ---
st.header("📊 Prediction Dashboard")

if st.button("🚀 Calculate Nova Score"):
    with st.spinner("⚡ Crunching numbers with AI..."):
        try:
            api_url = "http://127.0.0.1:5000/predict"
            response = requests.post(api_url, data=json.dumps(input_data), headers={'Content-Type': 'application/json'})

            if response.status_code == 200:
                result = response.json()

                # Safely parse repayment probability
                repayment_prob = result['repayment_probability']
                if isinstance(repayment_prob, str):
                    repayment_prob = repayment_prob.replace('%','')
                repayment_prob = float(repayment_prob)  # ensure numeric

                # --- Layout Columns ---
                col1, col2 = st.columns(2)

                # --- Nova Score Gauge ---
                with col1:
                    st.markdown("<div class='score-card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Nova Score</p>", unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['nova_score'],
                        title={'text': "Credit Score"},
                        gauge={
                            'axis': {'range': [300, 850]},
                            'bar': {'color': "#00B14F"},
                            'steps': [
                                {'range': [300, 580], 'color': "#F87171"},
                                {'range': [580, 670], 'color': "#FBBF24"},
                                {'range': [670, 850], 'color': "#34D399"}]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # --- Repayment Probability + Radar Chart ---
                with col2:
                    st.markdown("<div class='score-card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Repayment Probability</p>", unsafe_allow_html=True)
                    st.progress(int(repayment_prob))
                    st.write(f"**{repayment_prob:.2f}%** chance of repayment")

                    # Radar Chart
                    categories = ['Safety', 'Reliability', 'Engagement', 'GrabPay', 'Earnings Stability']
                    values = [
                        safety_score,
                        acceptance_rate_percent,
                        peak_hour_percentage_percent,
                        grab_pay_usage_rate_percent,
                        min(100, earnings_stability_score/50)  # scale to ~100
                    ]
                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Partner Profile',
                        line_color="#00B14F"
                    ))
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
                    st.plotly_chart(radar_fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # --- Success Feedback ---
                st.balloons()
                st.success("⚖️ Fairness Guarantee: Score generated with bias mitigation for equitable outcomes.")

            else:
                st.error(f"Error from API: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error: Could not connect to API. Run `app.py` first.")

else:
    st.info("👈 Adjust parameters in the sidebar and click **Calculate Nova Score** to see results.")
