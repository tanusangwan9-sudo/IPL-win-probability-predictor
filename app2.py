import os
import pandas as pd
import streamlit as st
import joblib
import sklearn.compose._column_transformer

# Fix
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

    pipe = joblib.load("pipe.pkl")
else:
    st.error("pipe.pkl file not found")
# Page Config
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    color: #9b59b6;
    text-align: center;
}

.stButton>button {
    background-color: #9b59b6;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stSelectbox label {
    color: #9b59b6;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    background: linear-gradient(90deg,#8e44ad,#9b59b6);
    color: white;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)



teams = ['Royal Challengers Bangalore',
 'Kings XI Punjab',
 'Mumbai Indians',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Royal Challengers Bengaluru']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
       'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
       'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Indore', 'Bengaluru', 'Dubai', 'Sharjah', 'Navi Mumbai',
       'Lucknow', 'Guwahati', 'Mohali']

# Load model safely


try:
    pipe = joblib.load('pipe.pkl')
except:
    pipe = None

st.title("🏏 IPL Win Probability Predictor")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Select the batting team ", sorted(teams))

with col2:
    bowling_team = st.selectbox(' 🔵 Select the bowling team', sorted(teams))

selected_city = st.selectbox(' 📍 Select host city', sorted(cities))
target = st.number_input('target')



st.markdown("#### 📊 Match Situation")

col3, col4, col5 = st.columns(3)

with col3:
        score = st.number_input("🏏 Current Score")

with col4:
        overs = st.number_input("⏱️ Overs Completed")

with col5:
        wickets = st.number_input("❌ Wickets Out")


if st.button("🔮 Predict Probability"):

    if pipe is None:
        st.error("Model not loaded properly")

    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets

        crr = score / overs if overs != 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)

        loss = result[0][0]
        win = result[0][1]

        st.header(f"{batting_team} Win Probability : {round(win*100)}%")
        st.header(f"{bowling_team} Win Probability : {round(loss*100)}%")

        st.markdown("---")

        result_df = pd.DataFrame({
            "Batting Team": [batting_team],
            "Bowling Team": [bowling_team],
            "City": [selected_city],
            "Runs Left": [runs_left],
            "Balls Left": [balls_left],
            "Wickets Left": [wickets_left],
            "Win %": [round(win*100)],
            "Loss %": [round(loss*100)]
        })

        st.subheader("####📊 Match Summary Table")
        st.dataframe(result_df)
