
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("trained_model.xgb", "rb") as f:
    model = pickle.load(f)

st.title("Dream11 Best XI Predictor - RCB vs RR")

st.markdown("Upload a CSV or enter player data below (Name, Team, Avg Points, Batting Order, Pitch)")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Expecting columns: player_name, team, avg_points, batting_order, pitch")
    df = pd.DataFrame(columns=["player_name", "team", "avg_points", "batting_order", "pitch"])

if not df.empty:
    df['bat_order_score'] = (12 - df['batting_order']) * 1.5
    df['venue_bias'] = df['pitch'].map({'batting': 5, 'bowling': -5, 'neutral': 0}).fillna(0)
    df['feature'] = df['avg_points'] + df['bat_order_score'] + df['venue_bias']
    df['predicted'] = model.predict(df[['feature']])

    best_xi = df.sort_values(by='predicted', ascending=False).head(11).copy()
    best_xi.iloc[0, best_xi.columns.get_loc('predicted')] *= 2  # Captain
    best_xi.iloc[1, best_xi.columns.get_loc('predicted')] *= 1.5  # Vice-captain
    best_xi.sort_values(by='predicted', ascending=False, inplace=True)

    st.subheader("Predicted Best XI")
    st.dataframe(best_xi[['player_name', 'team', 'avg_points', 'batting_order', 'predicted']])

    st.download_button(
        label="Download as Excel",
        data=best_xi.to_excel(index=False, engine='openpyxl'),
        file_name="best_xi_prediction.xlsx"
    )
