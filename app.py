import streamlit as st
import numpy as np
import requests
from scipy.stats import poisson

API_KEY = st.secrets["API_FOOTBALL_KEY"]

BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

def get_today_matches():
    import datetime
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/fixtures?date={today}"
    response = requests.get(url, headers=headers)
    return response.json()

def tau(i, j, lambda_h, lambda_a, rho):
    if i == 0 and j == 0:
        return 1 - (lambda_h * lambda_a * rho)
    elif i == 0 and j == 1:
        return 1 + (lambda_h * rho)
    elif i == 1 and j == 0:
        return 1 + (lambda_a * rho)
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1

def dixon_coles(lambda_h, lambda_a, rho=0.05, max_goals=6):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            base = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            matrix[i][j] = base * tau(i,j,lambda_h,lambda_a,rho)
    return matrix

def value(prob, odds):
    return (prob * odds) - 1

st.title("⚽ Football Quant PRO - Dixon Coles")

home_xg = st.number_input("Home xG", 0.1, 5.0, 1.5)
away_xg = st.number_input("Away xG", 0.1, 5.0, 1.2)
rho = st.slider("Rho (Dixon Coles)", 0.0, 0.2, 0.05)

matrix = dixon_coles(home_xg, away_xg, rho)

home_prob = np.sum(np.tril(matrix, -1))
draw_prob = np.sum(np.diag(matrix))
away_prob = np.sum(np.triu(matrix, 1))

st.subheader("Probabilidades Modelo")
st.write({
    "Home": round(home_prob,3),
    "Draw": round(draw_prob,3),
    "Away": round(away_prob,3)
})

home_odds = st.number_input("Home Odds", 1.0, 10.0, 2.1)

val = value(home_prob, home_odds)

st.subheader("Value Bet")
st.write("Value:", round(val,3))

if val > 0:
    st.success("🔥 Apuesta con valor positivo")
