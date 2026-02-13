import streamlit as st
import requests
import numpy as np
from math import exp, factorial

# =============================
# CONFIG
# =============================

st.set_page_config(page_title="Football Quant Pro", layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# =============================
# FUNCIONES API
# =============================

@st.cache_data(ttl=3600)
def get_countries():
    url = f"{BASE_URL}/countries"
    r = requests.get(url, headers=HEADERS)
    data = r.json()["response"]
    return sorted([c["name"] for c in data if c["name"]])


@st.cache_data(ttl=3600)
def get_leagues(country):
    url = f"{BASE_URL}/leagues?country={country}&season=2024"
    r = requests.get(url, headers=HEADERS)
    data = r.json()["response"]
    leagues = {}
    for l in data:
        leagues[l["league"]["name"]] = l["league"]["id"]
    return leagues


@st.cache_data(ttl=3600)
def get_teams(league_id):
    url = f"{BASE_URL}/teams?league={league_id}&season=2024"
    r = requests.get(url, headers=HEADERS)
    data = r.json()["response"]
    teams = {}
    for t in data:
        teams[t["team"]["name"]] = t["team"]["id"]
    return teams


@st.cache_data(ttl=3600)
def get_team_stats(league_id, team_id):
    url = f"{BASE_URL}/teams/statistics?league={league_id}&season=2024&team={team_id}"
    r = requests.get(url, headers=HEADERS)
    data = r.json()["response"]

    goals_for = data["goals"]["for"]["total"]["home"] + data["goals"]["for"]["total"]["away"]
    goals_against = data["goals"]["against"]["total"]["home"] + data["goals"]["against"]["total"]["away"]
    games = data["fixtures"]["played"]["home"] + data["fixtures"]["played"]["away"]

    avg_for = goals_for / games if games > 0 else 0
    avg_against = goals_against / games if games > 0 else 0

    return avg_for, avg_against


# =============================
# MODELO POISSON + DIXON
# =============================

def poisson_prob(lmbda, k):
    return (lmbda**k * exp(-lmbda)) / factorial(k)


def dixon_coles_matrix(home_attack, home_defense,
                       away_attack, away_defense,
                       rho=0.1, max_goals=6):

    matrix = np.zeros((max_goals, max_goals))

    lambda_home = home_attack * away_defense
    lambda_away = away_attack * home_defense

    for i in range(max_goals):
        for j in range(max_goals):
            base = poisson_prob(lambda_home, i) * poisson_prob(lambda_away, j)

            # Ajuste Dixon-Coles
            if i == 0 and j == 0:
                adj = 1 - (lambda_home * lambda_away * rho)
            elif i == 0 and j == 1:
                adj = 1 + (lambda_home * rho)
            elif i == 1 and j == 0:
                adj = 1 + (lambda_away * rho)
            elif i == 1 and j == 1:
                adj = 1 - rho
            else:
                adj = 1

            matrix[i, j] = base * adj

    return matrix


# =============================
# INTERFAZ
# =============================

st.title("⚽ Football Quant Pro")

country = st.selectbox("Seleccionar País", get_countries())

if country:
    leagues = get_leagues(country)
    league_name = st.selectbox("Seleccionar Liga", list(leagues.keys()))

    if league_name:
        league_id = leagues[league_name]

        teams = get_teams(league_id)
        home_team = st.selectbox("Equipo Local", list(teams.keys()))
        away_team = st.selectbox("Equipo Visitante", list(teams.keys()))

        st.markdown("---")
        st.subheader("💰 Introduce las Cuotas Manualmente")

        col1, col2, col3 = st.columns(3)

        with col1:
            odd_home = st.number_input("Cuota Local", min_value=1.01, step=0.01)
            odd_over = st.number_input("Cuota Over 2.5", min_value=1.01, step=0.01)

        with col2:
            odd_draw = st.number_input("Cuota Empate", min_value=1.01, step=0.01)
            odd_under = st.number_input("Cuota Under 2.5", min_value=1.01, step=0.01)

        with col3:
            odd_away = st.number_input("Cuota Visitante", min_value=1.01, step=0.01)
            odd_btts = st.number_input("Cuota Ambos Marcan", min_value=1.01, step=0.01)

        st.markdown("---")

        if st.button("🔎 Analizar Partido"):

            with st.spinner("Calculando..."):

                h_attack, h_def = get_team_stats(league_id, teams[home_team])
                a_attack, a_def = get_team_stats(league_id, teams[away_team])

                matrix = dixon_coles_matrix(h_attack, h_def,
                                            a_attack, a_def)

                max_goals = 6

                home_win = np.sum(np.tril(matrix, -1))
                draw = np.sum(np.diag(matrix))
                away_win = np.sum(np.triu(matrix, 1))

                over25 = 0
                btts = 0

                for i in range(max_goals):
                    for j in range(max_goals):
                        if i + j > 2:
                            over25 += matrix[i, j]
                        if i > 0 and j > 0:
                            btts += matrix[i, j]

                under25 = 1 - over25

                st.subheader("📊 Probabilidades (%)")

                st.write(f"Local: {home_win*100:.2f}%")
                st.write(f"Empate: {draw*100:.2f}%")
                st.write(f"Visitante: {away_win*100:.2f}%")
                st.write(f"Over 2.5: {over25*100:.2f}%")
                st.write(f"Under 2.5: {under25*100:.2f}%")
                st.write(f"Ambos marcan: {btts*100:.2f}%")

                st.markdown("---")
                st.subheader("🎯 Cuotas Justas")

                def fair(p):
                    return 1 / p if p > 0 else 0

                st.write(f"Local: {fair(home_win):.2f}")
                st.write(f"Empate: {fair(draw):.2f}")
                st.write(f"Visitante: {fair(away_win):.2f}")
                st.write(f"Over 2.5: {fair(over25):.2f}")
                st.write(f"Under 2.5: {fair(under25):.2f}")
                st.write(f"Ambos marcan: {fair(btts):.2f}")

                st.markdown("---")
                st.subheader("💎 Detección de Value")

                def value(prob, odd):
                    return (prob * odd) - 1

                values = {
                    "Local": value(home_win, odd_home),
                    "Empate": value(draw, odd_draw),
                    "Visitante": value(away_win, odd_away),
                    "Over 2.5": value(over25, odd_over),
                    "Under 2.5": value(under25, odd_under),
                    "BTTS": value(btts, odd_btts)
                }

                for market, val in values.items():
                    if val > 0:
                        st.success(f"🔥 VALUE en {market} (+{val*100:.2f}%)")
                    else:
                        st.write(f"{market}: {val*100:.2f}%")
