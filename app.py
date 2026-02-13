import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

# =============================
# CONFIG API
# =============================
API_KEY = st.secrets["API_FOOTBALL_KEY"]
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# =============================
# FUNCIONES API
# =============================

def get_leagues():
    url = f"{BASE_URL}/leagues?season=2024"
    r = requests.get(url, headers=HEADERS).json()
    leagues = []
    for item in r["response"]:
        leagues.append({
            "name": item["league"]["name"],
            "id": item["league"]["id"]
        })
    return leagues


def get_teams(league_id):
    url = f"{BASE_URL}/teams?league={league_id}&season=2024"
    r = requests.get(url, headers=HEADERS).json()
    teams = []
    for item in r["response"]:
        teams.append(item["team"]["name"])
    return teams


def get_team_stats(team_name, league_id):
    url = f"{BASE_URL}/teams?search={team_name}"
    team_data = requests.get(url, headers=HEADERS).json()
    team_id = team_data["response"][0]["team"]["id"]

    stats_url = f"{BASE_URL}/teams/statistics?team={team_id}&league={league_id}&season=2024"
    stats = requests.get(stats_url, headers=HEADERS).json()

    goals_for = stats["response"]["goals"]["for"]["total"]["total"]
    goals_against = stats["response"]["goals"]["against"]["total"]["total"]
    matches = stats["response"]["fixtures"]["played"]["total"]

    return goals_for/matches, goals_against/matches


# =============================
# MODELO DIXON COLES
# =============================

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
            matrix[i][j] = base * tau(i, j, lambda_h, lambda_a, rho)

    return matrix


def calculate_value(prob, odds):
    return (prob * odds) - 1


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(layout="wide")
st.title("⚽ Football Quant PRO - Liga y Equipos")

st.write("Selecciona la liga y equipos para análisis automático.")

leagues = get_leagues()

league_names = [l["name"] for l in leagues]
selected_league = st.selectbox("Seleccionar Liga", league_names)

league_id = next(l["id"] for l in leagues if l["name"] == selected_league)

teams = get_teams(league_id)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Equipo Local", teams)

with col2:
    away_team = st.selectbox("Equipo Visitante", teams)

rho = st.slider("Rho Dixon-Coles", 0.0, 0.2, 0.05)

if st.button("Calcular Probabilidades"):

    home_attack, home_defense = get_team_stats(home_team, league_id)
    away_attack, away_defense = get_team_stats(away_team, league_id)

    # Promedio liga estimado simple
    league_avg = (home_attack + away_attack) / 2

    lambda_home = (home_attack * away_defense) / league_avg
    lambda_away = (away_attack * home_defense) / league_avg

    matrix = dixon_coles(lambda_home, lambda_away, rho)

    home_prob = np.sum(np.tril(matrix, -1))
    draw_prob = np.sum(np.diag(matrix))
    away_prob = np.sum(np.triu(matrix, 1))

    st.subheader("📊 Probabilidades Modelo")

    st.write({
        "Local": round(home_prob, 3),
        "Empate": round(draw_prob, 3),
        "Visitante": round(away_prob, 3)
    })

    st.subheader("💰 Evaluar Value")

    home_odds = st.number_input("Cuota Local", 1.0, 15.0, 2.0)

    value = calculate_value(home_prob, home_odds)

    st.write("Value:", round(value, 3))

    if value > 0:
        st.success("🔥 Apuesta con valor positivo")
    else:
        st.warning("No hay valor matemático")
