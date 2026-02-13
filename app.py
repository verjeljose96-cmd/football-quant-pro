import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import math

st.set_page_config(page_title="Football Quant Pro", layout="wide")

# ==============================
# CONFIG API
# ==============================
API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

season = datetime.now().year


# ==============================
# API CALL OPTIMIZADA
# ==============================
@st.cache_data(ttl=600)
def api_get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return None, response.status_code

    return response.json(), 200


# ==============================
# OBTENER PAISES
# ==============================
@st.cache_data(ttl=3600)
def get_countries():
    data, status = api_get("countries")
    if status != 200:
        return []
    return sorted([c["name"] for c in data["response"]])


# ==============================
# OBTENER LIGAS POR PAIS
# ==============================
@st.cache_data(ttl=3600)
def get_leagues(country):
    data, status = api_get("leagues", {"country": country, "season": season})
    if status != 200:
        return {}

    leagues = {}
    for l in data["response"]:
        leagues[l["league"]["name"]] = l["league"]["id"]

    return leagues


# ==============================
# OBTENER EQUIPOS
# ==============================
@st.cache_data(ttl=3600)
def get_teams(league_id):
    data, status = api_get("teams", {"league": league_id, "season": season})
    if status != 200:
        return {}

    teams = {}
    for t in data["response"]:
        teams[t["team"]["name"]] = t["team"]["id"]

    return teams


# ==============================
# OBTENER ESTADISTICAS
# ==============================
@st.cache_data(ttl=600)
def get_team_stats(team_id, league_id):
    data, status = api_get(
        "teams/statistics",
        {"league": league_id, "season": season, "team": team_id}
    )

    if status != 200:
        return None

    return data["response"]


# ==============================
# MODELO POISSON
# ==============================
def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)


# ==============================
# DIXON COLES
# ==============================
def dixon_coles_adjustment(x, y, lambda_home, lambda_away, rho=0.1):
    if x == 0 and y == 0:
        return 1 - (lambda_home * lambda_away * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_home * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_away * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1


# ==============================
# UI
# ==============================

st.title("⚽ Football Quant Pro - Modelo Profesional")

countries = get_countries()
selected_country = st.selectbox("Selecciona País", countries)

if selected_country:

    leagues = get_leagues(selected_country)
    selected_league = st.selectbox("Selecciona Liga", list(leagues.keys()))

    if selected_league:

        league_id = leagues.get(selected_league)

        teams = get_teams(league_id)
        col1, col2 = st.columns(2)

        with col1:
            home_team = st.selectbox("Equipo Local", list(teams.keys()))

        with col2:
            away_team = st.selectbox("Equipo Visitante", list(teams.keys()))

        analyze = st.button("🔎 Analizar Partido")

        if analyze:

            home_id = teams.get(home_team)
            away_id = teams.get(away_team)

            home_stats = get_team_stats(home_id, league_id)
            away_stats = get_team_stats(away_id, league_id)

            if not home_stats or not away_stats:
                st.error("Error obteniendo estadísticas. Revisa límite API.")
                st.stop()

            # ======================
            # EXTRACCION GOLES
            # ======================
            home_goals_for = home_stats["goals"]["for"]["total"]["home"]
            home_goals_against = home_stats["goals"]["against"]["total"]["home"]

            away_goals_for = away_stats["goals"]["for"]["total"]["away"]
            away_goals_against = away_stats["goals"]["against"]["total"]["away"]

            home_games = home_stats["fixtures"]["played"]["home"]
            away_games = away_stats["fixtures"]["played"]["away"]

            lambda_home = home_goals_for / max(home_games, 1)
            lambda_away = away_goals_for / max(away_games, 1)

            # ======================
            # CALCULO MATRIZ
            # ======================
            max_goals = 5
            matrix = np.zeros((max_goals, max_goals))

            for i in range(max_goals):
                for j in range(max_goals):
                    p = poisson_prob(lambda_home, i) * poisson_prob(lambda_away, j)
                    p *= dixon_coles_adjustment(i, j, lambda_home, lambda_away)
                    matrix[i, j] = p

            home_win = np.sum(np.tril(matrix, -1))
            draw = np.sum(np.diag(matrix))
            away_win = np.sum(np.triu(matrix, 1))

            home_pct = round(home_win * 100, 2)
            draw_pct = round(draw * 100, 2)
            away_pct = round(away_win * 100, 2)

            # ======================
            # RESULTADOS
            # ======================
            st.subheader("📊 Probabilidades Modelo (%)")

            col1, col2, col3 = st.columns(3)
            col1.metric("Local", f"{home_pct}%")
            col2.metric("Empate", f"{draw_pct}%")
            col3.metric("Visitante", f"{away_pct}%")

            st.subheader("💰 Cuota Justa")

            if home_win > 0:
                st.write("Local:", round(1 / home_win, 2))
            if draw > 0:
                st.write("Empate:", round(1 / draw, 2))
            if away_win > 0:
                st.write("Visitante:", round(1 / away_win, 2))
