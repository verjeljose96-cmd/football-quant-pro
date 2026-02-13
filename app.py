import streamlit as st
import requests
import numpy as np

# =====================================
# CONFIG
# =====================================

st.set_page_config(page_title="Football Quant Pro ⚽", layout="wide")

API_KEY = "TU_API_KEY_AQUI"
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

st.title("⚽ Football Quant Pro")

# =====================================
# FUNCIONES SEGURAS
# =====================================

def safe_request(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        return response.json()
    except:
        return None


def get_countries():
    data = safe_request(f"{BASE_URL}/countries")
    if data and "response" in data:
        return sorted([c["name"] for c in data["response"]])
    return []


def get_leagues(country):
    data = safe_request(f"{BASE_URL}/leagues?country={country}&season=2024")
    leagues = {}

    if data and "response" in data:
        for league in data["response"]:
            leagues[league["league"]["name"]] = league["league"]["id"]

    return leagues


def get_teams(league_id):
    data = safe_request(f"{BASE_URL}/teams?league={league_id}&season=2024")
    if data and "response" in data:
        return [team["team"]["name"] for team in data["response"]]
    return []


def get_team_stats(team_name, league_id):

    data = safe_request(f"{BASE_URL}/teams?league={league_id}&season=2024")

    if not data or "response" not in data:
        return 1.2, 1.2

    team_id = None
    for team in data["response"]:
        if team["team"]["name"] == team_name:
            team_id = team["team"]["id"]
            break

    if team_id is None:
        return 1.2, 1.2

    stats_data = safe_request(
        f"{BASE_URL}/teams/statistics?league={league_id}&season=2024&team={team_id}"
    )

    if not stats_data or "response" not in stats_data:
        return 1.2, 1.2

    stats = stats_data["response"]

    goals_for = stats["goals"]["for"]["total"]["home"]
    goals_against = stats["goals"]["against"]["total"]["home"]
    matches = stats["fixtures"]["played"]["home"]

    if matches == 0:
        matches = 1

    return goals_for / matches, goals_against / matches


# =====================================
# MODELO
# =====================================

def montecarlo(lambda_h, lambda_a, sims=10000):
    home_goals = np.random.poisson(lambda_h, sims)
    away_goals = np.random.poisson(lambda_a, sims)

    return (
        np.mean(home_goals > away_goals),
        np.mean(home_goals == away_goals),
        np.mean(home_goals < away_goals),
        np.mean((home_goals + away_goals) > 2.5),
        np.mean((home_goals > 0) & (away_goals > 0))
    )


def fair_odds(prob):
    return round(1 / prob, 2) if prob > 0 else 0


def value(prob, odds):
    return round((prob * odds) - 1, 3)


def kelly(prob, odds):
    k = ((prob * (odds - 1)) - (1 - prob)) / (odds - 1)
    return round(max(k, 0), 3)


# =====================================
# INTERFAZ
# =====================================

countries = get_countries()

if not countries:
    st.error("Error con API o límite alcanzado.")
    st.stop()

selected_country = st.selectbox("🌎 País", countries)

leagues = get_leagues(selected_country)

if not leagues:
    st.warning("Este país no tiene ligas disponibles 2024.")
    st.stop()

selected_league = st.selectbox("🏆 Liga", list(leagues.keys()))
league_id = leagues.get(selected_league)

teams = get_teams(league_id)

if not teams:
    st.warning("No se pudieron cargar equipos.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Local", teams)

with col2:
    away_team = st.selectbox("✈ Visitante", teams)


# =====================================
# SESSION
# =====================================

if "resultado" not in st.session_state:
    st.session_state.resultado = None


if st.button("🔎 Analizar"):

    if home_team == away_team:
        st.warning("No puedes elegir el mismo equipo")
    else:
        home_attack, home_defense = get_team_stats(home_team, league_id)
        away_attack, away_defense = get_team_stats(away_team, league_id)

        league_avg = (home_attack + away_attack) / 2

        lambda_h = (home_attack * away_defense) / league_avg
        lambda_a = (away_attack * home_defense) / league_avg

        st.session_state.resultado = montecarlo(lambda_h, lambda_a)


# =====================================
# RESULTADOS
# =====================================

if st.session_state.resultado:

    mc_home, mc_draw, mc_away, mc_over25, mc_btts = st.session_state.resultado

    st.subheader("📊 Probabilidades y Cuota Justa")

    markets = {
        "Local": mc_home,
        "Empate": mc_draw,
        "Visitante": mc_away,
        "Over 2.5": mc_over25,
        "BTTS": mc_btts
    }

    for name, prob in markets.items():
        st.write(
            f"{name} → "
            f"{round(prob*100,2)}% | "
            f"Cuota Justa: {fair_odds(prob)}"
        )

    st.subheader("💰 Comparar con Casa")

    odds_home = st.number_input("Cuota Casa - Local", 1.0, 20.0, 2.0)

    st.write(
        f"Value: {value(mc_home, odds_home)} | "
        f"Kelly: {kelly(mc_home, odds_home)}"
        )
