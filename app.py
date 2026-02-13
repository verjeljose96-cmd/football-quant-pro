import streamlit as st
import requests
import numpy as np

# =====================================
# CONFIGURACIÓN
# =====================================

st.set_page_config(page_title="Football Quant Pro ⚽", layout="wide")

API_KEY = "TU_API_KEY_AQUI"
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}

st.title("⚽ Football Quant Pro - Modelo Cuantitativo")

# =====================================
# FUNCIONES API
# =====================================

def get_countries():
    url = f"{BASE_URL}/countries"
    response = requests.get(url, headers=headers)
    data = response.json()
    return sorted([c["name"] for c in data["response"]])


def get_leagues(country):
    url = f"{BASE_URL}/leagues?country={country}&season=2024"
    response = requests.get(url, headers=headers)
    data = response.json()

    leagues = {}
    for league in data["response"]:
        leagues[league["league"]["name"]] = league["league"]["id"]

    return leagues


def get_teams(league_id):
    url = f"{BASE_URL}/teams?league={league_id}&season=2024"
    response = requests.get(url, headers=headers)
    data = response.json()

    return [team["team"]["name"] for team in data["response"]]


def get_team_stats(team_name, league_id):

    url = f"{BASE_URL}/teams?league={league_id}&season=2024"
    response = requests.get(url, headers=headers)
    data = response.json()

    team_id = None

    for team in data["response"]:
        if team["team"]["name"] == team_name:
            team_id = team["team"]["id"]
            break

    if team_id is None:
        return 1.2, 1.2

    url_stats = f"{BASE_URL}/teams/statistics?league={league_id}&season=2024&team={team_id}"
    response_stats = requests.get(url_stats, headers=headers)
    stats = response_stats.json()["response"]

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

    home_win = np.mean(home_goals > away_goals)
    draw = np.mean(home_goals == away_goals)
    away_win = np.mean(home_goals < away_goals)
    over25 = np.mean((home_goals + away_goals) > 2.5)
    btts = np.mean((home_goals > 0) & (away_goals > 0))

    return home_win, draw, away_win, over25, btts


def value(prob, odds):
    return (prob * odds) - 1


def kelly(prob, odds):
    k = ((prob * (odds - 1)) - (1 - prob)) / (odds - 1)
    return max(k, 0)


def fair_odds(prob):
    if prob == 0:
        return 0
    return 1 / prob


# =====================================
# SELECTORES
# =====================================

countries = get_countries()
selected_country = st.selectbox("🌎 País", countries)

leagues = get_leagues(selected_country)
selected_league = st.selectbox("🏆 Liga", list(leagues.keys()))

league_id = leagues[selected_league]

teams = get_teams(league_id)

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


# =====================================
# ANALIZAR
# =====================================

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

    st.subheader("📊 Probabilidades (%) y Cuota Justa")

    results = {
        "Local": mc_home,
        "Empate": mc_draw,
        "Visitante": mc_away,
        "Over 2.5": mc_over25,
        "BTTS": mc_btts
    }

    for name, prob in results.items():
        st.write(
            f"{name}: {round(prob*100,2)}%  |  "
            f"Cuota Justa: {round(fair_odds(prob),2)}"
        )

    st.subheader("💰 Comparar contra Casa")

    odds_home = st.number_input("Cuota Casa - Local", 1.0, 20.0, 2.0)

    st.write(
        f"Value Local: {round(value(mc_home, odds_home),3)} | "
        f"Kelly: {round(kelly(mc_home, odds_home),3)}"
    )
