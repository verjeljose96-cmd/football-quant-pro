import streamlit as st
import numpy as np
import requests
from scipy.stats import poisson

# =============================
# CONFIG
# =============================

st.set_page_config(layout="wide")
st.title("⚽ Football Quant PRO")

API_KEY = st.secrets["API_FOOTBALL_KEY"]
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# =============================
# API FUNCTIONS (CACHED)
# =============================

@st.cache_data(ttl=3600)
def get_countries():
    url = f"{BASE_URL}/countries"
    r = requests.get(url, headers=HEADERS).json()
    countries = [c["name"] for c in r["response"] if c["name"]]
    return sorted(list(set(countries)))


@st.cache_data(ttl=3600)
def get_leagues_by_country(country):
    url = f"{BASE_URL}/leagues?country={country}&season=2024"
    r = requests.get(url, headers=HEADERS).json()
    leagues = []
    for item in r["response"]:
        leagues.append({
            "name": item["league"]["name"],
            "id": item["league"]["id"]
        })
    return leagues


@st.cache_data(ttl=3600)
def get_teams(league_id):
    url = f"{BASE_URL}/teams?league={league_id}&season=2024"
    r = requests.get(url, headers=HEADERS).json()
    return [item["team"]["name"] for item in r["response"]]


def get_team_stats(team_name, league_id):
    url = f"{BASE_URL}/teams?search={team_name}"
    team_data = requests.get(url, headers=HEADERS).json()
    team_id = team_data["response"][0]["team"]["id"]

    stats_url = f"{BASE_URL}/teams/statistics?team={team_id}&league={league_id}&season=2024"
    stats = requests.get(stats_url, headers=HEADERS).json()

    gf = stats["response"]["goals"]["for"]["total"]["total"]
    ga = stats["response"]["goals"]["against"]["total"]["total"]
    played = stats["response"]["fixtures"]["played"]["total"]

    if played == 0:
        return 1.2, 1.2  # fallback

    return gf/played, ga/played


# =============================
# MODELOS
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


def montecarlo(lambda_h, lambda_a, sims=20000):
    home = np.random.poisson(lambda_h, sims)
    away = np.random.poisson(lambda_a, sims)

    home_win = np.mean(home > away)
    draw = np.mean(home == away)
    away_win = np.mean(home < away)
    over25 = np.mean((home + away) > 2.5)
    btts = np.mean((home > 0) & (away > 0))

    return home_win, draw, away_win, over25, btts


def value(prob, odds):
    return (prob * odds) - 1


def kelly(prob, odds):
    b = odds - 1
    return max((prob*b - (1-prob))/b, 0)


# =============================
# UI SELECTION
# =============================

# 1️⃣ País
countries = get_countries()
selected_country = st.selectbox("🌍 País", countries)

# 2️⃣ Liga
leagues = get_leagues_by_country(selected_country)

if leagues:

    league_names = [l["name"] for l in leagues]
    selected_league = st.selectbox("🏆 Liga", league_names)
    league_id = next(l["id"] for l in leagues if l["name"] == selected_league)

    # 3️⃣ Equipos
    teams = get_teams(league_id)

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Equipo Local", teams)

    with col2:
        away_team = st.selectbox("Equipo Visitante", teams)

    if home_team == away_team:
        st.error("⚠ No puedes seleccionar el mismo equipo.")
        st.stop()

    rho = st.slider("Rho Dixon-Coles", 0.0, 0.2, 0.05)

    # =============================
    # SESSION STATE
    # =============================

    if "resultado" not in st.session_state:
        st.session_state.resultado = None

    if st.button("🔎 Analizar Partido"):

        home_attack, home_defense = get_team_stats(home_team, league_id)
        away_attack, away_defense = get_team_stats(away_team, league_id)

        league_avg = (home_attack + away_attack) / 2

        lambda_h = (home_attack * away_defense) / league_avg
        lambda_a = (away_attack * home_defense) / league_avg

        st.session_state.resultado = montecarlo(lambda_h, lambda_a)

    # =============================
    # MOSTRAR RESULTADOS
    # =============================

    if st.session_state.resultado:

        mc_home, mc_draw, mc_away, mc_over25, mc_btts = st.session_state.resultado

        st.subheader("📊 Probabilidades (Monte Carlo 20K)")

        st.write({
            "Local": round(mc_home,3),
            "Empate": round(mc_draw,3),
            "Visitante": round(mc_away,3),
            "Over 2.5": round(mc_over25,3),
            "BTTS": round(mc_btts,3)
        })

        st.subheader("💰 Evaluación de Valor")

        odds_home = st.number_input("Cuota Local", 1.0, 20.0, 2.0, key="odds1")
        odds_over = st.number_input("Cuota Over 2.5", 1.0, 20.0, 1.9, key="odds2")

        st.write({
            "Value Local": round(value(mc_home, odds_home),3),
            "Kelly Local": round(kelly(mc_home, odds_home),3),
            "Value Over 2.5": round(value(mc_over25, odds_over),3),
            "Kelly Over 2.5": round(kelly(mc_over25, odds_over),3)
        })

        if value(mc_home, odds_home) > 0:
            st.success("🔥 Local tiene Value")

        if value(mc_over25, odds_over) > 0:
            st.success("🔥 Over 2.5 tiene Value")

else:
    st.warning("No hay ligas disponibles para este país.")
