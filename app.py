import streamlit as st
import numpy as np
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


def dixon_coles(lambda_h, lambda_a, rho=0.05, max_goals=6):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            base = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            matrix[i][j] = base * tau(i, j, lambda_h, lambda_a, rho)
    return matrix


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
# INTERFAZ
# =============================

st.set_page_config(layout="wide")
st.title("⚽ Football Quant PRO - Sistema Profesional")

leagues = get_leagues()
league_names = [l["name"] for l in leagues]
selected_league = st.selectbox("Liga", league_names)
league_id = next(l["id"] for l in leagues if l["name"] == selected_league)

teams = get_teams(league_id)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Equipo Local", teams)

with col2:
    away_team = st.selectbox("Equipo Visitante", teams)

rho = st.slider("Rho Dixon-Coles", 0.0, 0.2, 0.05)

if st.button("Analizar Partido"):

    home_attack, home_defense = get_team_stats(home_team, league_id)
    away_attack, away_defense = get_team_stats(away_team, league_id)

    league_avg = (home_attack + away_attack) / 2

    lambda_h = (home_attack * away_defense) / league_avg
    lambda_a = (away_attack * home_defense) / league_avg

    matrix = dixon_coles(lambda_h, lambda_a, rho)

    home_prob = np.sum(np.tril(matrix, -1))
    draw_prob = np.sum(np.diag(matrix))
    away_prob = np.sum(np.triu(matrix, 1))

    mc = montecarlo(lambda_h, lambda_a)
    mc_home, mc_draw, mc_away, mc_over25, mc_btts = mc

    st.subheader("📊 Probabilidades Modelo (Monte Carlo)")

    st.write({
        "Local": round(mc_home,3),
        "Empate": round(mc_draw,3),
        "Visitante": round(mc_away,3),
        "Over 2.5": round(mc_over25,3),
        "BTTS": round(mc_btts,3)
    })

    st.subheader("💰 Evaluación de Valor")

    odds_home = st.number_input("Cuota Local", 1.0, 20.0, 2.0)
    odds_over = st.number_input("Cuota Over 2.5", 1.0, 20.0, 1.9)

    value_home = value(mc_home, odds_home)
    value_over = value(mc_over25, odds_over)

    st.write({
        "Value Local": round(value_home,3),
        "Kelly Local": round(kelly(mc_home, odds_home),3),
        "Value Over 2.5": round(value_over,3),
        "Kelly Over 2.5": round(kelly(mc_over25, odds_over),3)
    })

    if value_home > 0:
        st.success("🔥 Local tiene Value")
    if value_over > 0:
        st.success("🔥 Over 2.5 tiene Value")
