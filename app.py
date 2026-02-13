import streamlit as st
import requests
import numpy as np
import math
from datetime import datetime

st.set_page_config(page_title="Football Quant Pro", layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}
season = datetime.now().year

# ======================
# API
# ======================
@st.cache_data(ttl=600)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_data(ttl=3600)
def get_countries():
    data = api_get("countries")
    return sorted([c["name"] for c in data["response"]]) if data else []

@st.cache_data(ttl=3600)
def get_leagues(country):
    data = api_get("leagues", {"country": country, "season": season})
    return {l["league"]["name"]: l["league"]["id"] for l in data["response"]} if data else {}

@st.cache_data(ttl=3600)
def get_teams(league_id):
    data = api_get("teams", {"league": league_id, "season": season})
    return {t["team"]["name"]: t["team"]["id"] for t in data["response"]} if data else {}

@st.cache_data(ttl=600)
def get_stats(team_id, league_id):
    data = api_get("teams/statistics",
                   {"league": league_id, "season": season, "team": team_id})
    return data["response"] if data else None

@st.cache_data(ttl=600)
def get_odds(home_id, away_id):

    fixtures = api_get("fixtures", {
        "home": home_id,
        "away": away_id,
        "status": "NS"
    })

    if not fixtures or fixtures["results"] == 0:
        return None

    fixture_id = fixtures["response"][0]["fixture"]["id"]

    odds_data = api_get("odds", {"fixture": fixture_id})

    if not odds_data or odds_data["results"] == 0:
        return None

    odds = {}

    for bookmaker in odds_data["response"][0]["bookmakers"]:
        for bet in bookmaker["bets"]:

            if bet["name"] == "Match Winner":
                for v in bet["values"]:
                    odds[v["value"]] = float(v["odd"])

            if bet["name"] == "Goals Over/Under":
                for v in bet["values"]:
                    odds[v["value"]] = float(v["odd"])

            if bet["name"] == "Both Teams Score":
                for v in bet["values"]:
                    if v["value"] == "Yes":
                        odds["BTTS Yes"] = float(v["odd"])

    return odds if odds else None

# ======================
# MODEL
# ======================
def poisson(lmbda, k):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def value(prob, odd):
    return (prob * odd) - 1

# ======================
# UI
# ======================
st.title("⚽ Football Quant Pro - Hybrid Value Engine")

countries = get_countries()
country = st.selectbox("País", countries)

leagues = get_leagues(country)
league = st.selectbox("Liga", list(leagues.keys()))

league_id = leagues.get(league)
teams = get_teams(league_id)

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Local", list(teams.keys()))
with col2:
    away = st.selectbox("Visitante", list(teams.keys()))

analyze = st.button("🔎 Analizar Partido")

# ======================
# ANALYSIS
# ======================
if analyze:

    home_id = teams[home]
    away_id = teams[away]

    home_stats = get_stats(home_id, league_id)
    away_stats = get_stats(away_id, league_id)

    if not home_stats or not away_stats:
        st.error("No se pudieron obtener estadísticas.")
        st.stop()

    lambda_home = home_stats["goals"]["for"]["total"]["home"] / max(home_stats["fixtures"]["played"]["home"],1)
    lambda_away = away_stats["goals"]["for"]["total"]["away"] / max(away_stats["fixtures"]["played"]["away"],1)

    max_goals = 6
    matrix = np.zeros((max_goals,max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i,j] = poisson(lambda_home,i) * poisson(lambda_away,j)

    home_win = np.sum(np.tril(matrix,-1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix,1))

    over25 = 0
    btts = 0

    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > 2:
                over25 += matrix[i,j]
            if i > 0 and j > 0:
                btts += matrix[i,j]

    under25 = 1 - over25

    # ======================
    # OBTENER CUOTAS
    # ======================
    odds = get_odds(home_id, away_id)

    if odds:
        st.success("✅ Cuotas automáticas encontradas")
    else:
        st.warning("⚠ No se encontraron cuotas automáticas. Ingrese manualmente.")
        odds = {}

    # ======================
    # INPUT MANUAL SI NECESARIO
    # ======================
    col1, col2, col3 = st.columns(3)

    with col1:
        odds["Home"] = odds.get("Home") or st.number_input("Cuota Local", 1.01, 20.0, 2.0)
        odds["Over 2.5"] = odds.get("Over 2.5") or st.number_input("Cuota Over 2.5", 1.01, 20.0, 1.9)

    with col2:
        odds["Draw"] = odds.get("Draw") or st.number_input("Cuota Empate", 1.01, 20.0, 3.0)
        odds["Under 2.5"] = odds.get("Under 2.5") or st.number_input("Cuota Under 2.5", 1.01, 20.0, 1.9)

    with col3:
        odds["Away"] = odds.get("Away") or st.number_input("Cuota Visitante", 1.01, 20.0, 3.0)
        odds["BTTS Yes"] = odds.get("BTTS Yes") or st.number_input("Cuota BTTS Sí", 1.01, 20.0, 1.8)

    # ======================
    # VALUE ENGINE
    # ======================
    st.subheader("📊 Value Detection Engine")

    markets = {
        "Home": (home_win, odds["Home"]),
        "Draw": (draw, odds["Draw"]),
        "Away": (away_win, odds["Away"]),
        "Over 2.5": (over25, odds["Over 2.5"]),
        "Under 2.5": (under25, odds["Under 2.5"]),
        "BTTS Yes": (btts, odds["BTTS Yes"])
    }

    best_market = None
    best_value = -999

    for name,(prob,odd) in markets.items():

        val = value(prob,odd)
        pct = round(prob*100,2)
        fair = round(1/prob,2) if prob>0 else 0

        if val > best_value:
            best_value = val
            best_market = name

        emoji = "🟢" if val>0 else "🔴"

        st.write(
            f"{emoji} **{name}** → "
            f"{pct}% | Casa: {odd} | Justa: {fair} | "
            f"Value: {round(val,3)}"
        )

    if best_value > 0:
        st.success(f"⭐ MEJOR VALUE BET: {best_market} (EV={round(best_value,3)})")
    else:
        st.warning("No hay Value Bet positivo en este partido.")
