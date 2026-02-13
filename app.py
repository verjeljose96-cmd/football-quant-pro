import streamlit as st
import requests
import numpy as np
import pandas as pd
from math import factorial
from scipy.optimize import minimize

st.set_page_config(page_title="Football Quant Pro", layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# =====================================================
# API FUNCTIONS
# =====================================================

@st.cache_data(ttl=3600)
def get_leagues(country):
    url = f"{BASE_URL}/leagues?country={country}&season=2024"
    r = requests.get(url, headers=HEADERS)
    data = r.json().get("response", [])
    leagues = {}
    for l in data:
        leagues[l["league"]["name"]] = l["league"]["id"]
    return leagues


@st.cache_data(ttl=3600)
def get_matches(league_id):
    url = f"{BASE_URL}/fixtures?league={league_id}&season=2024&status=FT"
    r = requests.get(url, headers=HEADERS)
    data = r.json().get("response", [])

    rows = []
    for m in data:
        if m["goals"]["home"] is not None:
            rows.append({
                "home": m["teams"]["home"]["name"],
                "away": m["teams"]["away"]["name"],
                "home_goals": m["goals"]["home"],
                "away_goals": m["goals"]["away"]
            })
    return pd.DataFrame(rows)

# =====================================================
# DIXON-COLES MODEL
# =====================================================

def poisson(lmbda, k):
    return (lmbda ** k * np.exp(-lmbda)) / factorial(k)


def dc_adjustment(x, y, l1, l2, rho):
    if x == 0 and y == 0:
        return 1 - (l1 * l2 * rho)
    elif x == 0 and y == 1:
        return 1 + (l1 * rho)
    elif x == 1 and y == 0:
        return 1 + (l2 * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1


def log_likelihood(params, df, teams):
    n = len(teams)
    attack = params[:n]
    defense = params[n:2*n]
    gamma = params[-2]
    rho = params[-1]

    ll = 0

    for _, row in df.iterrows():
        i = teams.index(row["home"])
        j = teams.index(row["away"])

        l1 = np.exp(attack[i] - defense[j] + gamma)
        l2 = np.exp(attack[j] - defense[i])

        p = poisson(l1, row["home_goals"]) * poisson(l2, row["away_goals"])
        p *= dc_adjustment(row["home_goals"], row["away_goals"], l1, l2, rho)

        ll += np.log(p + 1e-10)

    return -ll

# =====================================================
# UI
# =====================================================

st.title("⚽ Football Quant Pro - Modelo Profesional")

country = st.text_input("País (Ej: Colombia)")

if country:

    leagues = get_leagues(country)
    league_name = st.selectbox("Liga", list(leagues.keys()))

    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if st.button("Entrenar Modelo"):

        league_id = leagues[league_name]
        df = get_matches(league_id)

        st.write(f"Partidos descargados: {len(df)}")

        if df.empty:
            st.error("No hay partidos finalizados.")
            st.stop()

        teams = sorted(list(set(df["home"]).union(set(df["away"]))))
        n = len(teams)

        init_params = np.concatenate([
            np.zeros(n),
            np.zeros(n),
            [0.1],
            [0.05]
        ])

        with st.spinner("Entrenando modelo..."):
            result = minimize(
                log_likelihood,
                init_params,
                args=(df, teams),
                method="L-BFGS-B"
            )

        params = result.x

        st.session_state.attack = params[:n]
        st.session_state.defense = params[n:2*n]
        st.session_state.gamma = params[-2]
        st.session_state.rho = params[-1]
        st.session_state.teams = teams
        st.session_state.model_trained = True

        st.success("Modelo entrenado")

    # =====================================================
    # PREDICTION SECTION
    # =====================================================

    if st.session_state.model_trained:

        teams = st.session_state.teams
        attack = st.session_state.attack
        defense = st.session_state.defense
        gamma = st.session_state.gamma
        rho = st.session_state.rho

        st.markdown("---")
        home_team = st.selectbox("Equipo Local", teams)
        away_team = st.selectbox("Equipo Visitante", teams)

        st.markdown("## 💰 Cuotas Manuales")

        col1, col2, col3 = st.columns(3)

        with col1:
            odd_home = st.number_input("Cuota Local", min_value=1.01)
            odd_over = st.number_input("Cuota Over 2.5", min_value=1.01)

        with col2:
            odd_draw = st.number_input("Cuota Empate", min_value=1.01)
            odd_under = st.number_input("Cuota Under 2.5", min_value=1.01)

        with col3:
            odd_away = st.number_input("Cuota Visitante", min_value=1.01)
            odd_btts = st.number_input("Cuota BTTS", min_value=1.01)

        if st.button("Calcular Probabilidades"):

            i = teams.index(home_team)
            j = teams.index(away_team)

            l1 = np.exp(attack[i] - defense[j] + gamma)
            l2 = np.exp(attack[j] - defense[i])

            max_goals = 6
            matrix = np.zeros((max_goals, max_goals))

            for x in range(max_goals):
                for y in range(max_goals):
                    p = poisson(l1, x) * poisson(l2, y)
                    p *= dc_adjustment(x, y, l1, l2, rho)
                    matrix[x, y] = p

            home_win = np.sum(np.tril(matrix, -1))
            draw = np.sum(np.diag(matrix))
            away_win = np.sum(np.triu(matrix, 1))

            over25 = 0
            btts = 0

            for x in range(max_goals):
                for y in range(max_goals):
                    if x + y > 2:
                        over25 += matrix[x, y]
                    if x > 0 and y > 0:
                        btts += matrix[x, y]

            under25 = 1 - over25

            st.markdown("## 📊 Probabilidades (%)")
            st.write(f"Local: {home_win*100:.2f}%")
            st.write(f"Empate: {draw*100:.2f}%")
            st.write(f"Visitante: {away_win*100:.2f}%")
            st.write(f"Over 2.5: {over25*100:.2f}%")
            st.write(f"Under 2.5: {under25*100:.2f}%")
            st.write(f"Ambos marcan: {btts*100:.2f}%")

            st.markdown("## 🎯 Cuotas Justas")

            def fair(p):
                return round(1/p, 2) if p > 0 else 0

            st.write(f"Local: {fair(home_win)}")
            st.write(f"Empate: {fair(draw)}")
            st.write(f"Visitante: {fair(away_win)}")
            st.write(f"Over 2.5: {fair(over25)}")
            st.write(f"Under 2.5: {fair(under25)}")
            st.write(f"BTTS: {fair(btts)}")

            st.markdown("## 💎 Detección de Value")

            def value(prob, odd):
                return (prob * odd) - 1

            values = {
                "Local": value(home_win, odd_home),
                "Empate": value(draw, odd_draw),
                "Visitante": value(away_win, odd_away),
                "Over 2.5": value(over25, odd_over),
                "Under 2.5": value(under25, odd_under),
                "BTTS": value(btts, odd_btts),
            }

            for market, val in values.items():
                if val > 0:
                    st.success(f"🔥 VALUE en {market} (+{val*100:.2f}%)")
                else:
                    st.write(f"{market}: {val*100:.2f}%")
