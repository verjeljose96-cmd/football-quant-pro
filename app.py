import streamlit as st
import requests
import numpy as np
import pandas as pd
from math import factorial
from scipy.optimize import minimize

st.set_page_config(layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# -----------------------------------
# FUNCIONES API
# -----------------------------------

@st.cache_data(ttl=3600)
def get_leagues(country):
    url = f"{BASE_URL}/leagues?country={country}&season=2024"
    r = requests.get(url, headers=HEADERS)
    leagues = {}
    for l in r.json()["response"]:
        leagues[l["league"]["name"]] = l["league"]["id"]
    return leagues


@st.cache_data(ttl=3600)
def get_matches(league_id):
    url = f"{BASE_URL}/fixtures?league={league_id}&season=2024&status=FT"
    r = requests.get(url, headers=HEADERS)
    data = r.json()["response"]

    rows = []
    for m in data:
        if m["goals"]["home"] is not None and m["goals"]["away"] is not None:
            rows.append({
                "home": m["teams"]["home"]["name"],
                "away": m["teams"]["away"]["name"],
                "home_goals": m["goals"]["home"],
                "away_goals": m["goals"]["away"]
            })

    return pd.DataFrame(rows)

# -----------------------------------
# MODELO DIXON COLES
# -----------------------------------

def poisson(lmbda, k):
    return (lmbda**k * np.exp(-lmbda)) / factorial(k)


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


# -----------------------------------
# UI
# -----------------------------------

st.title("Modelo Profesional Dixon-Coles (MLE Persistente)")

country = st.text_input("País (ej: Colombia)")

if country:

    leagues = get_leagues(country)
    league_name = st.selectbox("Liga", list(leagues.keys()))

    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if st.button("Entrenar Modelo"):

        league_id = leagues[league_name]
        df = get_matches(league_id)

        teams = sorted(list(set(df["home"]).union(set(df["away"]))))
        n = len(teams)

        init_params = np.concatenate([
            np.zeros(n),
            np.zeros(n),
            [0],
            [0.1]
        ])

        with st.spinner("Entrenando modelo..."):

            result = minimize(
                log_likelihood,
                init_params,
                args=(df, teams),
                method="L-BFGS-B"
            )

        st.session_state.attack = result.x[:n]
        st.session_state.defense = result.x[n:2*n]
        st.session_state.gamma = result.x[-2]
        st.session_state.rho = result.x[-1]
        st.session_state.teams = teams
        st.session_state.model_trained = True

        st.success("Modelo entrenado y guardado en sesión")

    # -----------------------------------
    # SI EL MODELO YA ESTÁ ENTRENADO
    # -----------------------------------

    if st.session_state.model_trained:

        teams = st.session_state.teams
        attack = st.session_state.attack
        defense = st.session_state.defense
        gamma = st.session_state.gamma
        rho = st.session_state.rho

        home_team = st.selectbox("Equipo Local", teams)
        away_team = st.selectbox("Equipo Visitante", teams)

        if st.button("Calcular Probabilidades"):

            i = teams.index(home_team)
            j = teams.index(away_team)

            l1 = np.exp(attack[i] - defense[j] + gamma)
            l2 = np.exp(attack[j] - defense[i])

            max_goals = 6
            matrix = np.zeros((max_goals, max_goals))

            for x in range(max_goals):
                for y in range(max_goals):
                    p = poisson(l1, x)*poisson(l2, y)
                    p *= dc_adjustment(x, y, l1, l2, rho)
                    matrix[x,y] = p

            home_win = np.sum(np.tril(matrix, -1))
            draw = np.sum(np.diag(matrix))
            away_win = np.sum(np.triu(matrix, 1))

            st.subheader("Probabilidades")
            st.write(f"Local: {home_win*100:.2f}%")
            st.write(f"Empate: {draw*100:.2f}%")
            st.write(f"Visitante: {away_win*100:.2f}%")
