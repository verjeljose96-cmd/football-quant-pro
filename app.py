import streamlit as st
import requests
import numpy as np
import pandas as pd
from math import exp, factorial
from scipy.optimize import minimize

st.set_page_config(layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# -------------------------------
# API CALLS
# -------------------------------

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
        rows.append({
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "home_goals": m["goals"]["home"],
            "away_goals": m["goals"]["away"]
        })
    return pd.DataFrame(rows)

# -------------------------------
# DIXON COLES LIKELIHOOD
# -------------------------------

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
    n_teams = len(teams)
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
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


# -------------------------------
# APP
# -------------------------------

st.title("Modelo Profesional Dixon-Coles MLE")

country = st.text_input("País (ej: Colombia)")

if country:

    leagues = get_leagues(country)
    league_name = st.selectbox("Liga", list(leagues.keys()))

    if st.button("Entrenar Modelo"):

        league_id = leagues[league_name]
        df = get_matches(league_id)

        teams = sorted(list(set(df["home"]).union(set(df["away"]))))
        n_teams = len(teams)

        init_params = np.concatenate([
            np.zeros(n_teams),   # ataque
            np.zeros(n_teams),   # defensa
            [0],                 # gamma
            [0.1]                # rho
        ])

        with st.spinner("Optimización en progreso..."):

            result = minimize(
                log_likelihood,
                init_params,
                args=(df, teams),
                method="L-BFGS-B"
            )

        params = result.x

        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        gamma = params[-2]
        rho = params[-1]

        st.success("Modelo entrenado correctamente")

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

            st.write(f"Local: {home_win*100:.2f}%")
            st.write(f"Empate: {draw*100:.2f}%")
            st.write(f"Visitante: {away_win*100:.2f}%")
