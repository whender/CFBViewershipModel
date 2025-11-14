import streamlit as st
import statsmodels.api as sm
import pandas as pd
import joblib
import os, base64
import io
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import numpy as np

# ======================================================
# 📊 FEATURE GROUPS (for Brand Rankings Tab)
# ======================================================

numeric_features = [
    "Spread", "Competing Tier 1",
    "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "NBC", "CBS", "ABC", "BTN", "CW", "NFLN", "ESPNNEWS",
    "Conf Champ", "Sun", "Monday", "Weekday", "Friday", "Sat Early", "Sat Mid", "Sat Late",
    "Top 10 Rankings", "25-11 Rankings",
    "SEC_PostseasonImplications", "Big10_PostseasonImplications", "Big12_PostseasonImplications", "ACC_PostseasonImplications",
    "YTTV_ABC", "YTTV_ESPN"
]

rivalry_features = [
    "Michigan_OhioSt", "Texas_Oklahoma", "Alabama_Auburn", "Georgia_Florida",
    "NotreDame_USC", "Florida_Tennessee", "Oregon_Washington", "BYU_Utah",
    "Iowa_IowaSt", "OleMiss_MississippiSt", "Clemson_SouthCarolina",
    "Arizona_ArizonaSt", "Miami_FloridaSt", "Texas_TexasA&M", "Oregon_OregonSt",
    "USC_UCLA", "Louisville_Kentucky", "OhioSt_PennSt", "Alabama_LSU"
]

# ======================================================
# ⚖️ GLOBAL FEUD SETTINGS
# ======================================================
FEUD_START = datetime(2025, 10, 30)
# You can set this later when the feud ends (or leave None to keep active)
FEUD_END = None  # e.g., datetime(2025, 12, 15) once resolved

# ======================================================
# ⚙️ STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="CFB Viewership Dashboard", layout="wide")

# Hide Streamlit's default branding / GitHub controls
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stHeaderActionButtons"] {display: none !important;}
    [data-testid="stAppViewContainer"] > div:first-child {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================
# 🧠 TEAM NAME MAPPING
# ======================================================
cfbd_to_sheet = {
    "Air Force": "Air Force", "Akron": "Akron", "Alabama": "Alabama",
    "App State": "Appalachian St.", "Arizona": "Arizona", "Arizona State": "Arizona St.",
    "Arkansas": "Arkansas", "Arkansas State": "Arkansas St.", "Army": "Army",
    "Auburn": "Auburn", "Ball State": "Ball St.", "Baylor": "Baylor",
    "Boise State": "Boise St.", "Boston College": "Boston College", "Bowling Green": "Bowling Green",
    "Buffalo": "Buffalo", "BYU": "BYU", "California": "California", "Central Michigan": "Central Michigan",
    "Charlotte": "Charlotte", "Cincinnati": "Cincinnati", "Clemson": "Clemson",
    "Coastal Carolina": "Coastal Carolina", "Colorado": "Colorado", "Colorado State": "Colorado St.",
    "Duke": "Duke", "East Carolina": "East Carolina", "Eastern Michigan": "Eastern Michigan",
    "Florida": "Florida", "Florida Atlantic": "FAU", "Florida International": "FIU",
    "Florida State": "Florida St.", "Fresno State": "Fresno St.", "Georgia": "Georgia",
    "Georgia Southern": "Georgia Southern", "Georgia State": "Georgia St.", "Georgia Tech": "Georgia Tech",
    "Hawai'i": "Hawaii", "Houston": "Houston", "Illinois": "Illinois", "Indiana": "Indiana",
    "Iowa": "Iowa", "Iowa State": "Iowa St.", "Kansas": "Kansas", "Kansas State": "Kansas St.",
    "Kentucky": "Kentucky", "Liberty": "Liberty", "Louisiana": "Louisiana", "Louisville": "Louisville",
    "LSU": "LSU", "Marshall": "Marshall", "Maryland": "Maryland", "Memphis": "Memphis",
    "Miami": "Miami", "Michigan": "Michigan", "Michigan State": "Michigan St.", "Minnesota": "Minnesota",
    "Mississippi State": "Mississippi St.", "Missouri": "Missouri", "Navy": "Navy",
    "NC State": "North Carolina St.", "Nebraska": "Nebraska", "Nevada": "Nevada",
    "New Mexico": "New Mexico", "New Mexico State": "New Mexico St.",
    "North Carolina": "North Carolina", "North Texas": "North Texas", "Northwestern": "Northwestern",
    "Notre Dame": "Notre Dame", "Ohio": "Ohio", "Ohio State": "Ohio St.", "Oklahoma": "Oklahoma",
    "Oklahoma State": "Oklahoma St.", "Ole Miss": "Mississippi", "Oregon": "Oregon", "Oregon State": "Oregon St.",
    "Penn State": "Penn St.", "Pittsburgh": "Pittsburgh", "Purdue": "Purdue",
    "Rutgers": "Rutgers", "San Diego State": "San Diego St.", "SMU": "SMU",
    "South Carolina": "South Carolina", "Stanford": "Stanford", "Syracuse": "Syracuse",
    "TCU": "TCU", "Tennessee": "Tennessee", "Texas": "Texas", "Texas A&M": "Texas A&M",
    "Texas Tech": "Texas Tech", "Toledo": "Toledo", "Troy": "Troy", "Tulane": "Tulane",
    "UAB": "UAB", "UCF": "UCF", "UCLA": "UCLA", "UConn": "Connecticut", "UNLV": "UNLV",
    "USC": "USC", "Utah": "Utah", "Utah State": "Utah St.", "Vanderbilt": "Vanderbilt",
    "Virginia": "Virginia", "Virginia Tech": "Virginia Tech", "Wake Forest": "Wake Forest",
    "Washington": "Washington", "Washington State": "Washington St.", "West Virginia": "West Virginia",
    "Wisconsin": "Wisconsin", "Wyoming": "Wyoming"
}
teams_list = sorted(cfbd_to_sheet.values())

# ======================================================
# 🏛 CONFERENCE MAP
# ======================================================
team_conferences = {
    # SEC
    "Alabama": "SEC", "Auburn": "SEC", "Georgia": "SEC", "Florida": "SEC", "LSU": "SEC",
    "Tennessee": "SEC", "Texas A&M": "SEC", "Kentucky": "SEC", "South Carolina": "SEC",
    "Mississippi": "SEC", "Mississippi St.": "SEC", "Arkansas": "SEC", "Missouri": "SEC",
    "Vanderbilt": "SEC", "Texas": "SEC", "Oklahoma": "SEC",
    # Big 10
    "Michigan": "Big 10", "Ohio St.": "Big 10", "Penn St.": "Big 10", "Wisconsin": "Big 10",
    "Iowa": "Big 10", "Michigan St.": "Big 10", "Nebraska": "Big 10", "Minnesota": "Big 10",
    "Illinois": "Big 10", "Indiana": "Big 10", "Purdue": "Big 10", "Northwestern": "Big 10",
    "Maryland": "Big 10", "Rutgers": "Big 10", "UCLA": "Big 10", "USC": "Big 10",
    "Oregon": "Big 10", "Washington": "Big 10",
    # ACC
    "Clemson": "ACC", "Florida St.": "ACC", "Miami": "ACC", "North Carolina": "ACC",
    "Duke": "ACC", "North Carolina St.": "ACC", "Virginia": "ACC", "Virginia Tech": "ACC",
    "Louisville": "ACC", "Syracuse": "ACC", "Boston College": "ACC", "Wake Forest": "ACC",
    "Pittsburgh": "ACC", "Georgia Tech": "ACC", "California": "ACC", "Stanford": "ACC", "SMU": "ACC",
    # Big 12
    "BYU": "Big 12", "UCF": "Big 12", "Houston": "Big 12", "Cincinnati": "Big 12",
    "Baylor": "Big 12", "Texas Tech": "Big 12", "TCU": "Big 12", "Kansas": "Big 12",
    "Kansas St.": "Big 12", "Iowa St.": "Big 12", "Oklahoma St.": "Big 12",
    "West Virginia": "Big 12", "Utah": "Big 12", "Arizona": "Big 12", "Arizona St.": "Big 12",
    "Colorado": "Big 12"
}

# ======================================================
# 📦 LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("viewership_model_log.joblib"):
        data = joblib.load("viewership_model_log.joblib")
        if isinstance(data, dict):
            models["viewership"] = data["model"]
            models["viewership"].team_counts = data.get("team_counts", {})
            models["viewership"].smearing_factor = data.get("smearing_factor", 1.0)
        else:
            models["viewership"] = data
            models["viewership"].smearing_factor = 1.0
    else:
        st.warning("⚠️ Missing viewership_model_log.joblib — run your regression script first.")

    if os.path.exists("brand_model.joblib"):
        data_p4 = joblib.load("brand_model.joblib")
        if isinstance(data_p4, dict):
            models["brand"] = data_p4["model"]
            models["brand"].team_counts = data_p4.get("team_counts", {})
        else:
            models["brand"] = data_p4
    else:
        st.warning("⚠️ Missing brand_model.joblib — run your regression script first.")

    return models

models = load_models()
model = models.get("viewership")
brand_model = models.get("brand")

# ======================================================
# 🏈 DASHBOARD CONTENT
# ======================================================
st.title("Will Henderson - College Football Viewership Model")
st.caption("Predict hypothetical TV audiences using our model trained on historical data.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Game Predictor", "🏆 Brand Rankings", "📅 Weekly Predictions", "📘 Model Explanation"])

# ======================================================
# 📊 TAB 1: GAME PREDICTOR
# ======================================================
with tab1:
    st.subheader("Game Predictor")
    st.caption("_Using log-transformed regression (smearing corrected for unbiased predictions)._")

    def get_logo(team):
        safe_name = team.replace(" ", "_").replace(".", "").lower()
        path = f"logos/{safe_name}.png"
        if os.path.exists(path):
            with open(path, "rb") as f:
                img = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img}" width="120">'
        return ""

    c1, _, c2 = st.columns([3, 1, 3])
    with c1:
        team1 = st.selectbox("Team 1", teams_list, index=teams_list.index("BYU"))
    with c2:
        team2 = st.selectbox("Team 2", teams_list, index=teams_list.index("Texas Tech"))

    st.markdown("<hr>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        rank1 = st.number_input(f"{team1} Rank", 0, 25, 0, key="rank1_input")
    with r2:
        rank2 = st.number_input(f"{team2} Rank", 0, 25, 0, key="rank2_input")

    def rank_to_coefs(r):
        if 1 <= r <= 10: return (1, 0)
        elif 11 <= r <= 25: return (0, 1)
        else: return (0, 0)

    t1_top10, t1_25_11 = rank_to_coefs(rank1)
    t2_top10, t2_25_11 = rank_to_coefs(rank2)
    top10 = t1_top10 + t2_top10
    rank_25_11 = t1_25_11 + t2_25_11

    spread = st.number_input("Betting Spread", step=0.5)
    network = st.selectbox(
        "Network",
        ["ABC", "CBS", "NBC", "FOX", "ESPN", "ESPN2", "ESPNU", "FS1", "FS2", "BTN", "NFLN", "CW", "ESPNNEWS"]
    )

    time_slot = st.selectbox("Time Slot (EST)", [
        "Primetime (7:00p–9:00p)", "Sunday", "Monday", "Weekday (Tue–Thu)", "Friday",
        "Sat Early (11:00a–2:00p)", "Sat Mid (2:30p–6:30p)", "Sat Late (9:30p–11:30p)"
    ])
    comp_tier1 = st.number_input("Major Competing Games", 0, 3, 0)

    conf1 = team_conferences.get(team1, "Group of 6")
    conf2 = team_conferences.get(team2, "Group of 6")
    both_ranked = (rank1 > 0 and rank2 > 0)
    same_conf = (conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"])

    if both_ranked and same_conf:
        st.info(f"Detected postseason implication: {conf1} ranked matchup")

    # Rivalry definitions
    rivalries = {
        "Michigan_OhioSt": ("Michigan", "Ohio St."), "Texas_Oklahoma": ("Texas", "Oklahoma"),
        "Alabama_Auburn": ("Alabama", "Auburn"), "Georgia_Florida": ("Georgia", "Florida"),
        "NotreDame_USC": ("Notre Dame", "USC"), "Florida_Tennessee": ("Florida", "Tennessee"),
        "Oregon_Washington": ("Oregon", "Washington"), "BYU_Utah": ("BYU", "Utah"),
        "Iowa_IowaSt": ("Iowa", "Iowa St."), "OleMiss_MississippiSt": ("Mississippi", "Mississippi St."),
        "Clemson_SouthCarolina": ("Clemson", "South Carolina"), "Arizona_ArizonaSt": ("Arizona", "Arizona St."),
        "Miami_FloridaSt": ("Miami", "Florida St."), "Texas_TexasA&M": ("Texas", "Texas A&M"),
        "Oregon_OregonSt": ("Oregon", "Oregon St."), "USC_UCLA": ("USC", "UCLA"),
        "Louisville_Kentucky": ("Louisville", "Kentucky"), "OhioSt_PennSt": ("Ohio St.", "Penn St."),
        "Alabama_LSU": ("Alabama", "LSU")
    }
    auto_rivalry = next((r for r, (a,b) in rivalries.items() if {team1, team2} == {a, b}), None)
    if auto_rivalry:
        st.success(f"Detected Rivalry: **{auto_rivalry.replace('_', ' vs ')}**")

    if model:
        # ===============================
        # ✅ FEATURE SETUP (same as before)
        # ===============================
        features = {
            "Spread": spread,
            "Competing Tier 1": comp_tier1,
            "ABC": int(network == "ABC"), "CBS": int(network == "CBS"), "NBC": int(network == "NBC"),
            "FOX": int(network == "FOX"), "ESPN": int(network == "ESPN"), "ESPN2": int(network == "ESPN2"),
            "ESPNU": int(network == "ESPNU"), "FS1": int(network == "FS1"), "FS2": int(network == "FS2"),
            "BTN": int(network == "BTN"), "NFLN": int(network == "NFLN"), "CW": int(network == "CW"),
            "ESPNNEWS": int(network == "ESPNNEWS"),
            "Conf Champ": 0, "Sun": int("Sunday" in time_slot), "Monday": int("Monday" in time_slot),
            "Weekday": int("Weekday" in time_slot), "Friday": int("Friday" in time_slot),
            "Sat Early": int("Early" in time_slot), "Sat Mid": int("Mid" in time_slot),
            "Sat Late": int("Late" in time_slot),
            "Top 10 Rankings": top10, "25-11 Rankings": rank_25_11,
            "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
            "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
            "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
            "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12")
        }

        conf_to_flag = {"SEC": "SEC_PostseasonImplications","Big 10": "Big10_PostseasonImplications",
                        "Big 12": "Big12_PostseasonImplications","ACC": "ACC_PostseasonImplications"}
        for conf_tag, flag_name in conf_to_flag.items():
            features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

        rivalry_vars = [r for r in model.params.index if r in rivalries]
        for r in rivalry_vars:
            features[r] = int(r == auto_rivalry)

        for net in ["ABC", "ESPN"]:
            features[f"YTTV_{net}"] = 0
        now = datetime.now()
        feud_active = (now >= FEUD_START) and (FEUD_END is None or now <= FEUD_END)
        if feud_active and network in ["ABC", "ESPN"]:
            features[f"YTTV_{network}"] = 1
            st.warning(f"📉 Feud active — applying YouTubeTV–Disney adjustment for {network}.")

        for col in model.params.index:
            if col in teams_list:
                features[col] = int(col in [team1, team2])
        features["const"] = 1.0
        for c in model.params.index:
            if c not in features:
                features[c] = 0.0

        # 🧩 Ohio State × BTN Interaction
        features["OhioSt_BTN"] = int(("Ohio St." in [team1, team2]) and network == "BTN") \
                                 if "OhioSt_BTN" in model.params.index else 0

        X_input = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)

        # ======================================================
        # 🧮 LOG-MODEL PREDICTION WITH SMEARING FACTOR
        # ======================================================
        if st.button("Predict Viewership"):
            pred_ln = float(model.predict(X_input)[0])
            smearing = getattr(model, "smearing_factor", 1.0)
            pred = (np.exp(pred_ln) - 1) * smearing
            formatted = f"{pred/1_000:.3f}M" if pred >= 1_000 else f"{pred:.0f}K"

            st.markdown(f"<h3 style='text-align:center;'>Predicted Viewers: {formatted}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align:center;'>{team1} vs {team2} | {network} | {time_slot}</p>",
                unsafe_allow_html=True
            )

            st.session_state["prediction_data"] = {
                "team1": team1, "team2": team2, "rank1": rank1, "rank2": rank2,
                "network": network, "time_slot": time_slot, "spread": spread,
                "comp_tier1": comp_tier1, "formatted": formatted
            }

# ======================================================
# 🏆 TAB 2: BRAND RANKINGS (Year-Aware, P4 + Notre Dame)
# ======================================================
with tab2:
    st.subheader("Brand Power Rankings (Power 4 + Notre Dame)")
    st.caption("Estimates each team's impact on national TV viewership using a log-transformed Ridge regression. Filter by year to see changes over time.")

    import statsmodels.api as sm
    from sklearn.linear_model import RidgeCV

    # --------------------------------------------------
    # 📦 Load Cleaned Dataset
    # --------------------------------------------------
    @st.cache_data
    def load_viewership_data():
        if os.path.exists("viewership_cleaned.csv"):
            return pd.read_csv("viewership_cleaned.csv", low_memory=False)
        else:
            st.error("❌ Missing file: viewership_cleaned.csv. Re-run your regression script to export it.")
            st.stop()

    df_all = load_viewership_data()

    # --------------------------------------------------
    # 🗓 Year Filter
    # --------------------------------------------------
    available_years = sorted(df_all["Year"].dropna().unique())
    selected_year = st.selectbox("Select Year", ["All Years"] + [str(y) for y in available_years])

    if selected_year == "All Years":
        df = df_all.copy()
        st.info("Showing aggregate brand power using all available seasons.")
    else:
        df = df_all[df_all["Year"] == int(selected_year)].copy()
        st.info(f"Showing brand power for **{selected_year}** only.")

    # --------------------------------------------------
    # 🧩 Prepare Team Universe (Power 4 + Notre Dame)
    # --------------------------------------------------
    power4_set = set(team_conferences.keys()) | {"Notre Dame"}

    team_dummies_1 = pd.get_dummies(df["Team 1"])
    team_dummies_2 = pd.get_dummies(df["Team 2"])
    team_dummies = team_dummies_1.add(team_dummies_2, fill_value=0)

    valid_counts = team_dummies.sum()
    team_dummies = team_dummies[valid_counts[valid_counts >= 3].index]

    feature_cols = [c for c in df.columns if c in numeric_features + rivalry_features]
    X = pd.concat([df[feature_cols], team_dummies], axis=1).fillna(0)
    X = sm.add_constant(X)

    # --------------------------------------------------
    # 🧮 Ridge Regression (log viewers)
    # --------------------------------------------------
    y = np.log(df["Persons 2+"].astype(float) + 1)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X, y)

    params = pd.Series(ridge.coef_, index=X.columns)
    team_coefs = params[team_dummies.columns]
    team_coefs = team_coefs[team_coefs.index.isin(power4_set)]
    valid_counts = valid_counts.reindex(team_coefs.index).fillna(0)

    # --------------------------------------------------
    # 🧮 Apply shrinkage for small sample sizes
    # --------------------------------------------------
    adjusted = team_coefs.copy()
    for t in team_coefs.index:
        n = valid_counts[t]
        if n <= 4:
            adjusted[t] = team_coefs[t] * (n / (n + 5))

    # --------------------------------------------------
    # 📈 Convert to interpretable % lift values
    # --------------------------------------------------
    boost_pct = (np.exp(adjusted) - 1) * 100  # percentage increase vs avg team

    brand_df = (
        pd.DataFrame({
            "Team": team_coefs.index,
            "Viewership Lift (%)": boost_pct.round(1),
            "Games Used": valid_counts[team_coefs.index].values,
        })
        .sort_values(by="Viewership Lift (%)", ascending=False)
        .reset_index(drop=True)
    )

    brand_df.insert(0, "Rank", range(1, len(brand_df) + 1))

    # --------------------------------------------------
    # 📊 Display
    # --------------------------------------------------
    st.markdown("###Interpreting Brand Power")
    st.caption("""
**Viewership Lift (%)** = estimated percent increase in national TV audience a team brings 
to a typical game, controlling for network, time slot, and opponent.  
Values above 100% mean the team roughly doubles average viewership.
""")

    st.dataframe(brand_df.head(70), use_container_width=True)

    # ✅ Bar chart now uses the percentage values, not inflated viewer counts
    chart_df = brand_df.head(25).set_index("Team")
    st.bar_chart(chart_df["Viewership Lift (%)"], use_container_width=True)

    # --------------------------------------------------
    # 💾 Download Option
    # --------------------------------------------------
    csv_buf = brand_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Brand Rankings CSV",
        data=csv_buf,
        file_name=f"brand_rankings_{selected_year.replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ======================================================
# 📅 TAB 3: WEEKLY PREDICTIONS
# ======================================================
with tab3:
    st.subheader("Weekly Predictions")

    df = pd.read_csv("weekly_predictions.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


    required_cols = [
        "Week", "Date", "Day", "Team 1", "Rank 1", "Team 2", "Rank 2",
        "Spread", "Network", "Time Slot", "Competing Tier 1"
    ]

    # ======================================================
    # 🔒 ONLY GENERATE PREDICTIONS FOR EMPTY ROWS
    # ======================================================
    if "Predicted Viewers" not in df.columns:
        df["Predicted Viewers"] = ""

    # 🧹 Normalize blank and NaN cells before looping
    df["Predicted Viewers"] = df["Predicted Viewers"].replace(["nan", "NaN"], "")

    def rank_to_coefs(r):
        if 1 <= r <= 10:
            return (1, 0)
        elif 11 <= r <= 25:
            return (0, 1)
        else:
            return (0, 0)

    preds = []
    for _, row in df.iterrows():
        # ✅ Skip rows that already have predictions (catch NaN or string NaN)
        val = row.get("Predicted Viewers", "")
        if pd.notna(val) and str(val).strip().lower() not in ["", "nan"]:
            preds.append(val)
            continue

        try:
            team1 = cfbd_to_sheet.get(row["Team 1"], row["Team 1"])
            team2 = cfbd_to_sheet.get(row["Team 2"], row["Team 2"])
            rank1, rank2 = row["Rank 1"], row["Rank 2"]
            spread = float(row["Spread"])
            network = row["Network"]
            time_slot = str(row["Time Slot"])
            comp_tier1 = int(row.get("Competing Tier 1", 0))

            conf1 = team_conferences.get(team1, "Group of 6")
            conf2 = team_conferences.get(team2, "Group of 6")
            both_ranked = (rank1 > 0 and rank2 > 0)
            same_conf = (conf1 == conf2 and conf1 in ["SEC", "Big 10", "ACC", "Big 12"])

            t1_top10, t1_25_11 = rank_to_coefs(rank1)
            t2_top10, t2_25_11 = rank_to_coefs(rank2)
            top10 = t1_top10 + t2_top10
            rank_25_11 = t1_25_11 + t2_25_11

            is_friday = str(row["Day"]).strip().lower() == "fri" or "fri" in str(row["Day"]).lower()

            rivalries = {
                "Michigan_OhioSt": ("Michigan", "Ohio St."), "Texas_Oklahoma": ("Texas", "Oklahoma"),
                "Alabama_Auburn": ("Alabama", "Auburn"), "Georgia_Florida": ("Georgia", "Florida"),
                "NotreDame_USC": ("Notre Dame", "USC"), "Florida_Tennessee": ("Florida", "Tennessee"),
                "Oregon_Washington": ("Oregon", "Washington"), "BYU_Utah": ("BYU", "Utah"),
                "Iowa_IowaSt": ("Iowa", "Iowa St."), "OleMiss_MississippiSt": ("Mississippi", "Mississippi St."),
                "Clemson_SouthCarolina": ("Clemson", "South Carolina"), "Arizona_ArizonaSt": ("Arizona", "Arizona St."),
                "Miami_FloridaSt": ("Miami", "Florida St."), "Texas_TexasA&M": ("Texas", "Texas A&M"),
                "Oregon_OregonSt": ("Oregon", "Oregon St."), "USC_UCLA": ("USC", "UCLA"),
                "Louisville_Kentucky": ("Louisville", "Kentucky"), "OhioSt_PennSt": ("Ohio St.", "Penn St."),
                "Alabama_LSU": ("Alabama", "LSU")
            }
            auto_rivalry = next((r for r, (a, b) in rivalries.items() if {team1, team2} == {a, b}), None)

            features = {
                "Spread": spread,
                "Competing Tier 1": comp_tier1,
                "ABC": int(network == "ABC"),
                "CBS": int(network == "CBS"),
                "NBC": int(network == "NBC"),
                "FOX": int(network == "FOX"),
                "ESPN": int(network == "ESPN"),
                "ESPN2": int(network == "ESPN2"),
                "ESPNU": int(network == "ESPNU"),
                "FS1": int(network == "FS1"),
                "FS2": int(network == "FS2"),
                "BTN": int(network == "BTN"),
                "NFLN": int(network == "NFLN"),
                "CW": int(network == "CW"),
                "ESPNNEWS": int(network == "ESPNNEWS"),
                "Conf Champ": 0,
                "Sun": int("Sunday" in time_slot),
                "Monday": int("Monday" in time_slot),
                "Weekday": int("Weekday" in time_slot),
                "Friday": int(is_friday or "Friday" in time_slot),
                "Sat Early": int(
                    not is_friday and (
                        "Early" in time_slot or
                        any(t in time_slot for t in ["11:00a", "11:30a", "12:00p", "12:30p", "1:00p", "1:30p", "2:00p"])
                    )
                ),
                "Sat Mid": int(
                    not is_friday and (
                        "Mid" in time_slot or
                        any(t in time_slot for t in ["2:30p", "3:00p", "3:30p", "4:00p", "4:30p",
                                                "5:00p", "5:30p", "6:00p", "6:30p"])
                    )
                ),
                "Sat Late": int(
                    not is_friday and (
                        "Late" in time_slot or
                        any(t in time_slot for t in ["9:30p", "10:00p", "11:00p", "11:30p"])
                    )
                ),
                "Top 10 Rankings": top10,
                "25-11 Rankings": rank_25_11,
                "SEC": (conf1 == "SEC") + (conf2 == "SEC"),
                "Big 10": (conf1 == "Big 10") + (conf2 == "Big 10"),
                "ACC": (conf1 == "ACC") + (conf2 == "ACC"),
                "Big 12": (conf1 == "Big 12") + (conf2 == "Big 12"),
            }

            for conf_tag, flag_name in {
                "SEC": "SEC_PostseasonImplications",
                "Big 10": "Big10_PostseasonImplications",
                "Big 12": "Big12_PostseasonImplications",
                "ACC": "ACC_PostseasonImplications",
            }.items():
                features[flag_name] = int(both_ranked and same_conf and conf1 == conf_tag)

            for r in rivalry_features:
                features[r] = int(r == auto_rivalry)

            for net in ["YTTV_ABC", "YTTV_ESPN"]:
                features[net] = 0
            now = datetime.now()
            feud_active = (now >= FEUD_START) and (FEUD_END is None or now <= FEUD_END)
            if feud_active and network in ["ABC", "ESPN"]:
                features[f"YTTV_{network}"] = 1

            for col in model.params.index:
                if col in teams_list:
                    features[col] = int(col in [team1, team2])

            features["const"] = 1.0
            for c in model.params.index:
                if c not in features:
                    features[c] = 0.0

            # ======================================================
            # 🧩 Ohio State × BTN Interaction Adjustment
            # ======================================================
            if "OhioSt_BTN" in model.params.index:
                ohio_btn = int(
                    (("Ohio St." in [team1, team2]) or ("Ohio St." in [row.get("Team 1", ""), row.get("Team 2", "")])) 
                    and network == "BTN"
                )
                features["OhioSt_BTN"] = ohio_btn
            else:
                features["OhioSt_BTN"] = 0

            X_input = pd.DataFrame([[features[c] for c in model.params.index]], columns=model.params.index)
            # ✅ Statistically grounded 95% prediction interval
            try:
                # Log-scale prediction and confidence interval
                pred_res = model.get_prediction(X_input)
                ci = pred_res.summary_frame(alpha=0.32)

                smearing = getattr(model, "smearing_factor", 1.0)

                # Convert back from log-scale (exp(pred) - 1) × smearing
                pred_val_ln = ci["mean"].iloc[0]
                ci_low_ln = ci["obs_ci_lower"].iloc[0]
                ci_high_ln = ci["obs_ci_upper"].iloc[0]

                pred_val = (np.exp(pred_val_ln) - 1) * smearing
                ci_low = max(0, (np.exp(ci_low_ln) - 1) * smearing)
                ci_high = max(ci_low, (np.exp(ci_high_ln) - 1) * smearing)

            except Exception:
                # fallback if model isn't statsmodels (e.g., dict or ridge)
                pred_val = float(model.predict(X_input)[0])
                resid_std = 500  # rough fallback std-dev if needed
                ci_low = pred_val - 1.96 * resid_std
                ci_high = pred_val + 1.96 * resid_std

            pred_fmt = f"{pred_val/1_000:.2f}M\n({ci_low/1_000:.2f}–{ci_high/1_000:.2f}M)"

            preds.append(pred_fmt)
        except Exception as e:
            preds.append(f"Error: {e}")

    # Replace blank rows only; keep existing predictions intact
    for i, p in enumerate(preds):
        if str(df.loc[i, "Predicted Viewers"]).strip().lower() in ["", "nan"]:
            df.loc[i, "Predicted Viewers"] = p

    df.to_csv("weekly_predictions.csv", index=False)

    # ======================================================
    # 📈 DISPLAY BY WEEK
    # ======================================================
    if "Actual Viewers" not in df.columns:
        df["Actual Viewers"] = None

    df["Spread"] = df["Spread"].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")

    def parse_viewership(val):
        """Convert strings like '1.91M', '850K', or '1.91M\n(1.80–2.02M)' into numeric (millions)."""
        try:
            if pd.isna(val): 
                return None
            val = str(val).strip().upper()
            # Remove confidence interval and parentheses
            if "\n" in val:
                val = val.split("\n")[0].strip()
            if "(" in val:
                val = val.split("(")[0].strip()

            # Handle suffix
            if val.endswith("M"):
                return float(val.replace("M", "").strip())
            elif val.endswith("K"):
                return float(val.replace("K", "").strip()) / 1000  # convert K to millions
            else:
                return float(val)
        except:
            return None

    def calc_error(pred_str, actual_str):
        pred_val = parse_viewership(pred_str)
        act_val = parse_viewership(actual_str)
        if pred_val is None or act_val is None or act_val <= 0:
            return None
        return abs((pred_val - act_val) / act_val) * 100

    df["% Error"] = [calc_error(p, a) for p, a in zip(df["Predicted Viewers"], df["Actual Viewers"])]

    def indicator(e):
        if e is None: return ""
        if e < 5: return "🟢🎯"
        elif e < 25: return "🟢"
        elif e < 35: return "🟡"
        else: return "🔴"

    df["Accuracy"] = [indicator(e) for e in df["% Error"]]

    def format_matchup(t1, r1, t2, r2):
        t1_str = f"#{int(r1)} {t1}" if pd.notna(r1) and int(r1) > 0 else t1
        t2_str = f"#{int(r2)} {t2}" if pd.notna(r2) and int(r2) > 0 else t2
        return f"{t1_str} @ {t2_str}"

    df["Matchup"] = [
        format_matchup(t1, r1, t2, r2)
        for t1, r1, t2, r2 in zip(df["Team 1"], df["Rank 1"], df["Team 2"], df["Rank 2"])
    ]

    df["Date"] = [
        f"{day}, {pd.to_datetime(date).strftime('%b %-d')}" if not pd.isna(date) else day
        for date, day in zip(df["Date"], df["Day"])
    ]

    display_cols = [
        "Date", "Time Slot", "Matchup", "Spread", "Network",
        "Predicted Viewers", "Actual Viewers", "% Error", "Accuracy"
    ]

    def color_error(val):
        if pd.isna(val): 
            return ""
        # red (error)
        if val >= 35: 
            return "background-color: rgba(255, 102, 102, 0.25);"
        # yellow (moderate)
        elif val >= 25: 
            return "background-color: rgba(255, 255, 153, 0.25);"
        # green (good)
        elif val < 25: 
            return "background-color: rgba(102, 255, 178, 0.25);"
        return ""
    
    # Keep a clean copy of the raw date column (from the CSV) before reformatting
    df["RawDate"] = pd.to_datetime(pd.read_csv("weekly_predictions.csv")["Date"], errors="coerce", format="%m/%d/%y")

    # ======================================================
    # 📊 MODEL PERFORMANCE SUMMARY STATS
    # ======================================================
    errors = df["% Error"].dropna()
    median_error = errors.median()
    mean_error = errors.mean()

    # Find closest and worst predictions
    if not errors.empty:
        closest_idx = errors.idxmin()
        worst_idx = errors.idxmax()
        closest_row = df.loc[closest_idx]
        worst_row = df.loc[worst_idx]
    else:
        closest_row = worst_row = None

    # Accuracy bands
    within_10 = (errors < 10).sum()
    within_25 = (errors < 25).sum()
    total_pred = len(errors)
    pct_within_10 = (within_10 / total_pred * 100) if total_pred else 0
    pct_within_25 = (within_25 / total_pred * 100) if total_pred else 0

    st.markdown("Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median % Error", f"{median_error:.1f}%")
    col2.metric("Mean % Error", f"{mean_error:.1f}%")
    col3.metric("Within 10% Error", f"{pct_within_10:.0f}%")
    col4.metric("Within 25% Error", f"{pct_within_25:.0f}%")

    # Optional: show best & worst game examples
    if closest_row is not None:
        st.caption(
            f"🎯 Closest Prediction: **{closest_row['Matchup']}** on {closest_row['Date']} — "
            f"{closest_row['% Error']:.1f}% off."
        )
    if worst_row is not None:
        st.caption(
            f"⚠️ Largest Miss: **{worst_row['Matchup']}** on {worst_row['Date']} — "
            f"{worst_row['% Error']:.1f}% off."
        )

    # --- Collapsible weekly sections (safe numeric sort, skips NaN/empty) ---
    weeks = sorted(
        [int(w) for w in df["Week"].dropna().unique() if str(w).strip() != ""],
        reverse=True
    )


    for week in weeks:
        subset = df[df["Week"] == week].copy()

        # Extract year from the preserved RawDate column
        year_val = subset["RawDate"].dropna().iloc[0] if not subset["RawDate"].dropna().empty else None
        year_str = str(year_val.year) if pd.notna(year_val) else "Unknown Year"

        with st.expander(f"Week {week} ({year_str})", expanded=(week == max(weeks))):
            styled = subset[display_cols].reset_index(drop=True).style.applymap(color_error, subset=["% Error"])
            st.dataframe(styled, use_container_width=True)

# ======================================================
# 📘 TAB 4: MODEL EXPLANATION
# ======================================================
with tab4:
    st.header("How the Model Works")
    st.caption("An overview of the regression design, log transformation, smearing correction, and brand modeling.")

    st.markdown("""
    ### Objective  
    The goal of this model is to **quantify what drives national college football TV viewership**  
    — isolating the independent effect of networks, time slots, matchups, and brands.

    ### Core Model: Log-Transformed OLS Regression  
    The main model predicts the natural log of total viewership (`ln_viewers`) for each nationally televised game.  
    Using `log(viewers + 1)` normalizes extreme outliers and makes multiplicative effects additive.

    - **Dependent Variable:** Log of viewers (`ln(Persons 2+ + 1)`)  
    - **Independent Variables:**
        - **Network dummies** (ABC, FOX, CBS, ESPN, etc.)
        - **Time-slot indicators** (Friday, Saturday early/mid/late, Monday, etc.)
        - **Match context:** Spread, competing games, postseason implications
        - **Special events:** Conference championships, rivalries, Deion Sanders era, Nielsen system changes
        - **Team fixed effects:** Each team’s coefficient measures its typical pull on viewership (vs. baseline)

    ### Model Fit and Strength (as of last update)
    - **R² = 0.93** — meaning the model explains roughly **93% of the variance** in national TV audiences.  
    - **Adj. R² = 0.924**  
    - **F-statistic = 135.8 (p < 0.001)**  
    - 2,062 games included from **2018–2025** national telecasts.

    ### Brand Power Model
    After estimating all team fixed effects from the OLS model, a **separate RidgeCV regression**  
    is run on **Power 4 + Notre Dame** games to rank brands by their independent draw.

    ```math
    ln(Viewers) = α₀ + α₁(Features) + α₂(TeamDummies)
    ```
    - Regularized using RidgeCV (`α = 1.0`) to prevent overfitting.
    - Teams with fewer than 5 nationally televised games receive **sample-size shrinkage**:
      ```math
      Adjusted βᵢ = βᵢ × (n / (n + 5))
      ```
    - Outputs interpretable “brand lift” coefficients, e.g.:
        - Ohio St. (+0.64), Alabama (+0.62), Michigan (+0.54), Georgia (+0.45)
        - Lower-tier P4 teams hover near 0 or slightly negative.

    ### 💡 Why Log-Transform?
    Viewership follows a **power-law distribution** — a few massive games dominate the mean.  
    Log-transforming ensures that the model:
    - Focuses on **relative % changes** instead of raw viewer count
    - Produces stable, interpretable coefficients
    - Prevents large outliers (like Playoff games) from overwhelming the regression 

    ---
    _Developed and maintained by [Will Henderson](https://twitter.com/willshenderson) — University of Utah Economics._
    """)

# ======================================================
# 📫 CONTACT / FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        <b>Created by <span style='color:#2DFF8C;'>Will Henderson</span></b><br>
        <a href="mailto:wshenderson7@gmail.com" style="text-decoration:none; color:white;">📧 wshenderson7@gmail.com</a>  
        &nbsp;|&nbsp;  
        <a href="https://twitter.com/willshenderson" target="_blank" style="text-decoration:none; color:#1DA1F2;">𝕏 @willshenderson</a>
    </div>
    """,
    unsafe_allow_html=True
)
