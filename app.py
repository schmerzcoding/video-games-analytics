import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Configuración de página
st.set_page_config(page_title="Video Game Dashboard", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/video_games.csv")

    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convertir fecha
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    return df

df = load_data()

@st.cache_data
def build_recommender(df):
    df_model = df[["title", "genre", "console"]].dropna()

    encoder = OneHotEncoder(handle_unknown='ignore')
    features = encoder.fit_transform(df_model[["genre", "console"]])

    similarity = cosine_similarity(features)

    return df_model.reset_index(drop=True), similarity

df_model, similarity_matrix = build_recommender(df)

st.title("🎮 Video Game Market Intelligence Dashboard")

# Sidebar filtros
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (2000, 2015)
)

genre = st.sidebar.multiselect(
    "Select Genre",
    options=df["genre"].dropna().unique(),
    default=df["genre"].dropna().unique()[:5]
)

console = st.sidebar.multiselect(
    "Select Console",
    options=df["console"].dropna().unique(),
    default=df["console"].dropna().unique()[:5]
)

# Filtrar datos
filtered_df = df[
    (df["year"].between(year_range[0], year_range[1])) &
    (df["genre"].isin(genre)) &
    (df["console"].isin(console))
]

# KPIs
total_sales = filtered_df["total_sales"].sum()
num_games = filtered_df.shape[0]

col1, col2 = st.columns(2)
col1.metric("Total Global Sales (Millions)", f"{total_sales:.2f}")
col2.metric("Number of Games", num_games)

top_genre = (
    filtered_df.groupby("genre")["total_sales"]
    .sum()
    .idxmax()
)

top_region = filtered_df[["na_sales", "pal_sales", "jp_sales"]].sum().idxmax()

st.markdown(f"""
## 📊 Key Insights
- 🎮 Top Genre: **{top_genre}**
- 🌍 Strongest Market: **{top_region.upper()}**
- 📅 Total Games Analyzed: **{num_games}**
""")
st.markdown("---")

# 📈 Ventas en el tiempo
st.header("Sales Trend Over Time")

sales_trend = (
    filtered_df.groupby("year")["total_sales"]
    .sum()
    .reset_index()
)

fig = px.line(sales_trend, x="year", y="total_sales", title="Global Sales Over Time")
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# TOP 5 Juegos
st.header("TOP 5 Games")

top_games = (
    filtered_df.sort_values(by="total_sales", ascending=False)
    [["title", "console", "total_sales"]]
    .head(5)
)

st.dataframe(top_games)
st.markdown("---")

# 🌍 Ventas por región
st.header("Regional Sales Comparison")

regional_sales = filtered_df[["na_sales", "pal_sales", "jp_sales", "other_sales"]].sum()
regional_df = regional_sales.reset_index()
regional_df.columns = ["region", "sales"]

fig2 = px.bar(regional_df, x="region", y="sales", title="Sales by Region")
st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

# 🏆 Top publishers
st.header("Top Publishers")

top_publishers = (
    filtered_df.groupby("publisher")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig3 = px.bar(top_publishers, x="publisher", y="total_sales", title="Top Publishers")
st.plotly_chart(fig3, use_container_width=True)
st.markdown("---")

# 🏆 Top Developers
st.header("Top Developers")

top_dev = (
    filtered_df.groupby("developer")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_dev = px.bar(top_dev, x="developer", y="total_sales")
st.plotly_chart(fig_dev, use_container_width=True)
st.markdown("---")

# 🎮 Top consolas
st.header("Top Consoles")

top_consoles = (
    filtered_df.groupby("console")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(top_consoles, x="console", y="total_sales", title="Top Consoles")
st.plotly_chart(fig4, use_container_width=True)
st.markdown("---")

# ⭐ Critic Score vs Sales
st.header("Critic Score vs Sales")

fig5 = px.scatter(
    filtered_df,
    x="critic_score",
    y="total_sales",
    trendline="ols",
    title="Do Better Reviews Mean More Sales?"
)

st.plotly_chart(fig5, use_container_width=True)
st.markdown("---")

# 🔍 Buscador de juegos

st.header("🔍 Search Game")

search = st.text_input("Type a game title")

if search:
    results = df[df["title"].str.contains(search, case=False, na=False)].head(10)

    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image("https://www.pngall.com/wp-content/uploads/15/Video-Game-Controller-PNG-Pic.png", width=120)

        with col2:
            st.markdown(f"### {row['title']}")
            st.write(f"🎮 Console: {row['console']}")
            st.write(f"🎭 Genre: {row['genre']}")
            st.write(f"🏢 Publisher: {row['publisher']}")
            st.write(f"⭐ Critic Score: {row['critic_score']}")
            st.write(f"💰 Sales: {row['total_sales']}M")

        st.markdown("---")

st.header("🧠 Game Recommendation System")

selected_game = st.selectbox(
    "Choose a game you like",
    df_model["title"].unique()
)

if selected_game:
    idx = df_model[df_model["title"] == selected_game].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = [df_model.iloc[i[0]]["title"] for i in scores]

    st.subheader("🎯 Recommended Games")

    cols = st.columns(5)

    for i, game in enumerate(recommendations):
        with cols[i]:
            st.image("https://www.pngall.com/wp-content/uploads/15/Video-Game-Controller-PNG-Pic.png")
            st.write(game)