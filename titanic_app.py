import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set page config for better layout
st.set_page_config(page_title="Titanic Data Analysis", layout="wide")

# Charger dataset Titanic depuis seaborn
@st.cache_data  # Cache for performance
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

# Sidebar for global options
st.sidebar.title("Options")
theme = st.sidebar.selectbox("Plot Theme", ["default", "dark", "whitegrid"])
if theme == "dark":
    sns.set_style("dark")
elif theme == "whitegrid":
    sns.set_style("whitegrid")
else:
    sns.set_style("default")

# Titre de l'application
st.title("🚢 Titanic Data Analysis App")

# Affichage rapide du dataset
if st.checkbox("Voir les données brutes"):
    st.write("### Aperçu des données")
    st.dataframe(df.head(10))  # Use st.dataframe for better interactivity
    st.write(f"Dataset shape: {df.shape}")
    st.write(df.describe(include='all'))  # Quick stats

# Choix d'une variable à analyser
feature = st.selectbox("Choisir une variable pour l'analyse:", ["age", "sex", "pclass", "fare"])

# Helper function for plots
@st.cache_data
def create_plot(feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    if feature == "age":
        # Handle NaNs
        df_clean = df.dropna(subset=['age'])
        sns.histplot(df_clean['age'], bins=30, kde=True, ax=ax, color="blue")
        ax.set_title("Distribution des âges (NaNs excluded)")
        ax.set_xlabel("Âge")
        ax.set_ylabel("Fréquence")
        st.write(f"**Stats pour 'age'**: Moyenne = {df['age'].mean():.1f}, Médiane = {df['age'].median():.1f}, NaNs = {df['age'].isna().sum()}")
    
    elif feature == "sex":
        # Use barplot for proportions instead of counts for better insight
        survival_by_sex = df.groupby('sex')['survived'].mean() * 100
        sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values, ax=ax, palette="Set2")
        ax.set_title("Taux de survie par sexe (%)")
        ax.set_ylabel("Taux de survie (%)")
        for i, v in enumerate(survival_by_sex.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
        st.write(f"**Taux de survie global**: {df['survived'].mean() * 100:.1f}%")
    
    elif feature == "pclass":
        sns.barplot(x="pclass", y="survived", data=df, ax=ax, palette="muted", ci=None)  # ci=None to remove error bars
        ax.set_title("Taux de survie par classe")
        ax.set_ylabel("Taux de survie moyen")
        for i, v in enumerate(ax.patches):
            ax.text(i, v.get_height() + 0.01, f"{v.get_height():.2f}", ha='center')
    
    elif feature == "fare":
        # New: Fare distribution
        df_clean = df.dropna(subset=['fare'])
        sns.histplot(df_clean['fare'], bins=30, kde=True, ax=ax, color="green")
        ax.set_title("Distribution des tarifs (fare)")
        ax.set_xlabel("Tarif (USD)")
        ax.set_ylabel("Fréquence")
        ax.set_yscale('log')  # Log scale for skewed data
        st.write(f"**Stats pour 'fare'**: Moyenne = ${df['fare'].mean():.2f}, Médiane = ${df['fare'].median():.2f}")
    
    plt.tight_layout()
    return fig

# Render the selected plot
if feature:
    st.subheader(f"Analyse de '{feature}'")
    fig = create_plot(feature)
    st.pyplot(fig)