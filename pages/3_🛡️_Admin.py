import streamlit as st
import pandas as pd
from src.db_auth import get_db_connection

# --- SÉCURITÉ : Réservé à l'Admin ---
if not st.session_state.get('logged_in') or st.session_state.get('role') != 'admin':
    st.error("Accès refusé. Cette page est réservée aux administrateurs.")
    st.stop()
# ------------------------------------

st.title("🛡️ Tableau de Bord Administrateur")

# Onglets pour structurer l'interface
tab1, tab2 = st.tabs(["👥 Gestion des Utilisateurs", "📜 Historique d'Utilisation"])

conn = get_db_connection()

with tab1:
    st.subheader("Liste des comptes")
    query_users = "SELECT id, username, role, created_at FROM users"
    df_users = pd.read_sql(query_users, conn)
    st.dataframe(df_users, use_container_width=True)

with tab2:
    st.subheader("Logs d'activité du système")
    query_history = """
        SELECT h.timestamp, u.username, u.role, h.action 
        FROM history h
        JOIN users u ON h.user_id = u.id
        ORDER BY h.timestamp DESC
    """
    df_history = pd.read_sql(query_history, conn)
    st.dataframe(df_history, use_container_width=True)

conn.close()