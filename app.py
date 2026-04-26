import streamlit as st
from src.db_auth import authenticate_user, log_action

st.set_page_config(page_title="Connexion - IA Mélanome", page_icon="🔒")

# Initialisation des variables de session
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = ''
    st.session_state['role'] = ''

if st.session_state['logged_in']:
    st.success(f"Bienvenue, {st.session_state['username']} ! Utilisez le menu à gauche pour naviguer.")
    if st.button("Se déconnecter"):
        log_action(st.session_state['user_id'], "Déconnexion")
        st.session_state.clear()
        st.rerun()
else:
    st.title("🔒 Portail de Connexion")
    st.write("Veuillez vous identifier pour accéder à l'outil de diagnostic.")

    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        user = authenticate_user(username, password)
        if user:
            st.session_state['logged_in'] = True
            st.session_state['user_id'] = user['id']
            st.session_state['username'] = user['username']
            st.session_state['role'] = user['role']
            
            log_action(user['id'], "Connexion réussie")
            st.rerun()
        else:
            st.error("Identifiants incorrects.")