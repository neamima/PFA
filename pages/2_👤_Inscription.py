import streamlit as st
from src.db_auth import create_user

st.set_page_config(page_title="Créer un compte", page_icon="👤")

st.title("👤 Créer un nouveau profil Utilisateur")

new_user = st.text_input("Nouveau nom d'utilisateur")
new_pwd = st.text_input("Nouveau mot de passe", type="password")
new_pwd_confirm = st.text_input("Confirmer le mot de passe", type="password")

if st.button("S'inscrire"):
    if new_pwd != new_pwd_confirm:
        st.error("Les mots de passe ne correspondent pas.")
    elif len(new_pwd) < 6:
        st.error("Le mot de passe doit contenir au moins 6 caractères.")
    else:
        success = create_user(new_user, new_pwd, role='user')
        if success:
            st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter via la page d'accueil.")
            st.balloons()
        else:
            st.error("Ce nom d'utilisateur existe déjà.")