import mysql.connector
import bcrypt
import streamlit as st
from datetime import datetime


# Paramètres de connexion MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',       # Remplace par ton utilisateur MySQL
    'password': '',       # Remplace par ton mot de passe MySQL
    'database': 'pfa_melanome'
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def hash_password(plain_text_password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_text_password.encode('utf-8'), salt)
    
    # Store this returned value in your database (decode to string if your DB column is VARCHAR)
    return hashed.decode('utf-8')

def verify_password(password, hashed):
    # 1. Guard against empty inputs or NoneTypes
    if not password or not hashed:
        return False

    # 2. Convert password to bytes safely
    if isinstance(password, str):
        password = password.encode('utf-8')

    # 3. Convert hashed to bytes safely (prevents AttributeError if it's already bytes)
    if isinstance(hashed, str):
        hashed = hashed.encode('utf-8')

    # 4. Catch ValueError in case the database contains plain-text or corrupt hashes
    try:
        return bcrypt.checkpw(password, hashed)
    except ValueError as e:
        print(f"Bcrypt verification failed (likely invalid hash format in DB): {e}")
        return False

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and verify_password(password, user['password_hash']):
        return user
    return None

def create_user(username, password, role='user'):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)", 
                       (username, hash_password(password), role))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False # Le nom d'utilisateur existe déjà
    finally:
        conn.close()

def log_action(user_id, action):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history (user_id, action) VALUES (%s, %s)", (user_id, action))
    conn.commit()
    conn.close()

def save_diagnosis(user_id, age, sex, localization, image_path, top_prediction, top_probability):
    """Sauvegarde un diagnostic dans la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO diagnoses (user_id, age, sex, localization, image_path, top_prediction, top_probability) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (user_id, age, sex, localization, image_path, top_prediction, top_probability))
    conn.commit()
    conn.close()

def get_user_diagnoses(user_id):
    """Récupère tout l'historique d'un utilisateur spécifique."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM diagnoses WHERE user_id = %s ORDER BY created_at DESC"
    cursor.execute(query, (user_id,))
    diagnoses = cursor.fetchall()
    conn.close()
    return diagnoses