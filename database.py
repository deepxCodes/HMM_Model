import sqlite3
import os
import hashlib

DB_NAME = "synapse.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    # Table for the base corpus (the larger, static foundational dataset)
    c.execute('''
        CREATE TABLE IF NOT EXISTS base_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL
        )
    ''')
    # Table for the user's manual interactions/corrections
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL
        )
    ''')
    # Table for user authentication
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_base_corpus():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT sentence FROM base_corpus')
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_user_corpus():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT sentence FROM user_corpus')
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def add_base_corpus(sentences):
    """Replaces the entire base corpus with a new set of sentences."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM base_corpus')
    c.executemany('INSERT INTO base_corpus (sentence) VALUES (?)', [(s,) for s in sentences])
    conn.commit()
    conn.close()

def add_user_sentence(sentence):
    """Appends a new sentence to the user corpus."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('INSERT INTO user_corpus (sentence) VALUES (?)', (sentence,))
    conn.commit()
    conn.close()

def hash_password(password):
    """Returns SHA-256 hash of a password string."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username, password):
    """Registers a new user. Returns True on success, False if username exists."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False # Username already exists
    conn.close()
    return success

def verify_user(username, password):
    """Verifies a user's credentials."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    
    if row is None:
        return False
    return row[0] == hash_password(password)
