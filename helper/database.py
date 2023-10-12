import sqlite3

def create_database():
    conn = sqlite3.connect('gpt.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id_user INTEGER PRIMARY KEY,
            nom TEXT,
            prenom TEXT,
            pseudo TEXT,
            mot2pass TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id_message INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            id_user INTEGER,
            FOREIGN KEY (id_user) REFERENCES users(id_user)
        )
    ''')

    conn.commit()
    conn.close()

def get_db_connection():
    create_database()
    conn = sqlite3.connect('gpt.db')
    return conn

# Operations for User

def create_user(nom, prenom, pseudo, mot2pass):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (nom, prenom, pseudo, mot2pass) VALUES (?, ?, ?, ?)',
              (nom, prenom, pseudo, mot2pass))
    conn.commit()
    conn.close()

def read_user(pseudo):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE pseudo = ?', (pseudo,))
    user = c.fetchone()
    conn.close()
    return user

def update_user(id_user, nom, prenom, pseudo, mot2pass):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE users SET nom = ?, prenom = ?, pseudo = ?, mot2pass = ? WHERE id_user = ?',
              (nom, prenom, pseudo, mot2pass, id_user))
    conn.commit()
    conn.close()

def delete_user(id_user):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE id_user = ?', (id_user,))
    conn.commit()
    conn.close()

# Operations for Message

def create_message(title, content, id_user):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO messages (title, content, id_user) VALUES (?, ?, ?)',
              (title, content, id_user))
    conn.commit()
    conn.close()

def get_all_messages():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM messages')
    messages = c.fetchall()
    conn.close()
    return [dict(message) for message in messages] 

def read_message(id_message):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM messages WHERE id_message = ?', (id_message,))
    message = c.fetchone()
    conn.close()
    return message

def update_message(id_message, title, content):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE messages SET title = ?, content = ? WHERE id_message = ?',
              (title, content, id_message))
    conn.commit()
    conn.close()

def delete_message(id_message):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE id_message = ?', (id_message,))
    conn.commit()
    conn.close()

def delete_all_messages():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('DELETE FROM messages')
    conn.commit()
    conn.close()
