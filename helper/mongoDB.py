from pymongo import MongoClient

# Initialiser la connexion à MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['gpt_database']



# Operations for User

def create_user(nom, prenom, pseudo, mot2pass):
    user = {"_id": pseudo, "nom": nom, "prenom": prenom, "pseudo": pseudo, "mot2pass": mot2pass}
    db.users.insert_one(user)

def read_user(pseudo):
    user = db.users.find_one({"_id": pseudo})
    return user

def read_all_users():
    users = list(db.users.find({}))
    return users

def update_user(pseudo, nom, prenom, mot2pass):
    db.users.update_one(
        {"_id": pseudo},
        {"$set": {"nom": nom, "prenom": prenom, "mot2pass": mot2pass}}
    )

def delete_user(pseudo):
    db.users.delete_one({"_id": pseudo})

def delete_all_users():
    db.users.delete_many({})

# Operations for Message

def create_message(title, content, pseudo):
    message = {"title": title, "content": content, "pseudo": pseudo}
    db.messages.insert_one(message)

def get_all_messages(pseudo):
    messages = list(db.messages.find({"pseudo": pseudo}))
    return messages

def read_message(id_message):
    message = db.messages.find_one({"_id": id_message})
    return message

def update_message(id_message, title, content):
    db.messages.update_one(
        {"_id": id_message},
        {"$set": {"title": title, "content": content}}
    )

def delete_message(id_message):
    db.messages.delete_one({"_id": id_message})

def delete_all_messages(pseudo):
    db.messages.delete_many({"pseudo": pseudo})

