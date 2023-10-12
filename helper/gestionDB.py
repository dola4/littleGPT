import streamlit as st
from pymongo import MongoClient
import pandas as pd
from mongoDB import (create_user, update_user, read_user, read_all_users, delete_user, delete_all_users, create_message, read_message, 
                    get_all_messages, update_message, delete_all_messages, delete_message)

# Page d'accueil
st.title('Gestion de la base de données')

# afficher les users
st.title("all Users")
st.write(read_all_users())
if st.button("delete a user"):
    id_user = st.text_input("select a pseudo!")
    delete_user(id_user)
        
st.title("see all message from a user")
user = st.text_input("user name")
st.write(get_all_messages(user))
id_message = st.text_input("delete a message by id")
if st.button("delete a massage!"):
    delete_message(id_message)

if st.button("delete all messages!"):
    delete_all_messages()
        
