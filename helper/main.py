from chatPage import chat_Page
from connectionPage import connection_Page
from mongoDB import read_user
import streamlit as st

# Initialisation du session_state
if "pseudo" not in st.session_state:
    st.session_state["pseudo"] = ""
if "mot2pass" not in st.session_state:
    st.session_state["mot2pass"] = ""
if "logged_in" not in st.session_state:  # New session_state variable to keep track of login status
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    chat_Page()
else:
    connection_Page()
