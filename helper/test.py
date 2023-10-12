
import streamlit as st
from app import generate_code, get_started, generate_chat_code
from database import (create_database, create_user, read_user, update_user, delete_user,
                      create_message, read_message, update_message, delete_message, get_all_messages, delete_all_messages)

st.markdown("<link rel='stylesheet' href='style.css'>", unsafe_allow_html=True)
st.title("My own Chat-Gpt")

st.sidebar.header("ChatBot setting")
model_option = ["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002", "code-davinci-002"]
model = st.sidebar.selectbox("choose a model", model_option)


if model == "gpt-4":
    max = 8000
else: 
    max = 4000
max_token = st.sidebar.number_input("Max token", min_value = 1000, max_value = max, step = 500)
temp = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 2.0, value = 0.3, step = 0.1 )
st.sidebar.markdown("---")

st.sidebar.header("Chat History")
if st.sidebar.button("Clear History"):
    delete_all_messages()
    st.experimental_rerun() 

content = ""
messages_data = get_all_messages()
if messages_data:
    for message_data in messages_data:
        title = message_data["title"]
        if st.sidebar.button(title):
            content = message_data["content"]
else:
    st.sidebar.write("no Chat History")

label = "Entrez votre demande ici!"
prompt = st.text_area(label)
content += f"\n\nUser : {prompt}"


if st.button("Answer me!"):
    if model == "text-davinci-003" or model == "text-davinci-002" or model == "code-davinci-002":
        response = generate_code(model, max_token, temp, prompt)

    elif model == "gpt-3.5-turbo":
        response = generate_chat_code(model, max_token, temp, prompt)

    content += f"\n\nChatBot : {response}"
    st.write(content)


else:
    response = get_started()
    st.write(f"ChatBot : {response}")

title = st.text_input("Entrez un titre")
if st.button("Save"):
    create_message(title, content)
    st.experimental_rerun()


