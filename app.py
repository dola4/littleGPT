import openai
import streamlit as st
from streamlit_chat import message

openai.api_key = "YOUR_API_KEY"

def generate_response(user_input):
    completion = openai.Completion.create(
        engine = "text-davinci-003", 
        prompt = user_input,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.5,
    )
    message = completion.choices[0].text
    return message

st.title(':gear: Welcome to my own Chat-Gpt created with Streamlit')
if "generated" not in st.session_state:
    st.session_state['generated'] = []
if "past" not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You:", "Hello, how are you?", key='input')
    return input_text


user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state.generated:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

