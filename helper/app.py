import openai
import os
from api import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_started():
    model_engine = "text-davinci-003"
    response = openai.Completion.create(
        model = model_engine,
        prompt = "say : hello boy!",
        max_tokens = 1000,
        n = 1,
        stop = None,
        temperature = 0
    )
    return response.choices[0].text.strip()

def generate_chat_code(model, max_token, temp, prompt, content):
    model_engine = model
    response = openai.ChatCompletion.create(
        model = model_engine,
        messages=[
        #{"role": "system", "content": "You are an outstanding assistant"},
        {"role": "system", "content": content},
        {"role": "user", "content": prompt},
        ],
    max_tokens = max_token,
    n = 1,
    stop = None,
    temperature = temp
    )
    return response.choices[0].message['content']


def generate_code(model, max_token, temp, prompt):
    model_engine = model
    response = openai.Completion.create(
        model = model_engine,
        prompt = prompt,
        max_tokens = max_token,
        n = 1,
        stop = None,
        temperature = temp
    )
    return response.choices[0].text.strip()

