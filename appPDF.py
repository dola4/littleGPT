import streamlit as st
import openai

from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1100,
        chunk_overlap = 220,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorStore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key = 'chat_history', return_message = True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorStore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userInput(user_input):
    if st.session_state.conversation is None:
        st.error("Please upload your PDFs")
        return 

    # Constructing chat history in the expected format
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Directly append the user's message to chat_history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.write("Chat history before call:", st.session_state.chat_history)
    response = st.session_state.conversation({'question': user_input, 'chat_history': st.session_state.chat_history})

    message_key = 'message' if 'message' in response else 'answer' if 'answer' in response else None

    if message_key:
        # Append the assistant's response
        st.session_state.chat_history.append({"role": "assistant", "content": response[message_key]})
    else:
        st.error("No message or answer found in the response!")
        return
    
    user_input = user_input.replace('"', '\"')
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response_message = response[message_key].replace('"', '\"')
    st.session_state.chat_history.append({"role": "assistant", "content": response_message})


    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=msg["content"] + '_user')
        else:
            message(msg["content"], key=msg["content"])
            



def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with PDF', page_icon=":books:")

    


    user_question = st.text_input("Ask your question here")

    


    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF here", type=['pdf'], accept_multiple_files=True)
        if st.button("Process ..."):
            with st.spinner("Procesing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector strore
                vectorStore = get_vectorStore(text_chunks)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorStore)
            



if __name__ == "__main__":
    main()