def chat_Page() : 

    ########## import ##########
    import streamlit as st
    from app import generate_code, get_started, generate_chat_code
    from mongoDB import (create_user, read_user, update_user, delete_user,
                         create_message, read_message, update_message, delete_message, get_all_messages, delete_all_messages)
    from connectionPage import connection_Page
    #from stylecss import css
    ############################

    ########## initialization ##########
    #st.markdown(css, unsafe_allow_html=True)
    pseudo = st.session_state["pseudo"]
    st.title("My own Chat-Gpt")

    ########## changement was made here !!! NEED TO CHECK !!! ##########
    #if st.button("Disconnect"):
    #    st.session_state["logged_in"] = False
    #    connection_Page()
    ####################################################################

    your_name =st.text_input("Entrez votre nom")
    chatbot_name = st.text_input("choisir un nom pour votre chatbot")
    ####################################

    ########### choix des parametres ##########
    st.sidebar.header("ChatBot setting")
    model_option = ["gpt-3.5-turbo", "text-davinci-003", "text-davinci-002", "code-davinci-002"]
    model = st.sidebar.selectbox("choose a model", model_option)

    # define max token accepted based on choosen model
    if model == "gpt-4":
        max = 8000
    else: 
        max = 4000
    max_token = st.sidebar.number_input("Max token", min_value = 1000, max_value = max, step = 500)
    temp = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 2.0, value = 0.5, step = 0.1 )
    st.sidebar.markdown("---")
    ###########################################

    ########## chat history ##########
    st.sidebar.header("Chat History")
    if st.sidebar.button("Clear History"):
        delete_all_messages(pseudo)
        st.experimental_rerun() 

    content = ""
    messages_data = get_all_messages(pseudo)
    if messages_data:
        for message_data in messages_data:
            title = message_data["title"]
            if st.sidebar.button(title):
                content = message_data["content"]
                st.write(content)
    else:
        st.sidebar.write("no Chat History")
    
    ##################################

    ########## client prompt ##########

    # enter prompt
    label = "Entrez votre demande ici!"
    prompt = st.text_area(label)
    # Initialize the state with an empty string if it doesn't exist
    if 'content' not in st.session_state:
        st.session_state['content'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Append user's input to the content
    if your_name != "":
        st.session_state['content'] += f"\n\n{your_name} : {prompt}"
    else:
        st.session_state['content'] += f"\n\nUser : {prompt}"

    # model condition
    if st.button("Answer me!"):
        if model == "text-davinci-003" or model == "text-davinci-002" or model == "code-davinci-002":
            response = generate_code(model, max_token, temp, prompt)
        elif model == "gpt-3.5-turbo":
            response = generate_chat_code(model, max_token, temp, prompt, st.session_state['content'])

        if chatbot_name != "":
            st.session_state['content'] += f"\n\n{chatbot_name} : {response}"
        else:
            st.session_state['content'] += f"\n\nChatBot : {response}"
    
        # Append the user's message to session_state
        user_msg = {"title": "User", "content": prompt}
        st.session_state['messages'].append(user_msg)

        # Append the bot's response to session_state
        message_data = {"title": "Bot", "content": response}
        st.session_state['messages'].append(message_data)

        # Display all messages
        for message_data in st.session_state['messages']:
            st.write(f"{message_data['title']} : {message_data['content']}")
        
    # if the bouton is not click
    else:
        response = get_started()
        st.write(f"ChatBot : {response}")
    ###################################

    ########## save conversation ##########
    title = st.text_input("Entrez un titre")
    if st.button("Save"):
        create_message(title, content, pseudo)
        st.experimental_rerun()
        
    #######################################
