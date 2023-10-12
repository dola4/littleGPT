def connection_Page() :

    import streamlit as st
    import re
    from mongoDB import create_user, read_user
    from execfile import execfile
    from chatPage import chat_Page
    #from stylecss import css
    # Choix du formulaire
    #st.markdown(css, unsafe_allow_html=True)
    choice = st.selectbox("Choisissez une action", ["Inscription", "Connexion"])

    if choice == "Inscription":
        with st.form("signup_form"):
            st.header("Inscription")
            prenom = st.text_input("Prénom")
            nom = st.text_input("Nom")
            pseudo = st.text_input("Pseudo")
            mot2pass = st.text_input("Mot de passe")
            st.write("doit contenir (Aa 1-0 et caracteres speciaux)")
            mot2passR = st.text_input("Confirmer le mot de passe")
        
            # Quand l'utilisateur soumet le formulaire d'inscription
            if st.form_submit_button("S'inscrire"):
                user = read_user(pseudo)
                if user:
                    st.write("Pseudo déjà utilisé !")
                else:
                    # Vérification des autres champs...
                    if mot2pass != mot2passR:
                        st.write("Les mots de passe ne correspondent pas !")
                    #elif len(mot2pass) < 8 or re.search('[^A-Za-z0-9]+', mot2pass) == False or any(c.isupper() == False for c in mot2pass) or any(c.islower() == False for c in mot2pass):
                        #st.write("Votre mot de passe doit respecter les règles de sécurité.")
                    else:
                        create_user(nom, prenom, pseudo, mot2pass)
                        st.write("Inscription réussie !")
                        chat_Page()


    elif choice == "Connexion":
        with st.form("login_form"):
            st.header("Connexion")
            pseudo = st.text_input("Pseudo")
            mot2pass = st.text_input("Mot de passe")

            # Quand l'utilisateur soumet le formulaire de connexion
            if st.form_submit_button("Se connecter"):
                user = read_user(pseudo)
                if user is None:
                    st.write("Utilisateur non trouvé.")
                elif user['mot2pass'] != mot2pass:
                    st.write("Mot de passe incorrect.")
                    st.write(user)
                else:
                    st.session_state["logged_in"] = True  # Set logged_in to True when login is successful
                    st.write("Connexion réussie.")
                    st.experimental_rerun()
