import sys, os
import streamlit as st
import streamlit_google_oauth as oauth

def user_login():
    if len(sys.argv)>1:
        if sys.argv[1]=="1":
            st.session_state["authentication_status"]=True
            user_id='syed'
            st.session_state["user_email"]='syed@datatecnica.com'
            st.write(f":green[**Welcome {st.session_state.user_email}**]")        
        else:
            st.warning(":red[**To Run Locally without authentication, Please re-run with 'streamlit run app.py 1'**]")
    else:

        login_info = oauth.login(
            client_id=os.environ["CLIENT_ID"], #client_id,
            client_secret=os.environ["CLIENT_SECRET"], #client_secret,
            redirect_uri=os.environ["REDIRECT_URI"],#redirect_uri,
            logout_button_text="**Logout**",
        )

        if login_info:
            user_id, st.session_state["user_email"] = login_info
            st.write(f"Welcome {st.session_state['user_email']}")
            st.session_state["authentication_status"]=True
        else:
            st.session_state["authentication_status"]=False
            st.warning(':red[**Something went wrong, User is not authorized to use the App, Please contact app Administrator to be added to the users list**]')
