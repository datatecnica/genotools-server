import streamlit as st
import streamlit.components.v1 as components

from utils.hold_data import place_logos, config_page
from utils.config import AppConfig

config = AppConfig()
HOME_CONTENT = config.HOME_CONTENT



def render_home():
    config_page('Home')
    place_logos()

    st.markdown(HOME_CONTENT["TITLE"], unsafe_allow_html=True)

    sent1, sent2, sent3 = st.columns([1, 6, 1])
    with sent2:
        st.markdown(HOME_CONTENT["INTRO"], unsafe_allow_html=True)

    exp1, exp2, exp3 = st.columns([1, 2, 1])
    with exp2.expander("Full Description", expanded=False):
        st.markdown(HOME_CONTENT["FULL_DESCRIPTION"], unsafe_allow_html=True)

    components.html(HOME_CONTENT["JS_EXPANDER_STYLE"], height=0, width=0)

def main():
    render_home()

if __name__ == "__main__":
    main()