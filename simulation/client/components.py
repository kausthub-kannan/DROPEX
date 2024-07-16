import streamlit as st

from utils import fetch_data


def search_bar(db):
    """ "
    :param db: firebase_admin.db.Reference

    :return data: dict - Latest entry from the database
    """
    st.markdown(
        """
    <style>
        .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color: #31333F;
        padding-left: 45px;
        padding-right: 20px;
        border-radius: 4px;
        border: 2px solid #f0f2f6;
        font-size: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4e54c8;
        box-shadow: 0 0 0 2px rgba(78,84,200,0.2);
    }
    .stTextInput > div > div::before {
        content: "🔍";
        font-size: 18px;
        position: absolute;
        left: 15px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([10, 3])

    with col1:
        search_term = st.text_input(
            "Search",
            placeholder="Enter your search term...",
            label_visibility="collapsed",
            key="search_input",
        )

    with col2:
        if st.button("Search"):
            data = fetch_data(db, search_term)
            return data
