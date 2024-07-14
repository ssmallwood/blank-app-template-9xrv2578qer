import streamlit as st

st.set_page_config(
    page_title="Hidden Gems",
    page_icon="💎",
)

st.write("# Hidden Gem Explorer 💎")

st.sidebar.success("Select a version above.")

st.markdown(
    """
    These tools explore data about colleges as starting points for inclusion in the hidden gems list.
    **👈 Select a version from the sidebar** to explore different approaches to filtering and highlighting colleges
    ### Basics of the datasets
    - Version 1 includes only the 900+ colleges in the Bain analysis.
    - Version 2 includes a total of 1,750 four year colleges. 


    ### Version 1
    - Initial pass at creating a simple formula looking for colleges that aren't super selective but perform well on certain outcome measures.

    ### Version 2
    - More thoughtful about using statistical approaches to build a weighted score that compares the colleges. 
    - Let's the user adjust those weights dynamically.
    - Scores all the colleges that have the Bain regression data.
    - Allows user to see the 'unscored' colleges in a given state as well, showing their admission and yield rates.
"""
)