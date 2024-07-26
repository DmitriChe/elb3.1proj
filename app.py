import streamlit as st

st.title('Супер классные нейросетки')
st.caption('От Димы, Толубая и Серёжи')
st.divider()

col1, col2, col3 = st.columns(3)

with col3:
    st.page_link('pages/kudinov_model.py', label='Модель Серёжи', icon='👾')