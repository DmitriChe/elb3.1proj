import streamlit as st

st.title('История обучения')
st.caption('Серёжиной модельки')
st.divider()

st.subheader('Глава 0. Датасет')
st.write('2 класса изображений. Доброкачественные и злокачественные опухоли.')
st.write('Тренировочный датасет: 1441 доброкачественная, 1198 злокачественных')
st.write('Валидация: 361 доброкачественная, 301 злокачественная')
st.divider()

st.subheader('Глава 1. Тренировка')
st.caption('ResNet 152')
st.image('images/loss.png')
st.image('images/acc.png')
st.divider()

st.subheader('Глава 2. Изучение метрик')
st.image('images/rocauc.png')
st.image('images/conf_old.png')
col = st.columns(4)

with col[0]:
    st.metric('Accuracy', 0.912)
with col[1]:
    st.metric('Recall', 0.889)
with col[2]:
    st.metric('Precision', 0.917)
with col[3]:
    st.metric('ROCAUC', 0.962)

st.divider()
st.subheader('Глава 3. Борьба за recall')
col = st.columns(3)
with col[0]:
    st.metric('Beta', 3)
with col[1]:
    st.metric('Threshold', 0.022)
with col[2]:
    st.metric('Maximum F-beta', 0.95)

st.image('images/conf.png')

col = st.columns(4)
with col[0]:
    st.metric('Accuracy', 0.824)
with col[1]:
    st.metric('Recall', 0.98)
with col[2]:
    st.metric('Precision', 0.72)
with col[3]:
    st.metric('ROCAUC', 0.962)

st.divider()
image_url = 'https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2RncGl3aDU0dG9vcjB3MTV4bzUzN3djMGhtczAyams2Mmc3aHZkMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xUPOqo6E1XvWXwlCyQ/giphy.webp'
html_code = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_url}" alt="centered image" />
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)



