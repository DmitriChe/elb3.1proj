import torch
from models.model_kudinov import Model
from models.kudinov_preprocessing import preprocess
import streamlit as st
from torchvision import io
from PIL import Image
from torchvision import transforms as T

TRESHOLD = 0.02257668599486351
idx2class = {0: 'Доброкачественная', 1: 'Злокачественная'}


def round_by_threshold(value):
    return 1 if value >= TRESHOLD else 0


@st.cache_resource()
def load_model():
    model = Model()
    model.load_state_dict(torch.load('models/model_kudinov.pt', map_location=torch.device('cpu')))
    return model

model = load_model()

def predict(img):
    img = preprocess(img)
    pred = model(img)
    model.eval()
    return pred

st.title('Модель по определению опухолей на коже')
st.caption('От Серёжи')
st.divider()

uploaded_image = st.file_uploader("Кидай свою опухоль")

if uploaded_image:
    img = Image.open(uploaded_image)
    pred_prob = predict(img).item()
    pred_class = round_by_threshold(pred_prob)

    st.write(f'Опухоль {idx2class[pred_class]}, p: {pred_prob}')
    st.image(uploaded_image, caption='Загруженное изображение', use_column_width=True)