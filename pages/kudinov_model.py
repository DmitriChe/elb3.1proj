import torch
from models.model_kudinov import Model
import streamlit as st
from torchvision import io
from PIL import Image
from torchvision import transforms as T

TRESHOLD = 0.02257668599486351
idx2class = {0: 'Доброкачественная', 1: 'Злокачественная'}


def round_by_threshold(value):
    return 1 if value >= TRESHOLD else 0


model = Model()
model.load_state_dict(torch.load('models/model_kudinov.pt', map_location=torch.device('cpu')))

st.title('Модель по определению опухолей на коже')
st.caption('От Серёжи')
st.divider()

uploaded_image = st.file_uploader("Кидай свою опухоль")

if uploaded_image is not None:

    img = Image.open(uploaded_image).convert('RGB')

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    resize = T.Resize((224, 224))
    img = transform(img)
    img = resize(img / 255)
    with torch.inference_mode():
        pred_prob = model(img.unsqueeze(0)).item()
        pred_class = round_by_threshold(pred_prob)

    st.write(f'Опухоль {idx2class[pred_class]}, p: {pred_prob:.2f}')
    st.image(uploaded_image, caption='Загруженное изображение', use_column_width=True)