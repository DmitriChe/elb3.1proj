import torch
from models.model_kudinov import Model
from models.kudinov_preprocessing import preprocess
import streamlit as st
import requests
from io import BytesIO
from torchvision import io
from PIL import Image
from torchvision import transforms as T
import zipfile
import time

TRESHOLD = 0.022721094478853047
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


def process_image(img):
    pred_prob = predict(img).item()
    pred_class = round_by_threshold(pred_prob)
    return pred_prob, pred_class


def display_progress_and_time(image_processing_func, img, description):
    progress_bar = st.progress(0)
    start_time = time.time()

    pred_prob, pred_class = image_processing_func(img)

    progress_bar.progress(100)  # Обновление прогресса на 100%
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Время выполнения: {execution_time:.2f} секунд")
    st.write(f'Опухоль {idx2class[pred_class]}, p: {pred_prob}')
    st.image(img, caption=description, use_column_width=True)
    return pred_prob, pred_class


st.title('Модель по определению опухолей на коже')
st.caption('От Серёжи')
st.divider()

uploaded_image = st.file_uploader("Кидай фото своей опухоли")
link = st.text_input('Кидай ссылку на свою опухоль')

if uploaded_image:
    if zipfile.is_zipfile(uploaded_image):
        with zipfile.ZipFile(uploaded_image, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith('__MACOSX/') or file.startswith('._'):
                    continue
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    with zip_ref.open(file) as image_file:
                        img = Image.open(image_file)
                        display_progress_and_time(process_image, img, f'Изображение из ZIP: {file}')
    else:
        img = Image.open(uploaded_image)
        display_progress_and_time(process_image, img, 'Загруженное изображение')

if link:
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    display_progress_and_time(process_image, img, 'Изображение по ссылке')
