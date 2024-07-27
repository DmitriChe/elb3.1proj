# pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu

import streamlit as st
import torch
from torchvision import transforms as T

# Для чтения изображений с диска
from torchvision import io # input/output
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
from PIL import Image
from models.model_dmche import MyResNet50

# Подготовка модели для классификации изобрежений
model = MyResNet50()
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, '../models/weights_dmche.pt')
# Загружаем веса модели
model.load_state_dict(torch.load(weights_path, weights_only=False))
# Словарь для декодирования названий классов
idx2class = {
    0: 'дома (buildings)',
    1: 'лес (forest)',
    2: 'ледник (glacier)',
    3: 'горы (mountain)',
    4: 'море (sea)',
    5: 'улица (street)'
    }

st.title("Распознавание датасета Intel Image Classification на дообученной ResNet50")
# st.subtitle("скачать картинки с Kaggle")


# Загрузка изображения через Streamlit
uploaded_file = st.file_uploader("Загрузите изображение для классификации:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Преобразование загруженного изображения в объект PIL
    img = Image.open(uploaded_file).convert('RGB')

    # Преобразования изображения с использованием torchvision
    transform = T.Compose([
        T.Resize((150, 150)),  # Измените размер в соответствии с требованиями вашей модели
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img)

    torch.permute(img_tensor, (1, 2, 0))
    model.eval()
    with torch.inference_mode():
        preds = model(img_tensor.unsqueeze(0))
        pred_class = torch.argmax(preds, dim=1).item()
        class_name = idx2class[pred_class]


    # Отображение картинки в Streamlit
    st.text(class_name)
    st.image(img, caption=class_name, use_column_width=True)

# Распознавание картинке по url

img_url = st.text_input("Введите url к картинке для распознавания")
# https://storage.googleapis.com/kagglesdsdata/datasets/111880/269359/seg_pred/seg_pred/1003.jpg
# https://img.freepik.com/free-photo/morskie-oko-in-tatry_1204-510.jpg
st.write('Пример: https://img.freepik.com/free-photo/morskie-oko-in-tatry_1204-510.jpg')

if img_url:
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # Преобразования изображения с использованием torchvision
    transform = T.Compose([
        T.Resize((150, 150)),  # Измените размер в соответствии с требованиями вашей модели
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img)

    torch.permute(img_tensor, (1, 2, 0))
    model.eval()
    with torch.inference_mode():
        preds = model(img_tensor.unsqueeze(0))
        pred_class = torch.argmax(preds, dim=1).item()
        class_name = idx2class[pred_class]


    # Отображение картинки в Streamlit
    st.text(class_name)
    st.image(img, caption=class_name, use_column_width=True)

else:
    st.stop()

