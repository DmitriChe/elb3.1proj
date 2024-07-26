from torchvision import transforms as T

preprocessing_func = T.Compose(
    [
    T.Resize((224, 224)),
    T.ToTensor(),
    ]
)

def preprocess(img):
    img = preprocessing_func(img)
    img = img.unsqueeze(0)
    return img
