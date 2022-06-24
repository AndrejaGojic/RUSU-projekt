from keras.models import load_model
from matplotlib import image
import numpy as np

import streamlit as st
from PIL import Image, ImageOps

def importAndPredict(image_data, choosed_model):    
    if choosed_model == 'ResNet 50':
        model = load_model('resnet50_model/')
        img = image_data.resize((96, 96))
        img = img.convert('RGB')
        img = ImageOps.grayscale(img)
        img = np.array(img).reshape(-1, 96, 96, 1)
        img= img/255.0
        pred = model.predict(img)
    elif choosed_model == 'VGG 16':
        model = load_model('vgg_model/')
        img = image_data.resize((96, 96))
        img = img.convert('RGB')
        img = np.array(img).reshape(-1, 96, 96, 3)
        img= img/255.0
        pred = model.predict(img)
    elif choosed_model=='AlexNet':
        model = load_model('alexnet_model/')
        img = image_data.resize((96, 96))
        img = img.convert('RGB')
        img = ImageOps.grayscale(img)
        img = np.array(img).reshape(-1, 96, 96, 1)
        img= img/255.0
        pred = model.predict(img)
    return pred
        

model_names = ['ResNet 50', 'AlexNet', 'VGG 16']
st.write("""
         # Klasifikcija spola na temelju otiska prsta
        """)
        
choosed_model = st.radio('Odaberite model za predikciju:', model_names)
file = st.file_uploader("Učetajte sliku otiska prsta", type=["jpg", "png", "bmp"])



if file is None:
    st.text("Molim Vas da učitate sliku")
else:
    image = Image.open(file)
    st.image(image, width=200)
    result = st.button("Predikcija")
    if result:
        pred = importAndPredict(image, choosed_model)
        index = np.argmax(pred, axis=1)
        if index == 1:
            percent = np.max(pred)*100
            string = 'Ovaj otisak prsta '+ str(round(percent, 2)) + '% pripada ženskoj osobi'
        else: 
            percent = np.max(pred)*100
            string = 'Ovaj otisak prista '+ str(round(percent, 2)) + '% pripada muškoj osobi'
        st.success(string)
        
