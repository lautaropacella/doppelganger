import streamlit as st
import numpy as np
import cv2
import cvlib as cv
import torch
from jina import Document
from docarray import DocumentArray
from PIL import Image
from functions import crop_face

st.set_page_config(page_title="Deep Doppelgangers", layout="wide", page_icon="🦊")

## Constants
MODEL_URI = "model.pth"
EMBEDDINGS_URI = "embeddings.lz4"
IMAGE_FORMATS = ["jpg", "jpeg", "png"]

## Main Functions
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load(MODEL_URI, map_location=torch.device("cpu"))
    return model


@st.cache(allow_output_mutation=True)
def load_embeddings():
    embeddings = DocumentArray.load_binary(EMBEDDINGS_URI)
    return embeddings


## Web App Format ##
st.markdown(
    "<h1 style='text-align: center; color: white;'>👬 Deep Doppelgängers 🧑‍🤝‍🧑</h1>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

"""
[![Star](https://img.shields.io/github/stars/lautaropacella/doppelganger.svg?logo=github&style=social)](https://github.com/lautaropacella/doppelganger)
&nbsp[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lautaro-pacella/)
"""
st.markdown("<br>", unsafe_allow_html=True)

"""
##### **¿Puede una máquina encontrar similitudes como los humanos?** \n
- Para empezar, simplemente cargue una foto con su rostro. Mientras mejor sea la calidad, más efectivo será el resultado.
- Los parecidos van a variar dependiendo de diferentes factores de la imagen que uses, por ejemplo, la luminosidad, el ángulo, si usas anteojos, etc.
- Se mostrarán las 5 celebridades más parecidas, en el órden de similtud respectivo.
---
"""
with st.expander("Más Info"):
    """
    - Para este proyecto, se entrenó un modelo de redes neuronales convolucionales profundas, más especificamente, un modelo de redes residuales (ResNet) a partir del [Totally-Looks-Like Dataset](https://sites.google.com/view/totally-looks-like-dataset). \n
    - Las similitudes se buscan en base a un recorte del [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) de 3000 celebridades.
    """

with st.spinner("Cargando Modelos..."):
    model = load_model()
with st.spinner("Cargando caras de celebridades..."):
    embeddings = load_embeddings()

img_buffer = st.file_uploader(
    "Cargue su retrato aquí", accept_multiple_files=False, type=IMAGE_FORMATS
)
if img_buffer:
    with st.spinner("Detectando rostro..."):
        ## Read Image
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        ## Detect Face
        face, confidence = cv.detect_face(img)
        ## Crop Face & get features
        for left, top, right, bottom in face:
            face_cropped = crop_face(img, left, top, right, bottom)
        ## Output
        original, middle, cropped = st.columns(3)

        middle.markdown("### Rostro detectado")
        middle.image(face_cropped, channels="BGR", use_column_width=True)

        """---"""

    with st.spinner("Buscando parecidos..."):
        ## Format for Jina
        face_doc = Document(blob=face_cropped)
        face_doc.set_image_blob_shape(
            shape=(224, 224)
        ).set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)
        face_doc.embed(model, device="cpu")
        face_doc.match(embeddings)

        uris = []
        for index, matched in enumerate(face_doc.matches[0:5]):
            uri = matched.uri.split("/")[-1]
            uris.append(uri)
        
        matches = st.columns(5)
        for i in range(len(matches)):
            image = Image.open("celeba/" + uris[i])
            matches[i].image(image, caption=f"""Doppelganger #{i+1}""")
