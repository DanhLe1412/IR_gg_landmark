import cv2
import numpy as np
import streamlit as st
from utils import make_query, Embedded
import os

model = Embedded()
uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    cv2.imwrite("tmp.jpg", opencv_image)
    
    image_path = "tmp.jpg"
    query = model.get_vector(image_path)
    query = np.array([query])
    D,I = make_query(query)

    db = sorted(os.listdir('datasets/database/'))
    
    for i,img_i in enumerate(I[0]):
        image_path = db[img_i]
        image_path = os.path.join('datasets/database', image_path)
        tmp = cv2.imread(image_path)

        # Now do something with the image! For example, let's display it:
        st.write(D[0][i])
        st.image(tmp, channels="BGR")
    os.remove(image_path)
