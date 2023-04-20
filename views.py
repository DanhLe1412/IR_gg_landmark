import cv2
import numpy as np
from utils.util import make_query, Embedded
from utils.gradcam import grad_cam_image
import streamlit as st
import os

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2



model = Embedded()
uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png"])
button = st.button("search :cat:")
if uploaded_file is not None:
        # Convert the file to an opencv image.
        st.write("# query image")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        cv2.imwrite("tmp.jpg", opencv_image)
        image_path = "tmp.jpg"

        st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        st.write("# return's value")

if button:
    query = model.get_vector(image_path)
    query = np.array([query])
    table = make_query(query)

    db = sorted(os.listdir('datasets/database/'))
    col1, col2 = st.columns(2)

    table = np.array(table)
    table[:, 1] = (table[:, 1] - np.min(table[:,1])) / (np.max(table[:,1])-np.min(table[:,1]))

    for i, (index, score) in enumerate(table):
        if index<0 or index > len(db):
            continue
        image_path = db[int(index)]
        image_path = os.path.join('datasets/database', image_path)
        tmp = cv2.imread(image_path)
        tmp = cv2.resize(tmp,(224,224))
        h,w,_ = tmp.shape
        pos = (int(w*0.1), int(h*0.1))
        cam_image = grad_cam_image(model.base_model, image_path)
        cam_image = cv2.resize(cam_image, (224,224))

        # Now do something with the image! For example, let's display it:
        cam_image = cv2.putText(cam_image,f'conf: {score:.2f}', 
                                pos, 
                                font, 
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
        col1.image(tmp, channels="BGR")
        col2.image(cam_image, channels="BGR")

                                                                                                                                             
    os.remove(image_path)
