import faiss
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import cv2
import streamlit as st


class Embedded:
    def __init__(self) -> None:
        self.base_model = models.resnet18(pretrained=True)
        self.model = models.resnet18(pretrained=True)
        self.layer = self.model._modules.get('avgpool')
        self.model.eval()
        self.scaler= transforms.Resize((224,224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
    

    def get_vector(self, image_path):
        if type(image_path) != type(""):
            return np.zeros(512)
        print("image_path: ", image_path)
        img = Image.open(image_path).convert('RGB')
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        my_embedding = torch.zeros(512)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        h = self.layer.register_forward_hook(copy_data)
        self.model(t_img)
        h.remove()
        X = my_embedding.numpy().astype('float64')
        return X


def make_indexing():
    d = 512
    database_dir = "datasets/database"
    indexdir="indexdir"

    index = faiss.IndexFlatL2(d)   # build the index
    model = Embedded()
    imgs = sorted(os.listdir(database_dir))
    nb = []
    for img in imgs:
        img_path = os.path.join(database_dir,img)
        vec = model.get_vector(img_path)
        nb.append(vec)
    nb = np.array(nb)
    index.add(nb)

    os.makedirs(indexdir, exist_ok=True)
    faiss.write_index(index, os.path.join(indexdir,'landmark.index'))


def make_query(query):
    indexdir="indexdir"
    index_file="landmark.index"
    if not os.path.exists(os.path.join(indexdir,index_file)):
        raise Exception("VCL")
    index=faiss.read_index(os.path.join(indexdir,index_file))
    D,I = index.search(query,k=20)
    table = []
    for i, data in enumerate(I[0]):
        table.append([data, D[0][i]])
    table = sorted(table,key=lambda x: x[1], reverse=True)
    return table


#
# PSEUDO CODE
#
# import cv2
# db = sorted(os.listdir('datasets/database/'))

# image_path = "datasets/query/0a312acc97bda9b1.jpg"
# img = cv2.imread(image_path)
# cv2.imshow("query", img)
# model = Embedded()
# query = model.get_vector(image_path)
# query = np.array([query])
# D,I = make_query(query)
# # print(D)


# print(db[I[0][0]])
# img_name = db[I[0][0]]
# image_path = os.path.join('datasets/database', img_name)
# img = cv2.imread(image_path)
# cv2.imshow('retrieve', img)
# cv2.waitKey(0)