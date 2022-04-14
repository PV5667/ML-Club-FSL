import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18
import torchvision.transforms as T


st.set_page_config(
    page_title="FSL Demo",
    page_icon=":smiley:",
    layout="wide"
 )


transforms = T.Compose([T.Resize((400)), T.CenterCrop(400), T.ToTensor()])

#Load in the backbone and the weights

backbone = resnet18(pretrained=True)
backbone.fc = nn.Flatten()

#Load in the prototype vectors

classes = {
    0: "Anish!!!!!!",
    1: "Patrick!!!!",
    2: "Marc!!!!!!!"
}

def output_prediction(image):
    prototypes = torch.load("prototypes.pt")
    query_vector = backbone(image.view(1, 3, 400, 400))
    scores = torch.cdist(query_vector, prototypes) * -1
    _, pred = F.softmax(scores).max(1)
    return pred.item()

image = st.camera_input("Take your image here")


if image is not None:
    image = Image.open(image)
    print(image.size)
    image = transforms(image)
    st.write(image.shape)
    result = output_prediction(image)
    st.write(classes[result])






