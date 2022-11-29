import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os

st.header('GRIP Team')
st.header("Stage predictor using images")
st.write("Upload image to get its corresponding stage")

uploaded_file = st.file_uploader("Choose an image...")


def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

model = keras.models.load_model('Classifier')

if uploaded_file is not None:
	#src_image = load_image(uploaded_file)
	upimage = Image.open(uploaded_file)	

	st.image(uploaded_file, caption='Input Image', use_column_width=True)

	img_width, img_height = 150, 150
	
	img = keras.preprocessing.image.load_img(uploaded_file)
	img = tf.image.central_crop(img, central_fraction=0.5)
	img = tf.image.resize(img,[img_width, img_height])
	img = keras.utils.img_to_array(img)
	img = np.expand_dims(img, axis = 0)

	pred=model.predict(img)

	#st.write(os.listdir())
	if st.button('Go'): 
		st.write('Image Loaded')
		st.write(pred[0].round)
