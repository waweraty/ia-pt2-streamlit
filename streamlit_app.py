import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
from helper import *
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



if uploaded_file is not None:
	#src_image = load_image(uploaded_file)
	upimage = Image.open(uploaded_file)	

	st.image(uploaded_file, caption='Input Image', use_column_width=True)

	pred=verify_image(uploaded_file)

	#st.write(os.listdir())
	if st.button('Go'): 
		st.write('Image Loaded')
		st.write(str(pred[0][0].round()))
