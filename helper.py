import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('Classifier')

def verify_image(image):
    img_width, img_height = 150, 150
	
	img = keras.preprocessing.image.load_img(image)
	img = tf.image.central_crop(img, central_fraction=0.5)
	img = tf.image.resize(img,[img_width, img_height])
	img = keras.utils.img_to_array(img)
	img = np.expand_dims(img, axis = 0)

	pred=model.predict(img)

    return pred