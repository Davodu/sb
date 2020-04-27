from dataset import *
import np
import image

#check that random images are loaded
img = image.img_to_array(image.load_img(np.random.choice(CLASSES_IMAGE_DICT['Tank']))).astype('uint8')
assert img.shape[2]==3

