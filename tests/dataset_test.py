from config import *
import np
import image

#check that random images are loaded
plt.imshow(image.img_to_array(image.load_img(np.random.choice(CLASSES_IMAGE_DICT['Tank']))).astype('uint8'))
plt.show()