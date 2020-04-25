import np
import image
from keras.applications.vgg16 import preprocess_input
from utils.util import *


def simple_inference(model,
                    train_set_vectors,
                    uploaded_image_path,
                    image_size,
                    distance='cosine'):
    '''
    Doing simple inference for single uploaded image.
    
    :param model: CNN model:
    :param session: tf.session, restored session
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: stringm path to uploaded image
    :param image_size: tuple, single image(height, width)
    :param distance: string, type of distance to be used,
                            this parameter is used to choose a way how to prepare vectors
    '''
    
    imagePath=uploaded_image_path
    img = image.load_img(imagePath, target_size=(image_size, image_size))
    img = image.img_to_array(img)

    # preprocess the image by (1) expanding the dimensions and
    # (2) subtracting the mean RGB pixel intensity from the
    # ImageNet dataset
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    features = model.predict([img])
    features_flatten = features.reshape(3*3*512)
    
    closest_ids = None
    
    if distance == 'hamming':
        train_set_vectors = np.where(train_set_vectors < 0.5,0,1)
        closest_ids = hamming_distance(train_set_vectors, features_flatten)
        
    elif distance == 'cosine':
        closest_ids = cosine_distance(train_set_vectors, features_flatten)
    
    return closest_ids
