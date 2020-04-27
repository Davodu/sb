import numpy as np
from scipy.spatial.distance import hamming, cosine
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

import config as cfg 

def hamming_distance(training_set_vectors, query_vector, top_n=20):
    '''
    Calculates hamming distance between query image and all training set images
    
    :param training_set_vectors: numpy matrix
    :param query_vector: numpy vector
    :param top_n: integer
    '''
    distances = []
    
    for training_vector in training_set_vectors:
        distances.append(hamming(training_vector, query_vector))
    return np.argsort(distances)[:top_n]

def cosine_distance(training_set_vectors, query_vector, top_n=20):
    '''
    Calculates cosine distance between query image and all training set images
    
    :param training_set_vectors: numpy matrix
    :param query_vector: numpy vector
    :param top_n: integer
    '''
    
    distances = []
    
    for training_vector in training_set_vectors:
        distances.append(cosine(training_vector, query_vector[0]))
    return np.argsort(distances)[:top_n]

def simple_inference(model,
                    train_set_vectors,
                    uploaded_image_path,
                    image_size,
                    distance='hamming'):
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
    
    model = VGG16(input_shape=cfg.IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in model.layers:
        layer.trainable = False
    
    #add more layers in future
    
    #x = Flatten()(vgg.output)
    #x = Dense(4096, activation='relu', )(x)
    #x = Dropout(0.5)(x)
    #prediction = Dense(6, activation='softmax')(x)
    
    # create a model object
    #model = Model(inputs=vgg.input, outputs=prediction)
    
    # specify cost and optimization method to use
        
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
        )
    model._make_predict_function()
    
    
    features = model.predict([img])
    features_flatten = features.reshape(3*3*512)
    
    closest_ids = None
    
    if distance == 'hamming':
        train_set_vectors = np.where(train_set_vectors < 0.5,0,1)
        closest_ids = hamming_distance(train_set_vectors, features_flatten)
        
    elif distance == 'cosine':
        closest_ids = cosine_distance(train_set_vectors, features_flatten)
    
    return closest_ids
