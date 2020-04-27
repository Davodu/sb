from keras.applications.vgg16 import VGG16


class ImageRetrievalModel(object):

    def __init__(self, image_size):
        model = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)

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