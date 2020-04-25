import image
import glob
import os, pickle
from tqdm.notebook import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

from keras.applications.vgg16 import preprocess_input

#local modules
import config as cfg
from utils import model

# constants
IMAGES_PATH = os.path.abspath('./dataset/Img/img')

CLASSES = ['Anorak',
 'Blouse',
 'Bomber',
 'Cardigan',
 'Jacket',
 'Parka',
 'Sweater',
 'Tank',
 'Tee']

NEW_CATEGORY_MAP = {'Anorak':'0','bomber':'0','Jacket':'0','Parka':'0', #Anorak, bomber, jacket,parka
                    'Cardigan':'1', #Cardigan
                    'Sweater':'2',#sweater
                    'Tank':'3',#Tank
                    'Blouse':'4',#blouse
                    'Tee':'5'#Tee
                   }

CLASSES_IMAGE_DICT = get_classes_image_dict(CLASSES, IMAGES_PATH)

# Dataset utilities
def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)

def get_bbox(bbox_file_path):
    rf = open(bbox_file_path).readlines()[2:]
    
    bboxes = {}
    for l in tqdm(range(len(rf))):
        info = rf[l].strip('\n').split(' ')
        img_path = '/home/daviesodu/sb-capstone/dataset/Img/'+info[0]
        bbox = info[-4:]
        bboxes[img_path]=[int(val) for val in bbox]
    return bboxes
    
def get_classes_image_dict(classes, images_path):
    class_images_dict = {}
    for c in classes:
        class_images_dict[c] = glob(images_path + '/*{0}/*'.format(c))
    return class_images_dict

def get_total_images(class_images_dict):
    total_images = []
    for c in class_images_dict.keys():
        cur_class_list = class_images_dict[c]
        temp_array = []
        for idx,img_path in enumerate(cur_class_list[:5000]):
            temp = cur_class_list[idx]
            temp = img_path + '+' + NEW_CATEGORY_MAP[c]
            temp_array.append(temp)
        total_images += temp_array

def create_symlinks(X):
    for img_path in X:
        img_class=img_path.split('/')[7].split('_')[-1]
        src = img_path
        dst = train_path_to + '/' + NEW_CATEGORY_MAP[img_class] + '/'+ ''.join(img_path.split('/')[7:])
        if os.path.exists(src):
            link(src, dst)
        else:
            os.unlink(dst)
            
def create_features(dataset, pre_model, batch_size,image_dim,distance='hamming'):
    x_scratch = []
 
    # loop over the images
    for imagePath in tqdm(dataset):
        imagePath=os.readlink(imagePath)
        img = image.load_img(imagePath, target_size=(image_dim, image_dim))
        img = image.img_to_array(img)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
 
        # add the image to the batch
        x_scratch.append(img)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=batch_size)
    features_flatten = features.reshape((features.shape[0], 3 * 3 * 512))
    
    if distance == 'hamming':
        features_flatten = np.where(features_flatten < 0.5, 0, 1)
        
        with open('./dataset/modified_new/catalog/hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(features_flatten, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif distance == 'cosine':
        with open('./dataset/modified_new/catalog/cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(features_flatten, f, protocol=pickle.HIGHEST_PROTOCOL)

#Create sub-directory for modified datset
mkdir('./dataset/modified_new')

train_path_to = os.path.abspath('./dataset/modified_new/Training')
valid_path_to = os.path.abspath('./dataset/modified_new/Validation')

mkdir(train_path_to)
mkdir(valid_path_to)

# load images to array and shuffle
total_images = get_total_images(CLASSES_IMAGE_DICT)
total_images = shuffle(total_images)

#train test split
X_t,y_t = zip(*total_images)
X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=42)


#create symbolic links for train and test dataset

'''
#utils to clear symlinks
arr = glob('./dataset/modified_new/Validation/*/*')
for a in arr:
    os.unlink(a)
print()

assert glob('./dataset/modified_new/Training/*/*')== []
'''
        
create_symlinks(X_train)
create_symlinks(X_test)

train_links = glob('./dataset/modified_new/Training/*/*')
val_links = glob('./dataset/modified_new/Validation/*/*')
assert(len(train_links)==len(X_train))
assert(len(val_links)==len(X_test))

#create feature vectors & save to appropriate folder.
val_data = glob('./dataset/modified_new/Validation/*/*')

#Extract and store feature vectors
create_features(val_data[:5000], model, cfg.BATCH_SIZE, cfg.IMAGE_SIZE[0], 'hamming')
