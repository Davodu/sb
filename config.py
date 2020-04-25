import os
from utils.dataset import get_classes_image_dict

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

EPOCHS = 10

IMAGE_SIZE = [100, 100]

BATCH_SIZE = 128