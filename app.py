import config as cfg
import os
import numpy as np

from flask import Flask, request, render_template, send_from_directory
from utils.util import simple_inference
import pickle
#run 
# utils/dataset.py
# config.py

#Extract and store feature vectors
#create_features(val_data[:5000], model, BATCH_SIZE, IMAGE_SIZE[0], 'hamming')

#set root directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))



#==================================
#define model
#model = ImageRetrievalModel(cfg.IMAGE_SIZE)
model = None
#==================================

#get training set vectors
with open('dataset/modified_new/catalog/hamming_train_vectors.pickle', 'rb') as f:
	train_vectors = pickle.load(f)

#Load training set paths
with open('dataset/modified_new/catalog/val_paths.pickle', 'rb') as f:
	train_images_paths = pickle.load(f)


#Define Flask app
app = Flask(__name__, static_url_path='/static')

#Define apps home page
@app.route("/") 
def index():
	return render_template("index.html")

#Define upload function
@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = os.path.join(APP_ROOT, "uploads/")
    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)
    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = "/".join([upload_dir, img_name])
        img.save(destination)
	#inference fetches 20 images
    result = np.array(train_images_paths)[simple_inference(model, train_vectors, os.path.join(upload_dir, img_name), cfg.IMAGE_SIZE[0])]
    #get original result path from symlink
    result = [os.readlink(sym_path[3:]) for sym_path in result] 
    
    result_final = []
    for img in result:
        result_final.append("Img/img/"+ ('/').join(img.split("/")[-2:])) 
    return render_template("result.html", image_name=img_name, result_paths=result_final)#display only 12 images in user interface

#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
	return send_from_directory("uploads", filename)

#Start the application
if __name__ == "__main__":
	app.run(port=8787, debug=True)