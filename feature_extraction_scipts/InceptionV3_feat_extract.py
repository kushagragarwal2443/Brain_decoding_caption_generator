import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
from tqdm import tqdm

# Download caption annotation files

for pt in ["val"]:
  if pt == "train":
    annotation_file = './images/annotations/captions_train2014.json'
    image_folder = './images/train2014/'
    PATH = image_folder
    layerst = [-1]
  else:
    annotation_file = './images/annotations/captions_val2014.json'
    image_folder = './images/val2014/'
    PATH = image_folder
    layerst = [248, 279, 310, -1]
  
  with open(annotation_file, 'r') as f:
      annotations = json.load(f)
      
  # Group all captions together having the same image ID.
  image_path_to_caption = collections.defaultdict(list)
  for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)

  image_paths = list(image_path_to_caption.keys())
  random.shuffle(image_paths)  
  
  train_captions = []
  img_name_vector = []

  for image_path in image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

  def load_image(image_path):
      img = tf.io.read_file(image_path)
      img = tf.io.decode_jpeg(img, channels=3)
      img = tf.keras.layers.Resizing(299, 299)(img)
      img = tf.keras.applications.inception_v3.preprocess_input(img)
      return img, image_path

  image_model = tf.keras.applications.InceptionV3(include_top=True)
  new_input = image_model.input
  
  # Get unique images
  encode_train = sorted(set(img_name_vector))

  # Feel free to change batch_size according to your system configuration
  image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
  image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32)

  for layern in layerst:
    hidden_layer = image_model.layers[layern].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    
    for img, path in tqdm(image_dataset):
      batch_features = image_features_extract_model(img)
      # batch_features = tf.reshape(batch_features,
                                  # (batch_features.shape[0], -1, batch_features.shape[3]))

      for bf, p in zip(batch_features, path):
        x = p.numpy().decode("utf-8")
        # print("p:", p)
        # print(os.getcwd())
        # print("p0:", p[0])
        x = x.split("/")
        x[1] = 'image_feature/InceptionV3'
        x[2] = str(pt) + "/" + str(layern)
        x = "/".join(x)
        np.save(x, bf.numpy())
