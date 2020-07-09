import os
import re
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import sklearn.preprocessing
import sklearn.model_selection


# Getting classes for Woodpeckers and Pigeons
dir_name = os.path.dirname(__file__)
file_name = os.path.join(dir_name, "Cornell Data Dump/classes.txt")
classes_txt = open(file_name, 'r')
classes = []
for line in classes_txt:
    bird_class = re.sub('^[0-9]*', '', line)
    bird_class = bird_class.lstrip()
    bird_class = re.sub('\n', '', bird_class)
    classes.append(bird_class)
classes_txt.close()

Pigeon_id = classes.index('Pigeons and Doves')
Woodpecker_id = classes.index('Woodpeckers')


# Getting hierarchy information to identify all woodpeckers and pigeons
file_name = os.path.join(dir_name, "Cornell Data Dump/hierarchy.txt")
hierarchy_txt = open(file_name, 'r')
woodpeckers = []
pigeons = []
for line in hierarchy_txt:
    split = re.split(' ', line)
    first = int(split[0])
    second = int(re.sub('\n', '', split[1]))
    if second == Woodpecker_id or second in woodpeckers:
        woodpeckers.append(first)
    if second == Pigeon_id or second in pigeons:
        pigeons.append(first)
hierarchy_txt.close()


# Getting image ids for all woodpeckers and pigeons
file_name = os.path.join(dir_name, "Cornell Data Dump/image_class_labels.txt")
image_labels_txt = open(file_name, 'r')
woodpecker_image_id = []
pigeon_image_id = []
for line in image_labels_txt:
    split = re.split(' ', line)
    first = split[0]
    second = int(re.sub('\n', '', split[1]))
    if second in woodpeckers:
        woodpecker_image_id.append(first)
    if second in pigeons:
        pigeon_image_id.append(first)
image_labels_txt.close()


# Getting images names for all woodpeckers and pigeons
file_name = os.path.join(dir_name, "Cornell Data Dump/images.txt")
image_txt = open(file_name, 'r')
woodpecker_images = []
pigeon_images = []
for line in image_txt:
    split = re.split(' ', line)
    first = split[0]
    second = re.sub('\n', '', split[1])
    if first in woodpecker_image_id:
        woodpecker_images.append(second)
    if first in pigeon_image_id:
        pigeon_images.append(second)
image_txt.close()


#Getting Woodpecker and Pigeon images
Height = 500
Width = 500
Woodpecker_Images = np.zeros((len(woodpecker_images), Height, Width, 3))
for i in range(len(woodpecker_images)):
    image_path = woodpecker_images[i]
    image_path = os.path.join(dir_name, "Cornell Data Dump/photos", image_path)
    image = Image.open(image_path)
    image = image.resize((Height, Width))
    image_array = np.array(image)/255.0
    Woodpecker_Images[i, :, :, :] = image_array
Pigeon_Images = np.zeros((len(pigeon_images), Height, Width, 3))
for i in range(len(pigeon_images)):
    image_path = pigeon_images[i]
    image_path = os.path.join(dir_name, "Cornell Data Dump/photos", image_path)
    image = Image.open(image_path)
    image = image.resize((Height, Width))
    image_array = np.array(image)/255.0
    Pigeon_Images[i, :, :, :] = image_array


# Splitting data into training and testing sets
### Pigeons only have 712 images, so 600 training, 100 testing for each
pigeon_set = random.sample(range(pigeon_images.__len__()), 700)
woodpecker_set = random.sample(range(woodpecker_images.__len__()), 700)
Total_set = np.zeros((1400, Height, Width, 3))
Woodpeckers = Woodpecker_Images[woodpecker_set, :, :, :]
Pigeons = Pigeon_Images[pigeon_set, :, :, :]
Total_set[0:700, :, :, :] = Woodpeckers
Total_set[700:1400, :, :, :] = Pigeons
Y_set = ["Woodpecker"]*700 + ["Pigeon"]*700
Y_set = np.array(Y_set)
lb = sklearn.preprocessing.LabelBinarizer()
Y_label = lb.fit_transform(Y_set)
Y_label = keras.utils.to_categorical(Y_label)
(X_Train, X_Test, Y_Train, Y_Test) = sklearn.model_selection.train_test_split(Total_set, Y_label, test_size=1/7, stratify=Y_label)



#Creating Residual Net architecture
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return(self.activation(Z + skip_Z))

# Creating ResNet-34 Model
ResNet_model = keras.models.Sequential()
ResNet_model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[Height, Width, 3],
                                     padding="same", use_bias=False))
ResNet_model.add(keras.layers.BatchNormalization())
ResNet_model.add(keras.layers.Activation("relu"))
ResNet_model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    ResNet_model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
ResNet_model.add(keras.layers.GlobalAvgPool2D())
ResNet_model.add(keras.layers.Flatten())
ResNet_model.add(keras.layers.Dense(2, activation="softmax"))

#can also use keras.applications.resnet50.ResNet50(), but would need to resize images to 224x224
print("Starting to Train Model")
ResNet_model.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.Adam(lr = 0.001, beta_1=0.9, beta_2=0.999))
ResNet_history = ResNet_model.fit(X_Train, Y_Train, epochs=100,
                                  validation_data=(X_Test, Y_Test))




