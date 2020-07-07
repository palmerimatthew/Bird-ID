import os
import re
from PIL import Image
import numpy as np
import random

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

print(pigeon_images.__len__())
print(woodpecker_images.__len__())

#Getting Woodpecker and Pigeon images
Length = 500
Width = 500
Woodpecker_Images = np.zeros((len(woodpecker_images), Length, Width, 3))
for i in range(len(woodpecker_images)):
    image_path = woodpecker_images[i]
    image_path = os.path.join(dir_name, "Cornell Data Dump/photos", image_path)
    image = Image.open(image_path)
    image = image.resize((Length, Width))
    image_array = np.array(image)/255.0
    Woodpecker_Images[i, :, :, :] = image_array
Pigeon_Images = np.zeros((len(pigeon_images), Length, Width, 3))
for i in range(len(pigeon_images)):
    image_path = pigeon_images[i]
    image_path = os.path.join(dir_name, "Cornell Data Dump/photos", image_path)
    image = Image.open(image_path)
    image = image.resize((Length, Width))
    image_array = np.array(image)/255.0
    Pigeon_Images[i, :, :, :] = image_array

# Splitting data into training and testing sets
### Pigeons only have 712 images, so 600 training, 100 testing for each
pigeon_set = random.sample(range(pigeon_images.__len__()), 600)
Pigeon_Train = Pigeon_Images[pigeon_set[0:499], :, :, :]

#Creating Residual Net architecture


