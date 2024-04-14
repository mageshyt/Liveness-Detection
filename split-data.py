import os
import random
import shutil
from itertools import islice
import yaml

# Constants for offset percentages
OUTPUT_FOLDER = 'datasets/splitData'  # Folder to save images
INPUT_FOLDER = 'datasets/all'  # Folder to read images
classes = ['fake', 'real']

try:
    shutil.rmtree(OUTPUT_FOLDER)
    print("Removed existing folder")
except OSError as e:
    print("Creating new folder")
    os.mkdir(OUTPUT_FOLDER)

# Split data into training and testing sets
os.makedirs(OUTPUT_FOLDER + '/train/images', exist_ok=True)
os.makedirs(OUTPUT_FOLDER + '/train/labels', exist_ok=True)

os.makedirs(OUTPUT_FOLDER + '/val/images', exist_ok=True)
os.makedirs(OUTPUT_FOLDER + '/val/labels', exist_ok=True)

os.makedirs(OUTPUT_FOLDER + '/test/images', exist_ok=True)
os.makedirs(OUTPUT_FOLDER + '/test/labels', exist_ok=True)

# get the names

listNames = os.listdir(INPUT_FOLDER)
print(listNames)
print("Size of the list is: ", len(listNames))

uniqueNames = []

for name in listNames:
    uniqueNames.append(name.split('.')[0])  # get the name without the extension

# remove duplicates
uniqueNames = list(set(uniqueNames))

print("Unique names size : ", len(uniqueNames))

# shuffle the data

random.shuffle(uniqueNames)

# split the data into 70% training, 20% validation, and 10% testing

train_size = int(0.7 * len(uniqueNames))
val_size = int(0.2 * len(uniqueNames))
test_size = int(0.1 * len(uniqueNames))

print("Train size: ", train_size)
print("Validation size: ", val_size)
print("Test size: ", test_size)

train_names = uniqueNames[:train_size]
val_names = uniqueNames[train_size:train_size + val_size]
test_names = uniqueNames[train_size + val_size:]


# copy the images and labels to the respective folders
def copyFiles(names, folder):
    for name in names:
        # copy on Input folder
        shutil.copyfile(f"{INPUT_FOLDER}/{name}.jpg", f"{OUTPUT_FOLDER}/{folder}/images/{name}.jpg")
        shutil.copyfile(f"{INPUT_FOLDER}/{name}.txt", f"{OUTPUT_FOLDER}/{folder}/labels/{name}.txt")


copyFiles(train_names, 'train')
copyFiles(val_names, 'val')
copyFiles(test_names, 'test')

print("Data split successfully!")

# create data yaml
data = dict(
    train=f"{OUTPUT_FOLDER}/train/images",
    val=f"{OUTPUT_FOLDER}/val/images",
    nc=len(classes),
    names=classes
)

# write yaml file

with open(f"{OUTPUT_FOLDER}/data.yaml", 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

print("Data yaml created successfully!")
