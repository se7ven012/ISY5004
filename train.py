#%%
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K

from utils.models import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = "./data-science-bowl-2018/stage1_train/"
TEST_PATH = "./data-science-bowl-2018/stage1_test/"

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
seed = 42
random.seed = seed
np.random.seed = seed
#%%
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

#%%
# Get and resize train images and masks
X_train = np.zeros(
    (len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print("Getting and resizing train images and masks ... ")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + "/masks/"))[2]:
        mask_ = imread(path + "/masks/" + mask_file)
        mask_ = np.expand_dims(
            resize(
                mask_, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True
            ),
            axis=-1,
        )
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
    X_test[n] = img

print("Done!")

#%%
# Check if training data looks all right
ix = random.randint(0, len(train_ids))
fig = plt.figure(dpi=200)
a = fig.add_subplot(1, 2, 1)
plt.imshow(X_train[ix])
a.set_title("Original")

a = fig.add_subplot(1, 2, 2)
plt.imshow(np.squeeze(Y_train[ix]))
a.set_title("Mask")

print("Height: " + str(IMG_HEIGHT) + " Width: " + str(IMG_WIDTH))
fig.savefig('./model_plot/OrignalMask.png')
#%%
def train(model, x_train, y_train, plotfile):
    # Fit model
    results = model.fit(
        X_train,
        Y_train,
        validation_split=0.1,
        batch_size=16,
        epochs=60,
        callbacks=callbacks_list,
    )
    plot_model(
        model, to_file=plotfile, show_shapes=True, show_layer_names=False, rankdir="TB",
    )
    print("Path to plot:", plotfile)

# Fit models
names = ["Unet", "SegNet", "SegUnet", "UnetPP", "ResUNet"]
for name in names:
    # model name
    plotpath = "./model_plot/" + name
    plotfile = plotpath + ".png"
    # Setup callbacks
    earlystopper = EarlyStopping(patience=15, verbose=1)
    checkpointer = ModelCheckpoint(
        "h5/" + name + ".h5", monitor="val_loss", verbose=1, save_best_only=True
    )
    csv_logger = CSVLogger(plotpath + ".csv")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, mode="auto", min_lr=0.00000001
    )
    callbacks_list = [earlystopper, checkpointer, csv_logger, reduce_lr]
    if name == "Unet":
        m = get_unet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        train(m, X_train, Y_train, plotfile)
    elif name == "SegNet":
        m = get_segnet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        train(m, X_train, Y_train, plotfile)
    elif name == "SegUnet":
        m = get_segunet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        train(m, X_train, Y_train, plotfile)
    elif name == "UnetPP":
        m = get_unetpp(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        train(m, X_train, Y_train, plotfile)
    elif name == "ResUNet":
        m = get_ResUNet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        train(m, X_train, Y_train, plotfile)
    else:
        print("Error, Model name does not match")
        break
