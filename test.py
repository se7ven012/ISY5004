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
from utils.dice import dice_coef
from utils.models import combined_loss, mean_iou, MaxUnpooling2D, MaxPoolingWithArgmax2D
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
fig = plt.figure(figsize=(9,24), dpi=200)
for ix in range(5,8):
    a = fig.add_subplot(7, 3, ix-4)
    plt.imshow(X_train[int(X_train.shape[0] * 0.9) :][ix])
    a.set_title("Original_"+str(ix-5))
for ix in range(8,11):
    a = fig.add_subplot(7, 3, ix-4)
    plt.imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9) :][int(ix-3)]))
    a.set_title("Ground Truth_"+str(int(ix-8)))

# Predict on train, val and test
counter = 0
names = ["Unet", "SegNet", "SegUnet", "UnetPP", "ResUNet"]

for name in names:
    model=load_model(
        "h5/" + name + ".h5",
        custom_objects={
            "mean_iou": mean_iou,
            "combined_loss": combined_loss,
            "MaxPoolingWithArgmax2D": MaxPoolingWithArgmax2D,
            "MaxUnpooling2D": MaxUnpooling2D,
        },
    )
    preds_train = model.predict(X_train[: int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9) :], verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(
            resize(
                np.squeeze(preds_test[i]),
                (sizes_test[i][0], sizes_test[i][1]),
                mode="constant",
                preserve_range=True,
            )
        )

    for ix in range(11,14):
        a = fig.add_subplot(7, 3, ix-4+counter)
        plt.imshow(np.squeeze(preds_val_t[ix-6]))
        a.set_title(name +"_"+str(ix-11))
    counter+=3

    dice_coefs = []
    for i in range(len(preds_val_t)):
        dice_coefs.append(
            dice_coef(
                np.squeeze(Y_train[int(Y_train.shape[0] * 0.9) :][i]),
                np.squeeze(preds_val_t[i]),
            )
        )
    print("The mean dice coeffient of " + name + ": " + str(np.mean(dice_coefs)))
fig.savefig('./model_plot/SegmentationResults.png')
# %%
