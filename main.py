#import all the libraries
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import skimage.io as io
from PIL import Image
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from model import *

def trainGenerator(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"*.png"))
    mask_name_arr = glob.glob(os.path.join(mask_path, "*.png"))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        img = img / 255
        image_arr.append(img)

    for index, item in enumerate(mask_name_arr):
        mask = io.imread(item, as_gray = image_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        mask = mask/255
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        mask_arr.append(mask)

    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def testGenerator(test_path, image_num):
    for i in range(image_num):
        idx = str(i+1).zfill(5) + ".png"
        img = io.imread(os.path.join(test_path, idx), as_gray=True)
        img = img / 255
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

def getItem(img_path, item):
    idx = str(item+1).zfill(5) + ".png"
    img = io.imread(os.path.join(img_path, idx), as_gray=True)
    img = img/255.0
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,)+img.shape)
    return img

def saveResult(save_path, files):
    for i, item in enumerate(files):
        img = item[:,:,0]
        idx = str(i+1).zfill(5) + ".png"
        io.imsave(os.path.join(save_path, idx), img)

def viewImage(array1, array2, i):
    img = np.reshape(array1[i], (256, 256))
    mask = np.reshape(array2[i], (256, 256))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img, cmap=plt.cm.gray)
    axarr[1].imshow(mask, cmap=plt.cm.gray)
    plt.show()

# Load the train data and their corresponding labels
# Visualize a sample image
imgs_train,imgs_mask_train= trainGenerator("train/images/","train/masks/")
print(imgs_train.shape)   
print(imgs_mask_train.shape)
index = np.random.randint(1, 100)
viewImage(imgs_train, imgs_mask_train, index)

# laod model and show the model summary
model = unet()
#model_checkpoint = ModelCheckpoint('fiber_segmentation.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.summary()

# train the model
hm_epochs = 3
batch_size = 5
#model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=hm_epochs, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=hm_epochs, verbose=1, validation_split=0.2, shuffle=True)

# testing with new images
# the test image folder contains 30 images
test_path = 'test/images/'
test_image_arr = glob.glob(os.path.join(test_path, "*.png"))
num_of_test_images = len(test_image_arr)

testGene = testGenerator(test_path, num_of_test_images)

results = model.predict_generator(testGene, num_of_test_images, verbose=1)

# save the segmentation result in the "outputs" folder
saveResult("outputs/", results)

