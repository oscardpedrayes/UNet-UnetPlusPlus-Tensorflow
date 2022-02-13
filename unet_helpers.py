import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
import imageio

def load_images(image_dir, mask_dir, index):
    '''
    Params: image_dir -- the directory containing images that are in
                         the correct shape and format for the model to consume
            mask_dir -- the directory containing the corresponding masks
                        that are in the correct shape and format for
                        the model to consume

            Note: The images should be in RGB format and have the shape
            (width_or_height, width_or_height, 3), and masks should be
            grayscale and have the shape (width_or_height, width_or_height, 1)
            where the last number in the tuple denotes number of channels.
            The images and masks should be named so that when each respective
            directory is sorted, the filenames align.

    Returns (tuple): images -- numpy array of images
                     masks -- numpy array of masks

    '''
    names = open(index, 'r')
    masks = []
    images = []
    # sorted_image_names = sorted(os.listdir(image_dir))
    # sorted_mask_names = sorted(os.listdir(mask_dir))
    #  for image_name, mask_name in zip(sorted_image_names, sorted_mask_names):
    for name in names:
        # reads images 
        name = name.split("\n")[0] 
        #print(name)
        image = imageio.imread(os.path.join(image_dir, name))
      
        images.append(image)

        mask = imageio.imread(os.path.join(mask_dir, name.replace(".tif",".png")))

        # the masked images may not be all 0 and 1s, because of
        # artifacts/noise in preprocessing. Convert each pixel to 0 or 1
        # depending on a reasonable threshold from looking at the data
        #reasonable_threshold = 100
        #mask[mask < reasonable_threshold] = 0
        #mask[mask >= reasonable_threshold] = 1
        masks.append(mask)
    return np.array(images), np.array(masks)

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        #classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors
        classSelectors = true

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(K.cast(i, "uint8"), K.cast(classSelectors, "uint8")) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        print("AAAAAAAAAAAAA", loss.shape)
        print("AAAAAAAAAAA", weightMultiplier.shape)
        loss = loss * weightMultiplier

        return loss
    return lossFunc