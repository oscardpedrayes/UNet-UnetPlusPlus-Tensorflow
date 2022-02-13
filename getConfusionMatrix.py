import os
import numpy as np
from PIL import Image
import PIL 
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
path = os.getcwd()
import sys

#EXP_NAME = sys.argv[1] #"E001-06"
SAVE_PATH = sys.argv[1]  #'/home/oscar/Desktop/exps/levels/metrics/' + EXP_NAME +'/'
GROUNDTRUTH_PATH = sys.argv[2]  #'/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/masksLevels512x384_gt/'
PREDICTED_PATH = sys.argv[3]   #"/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/expLevels/vis/" + EXP_NAME + "/detections2/"


def toGrayscale(img, width, height):

    # Grayscale mask
    imgGray = np.random.randn(height, width, 1)

    for i in range(height):
        for j in range(width):  
            pixel = [img[i,j,0],img[i,j,1],img[i,j,2]] 

            if pixel == [0, 0, 0]: color = 0             
            elif pixel == [0, 0, 255]: color = 1 
            elif pixel == [0, 255, 0]: color = 2 
            elif pixel == [255, 0, 0]: color = 3
            else:
                color = 255 

            imgGray[i,j] = color


    return imgGray


sumCM = None
for _, _, files in os.walk(GROUNDTRUTH_PATH, topdown=False):
    for name in files:
        if os.path.isfile(PREDICTED_PATH + name.replace('_gt','_image')):
            print(name)
            # Read GT image
            gtImage = Image.open(GROUNDTRUTH_PATH + '/' + name)
            width = gtImage.width 
            height = gtImage.height 
            #print(width, height)
            # Read PREDICTED image
            predImage = Image.open(PREDICTED_PATH + '/' + name.replace('_gt','_image'))

            # Prepare data
            gtImg = np.array(gtImage)
            gtImg = gtImg.reshape(height*width)

            predImg = np.array(predImage)
            predImg = toGrayscale(predImg, width, height).reshape(height*width)

            # Calculate Confusion Matrix
            tmp_cm = tf.math.confusion_matrix(gtImg, predImg, num_classes=3)
            if sumCM == None:
                sumCM = tmp_cm
            else:
                sumCM = sumCM + tmp_cm

conf_mat = np.transpose(sumCM.numpy())
print(conf_mat)
# Columns (GT)
cols = []
recalls = []
diagonals = []
for i in range(conf_mat.shape[0]):
    col = np.sum(conf_mat[:,i])
    cols.append(col)
    recall = conf_mat[i,i] / col
    if col == 0:
        recall = 0.0
    recalls.append(recall)
    diagonals.append(conf_mat[i,i])
    
# Rows (Predicted)
rows = []
precisions = []
for i in range(conf_mat.shape[1]):
    row = np.sum(conf_mat[i,:])
    rows.append(row)
    precision = conf_mat[i,i] / row
    if row == 0:
        precision = 0.0
    precisions.append(precision)

# GlobalAccuracy
globalAccuracy = np.sum(diagonals) / np.sum(rows)
print(np.sum(rows))
# Prevent cientific notation
np.set_printoptions(suppress=True)

# Create new confusion matrix with +2 en each axis
finalConfusionMatrix = np.zeros((sumCM.shape[0]+2, sumCM.shape[1]+2))
# Set to None unused cells
finalConfusionMatrix[0,0] = None #top-left
finalConfusionMatrix[-1,0] = None #botttom-left
finalConfusionMatrix[0,-1] = None #top-right
# Add cols sums
finalConfusionMatrix[0,1:1+len(rows)] = rows
# Add rows sums
finalConfusionMatrix[1:1+len(cols),0] = cols
# Add recalls in last row
finalConfusionMatrix[-1, 1:1+len(precisions)] = np.round(precisions,2)
# Add precisions in last column
finalConfusionMatrix[ 1:1+len(recalls),-1] = np.round(recalls,2)
# Add globalAccuracy in last cell
finalConfusionMatrix[-1,-1] = np.round(globalAccuracy,2)
# Add the confusion matrix
finalConfusionMatrix[1:1+sumCM.shape[0],1:1+sumCM.shape[1]] = sumCM
finalConfusionMatrix = np.transpose(finalConfusionMatrix)
print(finalConfusionMatrix)

meanRecall = np.sum(recalls)/len(recalls) 
meanPrecision = np.sum(precisions)/len(precisions)
print()
print("MeanRecall: ", "{:.3f}".format(meanRecall))
print("MeanPrecision: ", "{:.3f}".format(meanPrecision))
print("MeanF1: ", "{:.3f}".format(2*meanPrecision*meanRecall/(meanRecall+meanPrecision)))
print("Recalls: ", recalls)
print("Precision: ", precisions)
print()

np.savetxt(SAVE_PATH + '/conf_mat.csv', finalConfusionMatrix, fmt='%.2f')
