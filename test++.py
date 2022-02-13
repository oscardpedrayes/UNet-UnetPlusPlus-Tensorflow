import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from unet import unet
from unet_helpers import load_images
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.losses import sparse_categorical_crossentropy
import keras.backend as K
import imageio
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import os
path = os.getcwd()
print(path)
expName = "P001-03"

RESIZED_WIDTH=256 
RESIZED_HEIGHT=256 
PREPROCESSED_IMAGE_ABSOLUTE_PATH= '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Images10b/'
PREPROCESSED_MASK_ABSOLUTE_PATH= '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Labels3ClassesGrayBKG_2/'
#PREPROCESSED_IMAGE_ABSOLUTE_PATH= '/mtn/20201019/20201019_Sentinel2_Dataset_Clean/Images10b/'
#PREPROCESSED_MASK_ABSOLUTE_PATH= '/mtn/20201019/20201019_Sentinel2_Dataset_Clean/Labels3ClassesGrayBKG/'
TEST_INDEX = path + "/index/3Classes_Test1.txt"

EXP_RESULTS = path + "/exp/" + expName
if not os.path.exists(EXP_RESULTS):
    os.mkdir(EXP_RESULTS)
    os.mkdir(EXP_RESULTS+"/vis")
    os.mkdir(EXP_RESULTS+"/vis/gt")
    os.mkdir(EXP_RESULTS+"/vis/pred")
    os.mkdir(EXP_RESULTS+"/vis/or")

MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH= path + '/models/val_loss_min_unet_' + expName +'.hdf5'
FINAL_MODEL_ABSOLUTE_FPATH= path + '/models/unet_' + expName +'.hdf5'
HISTORY_ABSOLUTE_FPATH= path + '/history/history_' + expName +'.pickle'

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors
        #classSelectors = true

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
        #print("AAAAAAAAAAAAA", loss.shape)
        #print("AAAAAAAAAAA", weightMultiplier.shape)
        loss = loss * weightMultiplier

        return loss
    return lossFunc

def toRGB(predicted_classes, width, height):
    predicted_rgb = np.zeros((width, height, 3))
    for ii in range(width):
        for jj in range(height):
            if predicted_classes[ii,jj] == 0:
                predicted_rgb[ii,jj,0] = 240
                predicted_rgb[ii,jj,1] = 228
                predicted_rgb[ii,jj,2] = 66

            elif predicted_classes[ii,jj] == 1:
                predicted_rgb[ii,jj,0] = 86
                predicted_rgb[ii,jj,1] = 180
                predicted_rgb[ii,jj,2] = 233

            elif predicted_classes[ii,jj] == 2:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 158
                predicted_rgb[ii,jj,2] = 115

            elif predicted_classes[ii,jj] == 3:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 0
                predicted_rgb[ii,jj,2] = 0

            else:
                predicted_rgb[ii,jj,0] = 0
                predicted_rgb[ii,jj,1] = 255
                predicted_rgb[ii,jj,2] = 0
    predicted_rgb = predicted_rgb.astype(int)
    return predicted_rgb

# load saved model, replacing saved_model_name with the saved_model from your training
#class_weights = [1/1.1175, 1/6.0888, 1/0.4397, 1/0.9048]
class_weights = [1, 1, 1, 1]

custom_loss = {"lossFunc":weightedLoss(sparse_categorical_crossentropy, class_weights)}
model = load_model(MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH, custom_objects= custom_loss)
#model = load_model(FINAL_MODEL_ABSOLUTE_FPATH)

# split data, keeping the validation set
X_test, y_test = load_images(PREPROCESSED_IMAGE_ABSOLUTE_PATH,
                            PREPROCESSED_MASK_ABSOLUTE_PATH, TEST_INDEX)

X_test.shape

# predict model masks
predicted_masks = model.predict(X_test)



predicted_classes_set= np.zeros(y_test.shape)
#print(predicted_classes_set.shape)

x_vis = [X_test[17],X_test[1],X_test[4],X_test[10],X_test[7],X_test[3],X_test[6]]
y_vis = [y_test[17],y_test[1],y_test[4],y_test[10],y_test[7],y_test[3],y_test[6]]
pred_vis = [predicted_masks[0][17],predicted_masks[0][1],predicted_masks[0][4],predicted_masks[0][10],predicted_masks[0][7],predicted_masks[0][3],predicted_masks[0][6]]
for i, (image_to_predict, predicted_mask, true_mask) in enumerate(zip(x_vis, pred_vis, y_vis)):
    print("Color Image #: " + str(i))
    plt.imshow(image_to_predict[:,:,[3,2,1]]) # Show only RGB bands
    imageio.imwrite(EXP_RESULTS+"/vis/or/or"+str(i)+".png", image_to_predict[:,:,[3,2,1]])
    #plt.show()
    print("Predicted mask #: " + str(i))
    print(predicted_masks[0].shape)
    #predicted_mask[:,:,1]= predicted_mask[:,:,1]*1.3
    print()

    predicted_classes = np.argmax(predicted_mask, axis=-1)
    print(predicted_mask[1,1,:])
    print(predicted_classes[1,1])
    predicted_rgb = toRGB(predicted_classes, RESIZED_WIDTH, RESIZED_HEIGHT)
    
    plt.imshow(predicted_rgb)
    imageio.imwrite(EXP_RESULTS+"/vis/pred"+"/pred"+str(i)+".png", predicted_rgb)
    #plt.show()
    
    print("True mask #: " + str(i))
    true_mask_rgb = toRGB(true_mask, RESIZED_WIDTH, RESIZED_HEIGHT)   
    plt.imshow(true_mask_rgb)
    imageio.imwrite(EXP_RESULTS+"/vis/gt/gt"+str(i)+".png", true_mask_rgb)
    #plt.show()

plt.show()
# Read in the history for training and validation sets
with open(HISTORY_ABSOLUTE_FPATH, 'rb') as history:
    history = pickle.load(history)
    print(history)

    for i in range(4):
        ii=i+1
        plt.plot(history['output_'+ str(ii) +'_accuracy'])
        plt.plot(history['val_output_'+ str(ii) +'_accuracy'])
        plt.title('Global Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.ylim(0.0,1.0)
        plt.grid(True)
        plt.savefig(EXP_RESULTS + '/'+'output_'+ str(ii) +'_g_accuracy.svg')
        plt.show()
        

        plt.plot(history['output_'+ str(ii) +'_loss'])
        plt.plot(history['val_output_'+ str(ii) +'_loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper right')
        plt.ylim(0.0,1.5)
        plt.grid(True)
        plt.savefig(EXP_RESULTS + '/'+'output_'+ str(ii) +'_loss.svg')
        plt.show()

j=0
for predicted_masks1 in predicted_masks:
    j=j+1
    for i, (image_to_predict, predicted_mask, true_mask) in enumerate(zip(X_test, predicted_masks1, y_test)):
        predicted_classes = np.argmax(predicted_mask, axis=-1)
        predicted_classes_set[i,:,:]=predicted_classes
    predicted_classes_set = predicted_classes_set.astype(int)

    # Calculate Confusion Matrix
    sumCM = tf.math.confusion_matrix(y_test[0,:,:].reshape(RESIZED_HEIGHT*RESIZED_WIDTH), predicted_classes_set[0,:,:].reshape(RESIZED_HEIGHT*RESIZED_WIDTH))
    for i in range(1, y_test.shape[0]):
        sumCM = sumCM + tf.math.confusion_matrix(y_test[i,:,:].reshape(RESIZED_HEIGHT*RESIZED_WIDTH), predicted_classes_set[i,:,:].reshape(RESIZED_HEIGHT*RESIZED_WIDTH))
    conf_mat = np.transpose(sumCM.numpy())
    # Show Confusion Matrix
    #print(conf_mat)

    # Columns (Real)
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
        
    # Row (Predicted)
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
    print("MeanRecall: ", "{:.3f}".format(np.sum(recalls)/len(recalls)))
    print("MeanPrecision: ", "{:.3f}".format(np.sum(precisions)/len(precisions)))
    np.savetxt(EXP_RESULTS + '/'+str(j)+'_conf_mat.csv', finalConfusionMatrix, fmt='%.2f')