import pickle
import os
import numpy as np
from unet_helpers import (load_images)
from keras.models import Model
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          UpSampling2D,
                          Dropout,
                          Concatenate,
                          Activation,
                          MaxPool2D,
                          Conv2DTranspose,
                          BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import sparse_categorical_crossentropy
import keras.backend as K
import tensorflow as tf
from keras import regularizers


import datetime

# Clear GPU memory
from numba import cuda
cuda.select_device(0)
cuda.close()

# Check if it's using GPU

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
tf.debugging.set_log_device_placement(False)

# Get actual Path
import os
path = os.getcwd()
print(path)
expName = "TF2.2"


# Parameters
RESIZED_WIDTH=256 
RESIZED_HEIGHT=256 
PREPROCESSED_IMAGE_ABSOLUTE_PATH= '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Images10b/'
PREPROCESSED_MASK_ABSOLUTE_PATH= '/home/oscar/Desktop/20201019_Sentinel2_Dataset_Clean/Labels3ClassesGrayBKG_2/'
TRAIN_INDEX = path + "/index/3Classes_Train1.txt" 
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
BATCH_SIZE=32
NUM_EPOCHS=125 #1500
LEARNING_RATE=5e-4
EARLY_STOP_VAL_PATIENCE=30


def conv_block(inputs, filters, pool=True, drop=False):
    x = Conv2D(filters, 3, padding="same")(inputs) #kernel_regularizer=regularizers.l2(0.0001)
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    #x = BatchNormalization()(x)
    x = Activation("relu")(x) 
    
    if pool == True and drop == True:
        p = Dropout(0.5)(x)
        p = MaxPool2D((2, 2))(p)
        return x, p
    elif pool == True and drop == False:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def build_unet(shape, num_classes):
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 32, pool=True) #16
    x2, p2 = conv_block(p1, 64, pool=True) #32
    x3, p3 = conv_block(p2, 128, pool=True) #48
    x4, p4 = conv_block(p3, 256, pool=True, drop=True) #64

    """ Bridge """
    b1 = conv_block(p4, 512, pool=False)
    drop2 = Dropout(0.5)(b1)

    """ Decoder """
    u1 = Conv2DTranspose(256, 2, strides=2, padding="same")(drop2) #UpSampling2D((2, 2), interpolation="bilinear")(drop2)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 256, pool=False) #64

    u2 = Conv2DTranspose(128, 2,strides=2, padding="same")(x5) #UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 128, pool=False) #48

    u3 = Conv2DTranspose(64, 2,strides=2, padding="same")(x6) #UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 64, pool=False) #32

    u4 = Conv2DTranspose(32, 2, strides=2, padding="same")(x7) #UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 32, pool=False) #16

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return Model(inputs, output)

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
        loss = loss * weightMultiplier


        return loss
    return lossFunc

def train(input_shape, num_classes,
          image_dir,
          mask_dir,
          val_loss_min_fpath,
          final_model_output_fpath,
          history_output_fpath,
          batch_size,
          num_epochs,
          learning_rate,
          early_stop_val_patience):
    '''Trains the unet model, saving the model+weights that performs the
       best on the validation set after each epoch (if there is improvement)
       as well as the final model+weights at the last epoch,
       regardless of performance. History is pickeled once training is
       finished.

    Params: input_shape -- the shape of one image in the dataset,
                           for instance (512, 512, 3)
            batch_size -- the number of images to be processed together
                          in one step through the model
            num_epochs -- the number of times the entire data set is passed
                          through the model
                          (1 epoch=(num steps through the model)*(batch_size))
            learning_rate -- a scaler value that controls how much the
                             weights get updated in the descent
            early_stop_val_patience -- number of epochs to stop training after
                                       if no improvement on the validation
                                       dataset occurs
            val_loss_min_fpath -- the path (including file name) where the
                                  model that performed the best on the
                                  validation set should be saved
            final_model_output_fpath -- the path (including file name) where
                                        the final trained model is saved
            history_output_fpath -- the path (including file name) where
                                    the history's dictionary is saved
            image_dir -- directory of images that will be input into
                         the model (when sorted the filenames in
                         this directory must correspond to the sorted
                         filenames in mask_dir)
            mask_dir -- directory of masks that will be input into the
                        model (when sorted the filenames in this directory
                        must correspond to the sorted filenames in image_dir)

    Returns (tuple): history -- keys and values that are useful for analyzing
                                (and plotting) data collected during training
                     model -- the trained model. Note: this model likely
                              overfits to the training data. To use the most
                              performant model on the validation set,
                              refer to the model saved to 'val_loss_min_fpath'
    '''

    # Get images and groundTruth sets for training and test
    X_train, y_train = load_images(image_dir, mask_dir, TRAIN_INDEX)
    X_test, y_test = load_images(image_dir, mask_dir, TEST_INDEX)
    #print(X_train,y_train)

    # Add weights to fix unbalanced classes
    class_weights = [1.1175, 6.0888, 0.4397, 0.9048]

    #model = unet(input_shape, num_classes)
    model = build_unet(input_shape, num_classes)

    model.compile(optimizer=Adam(lr=learning_rate, epsilon=0.00000001),
                  loss= weightedLoss(sparse_categorical_crossentropy, class_weights), metrics=['accuracy']) #loss="sparse_categorical_crossentropy", sample_weight_mode="temporal") # loss_weights=[1/1.1175, 0.01/6.0888, 1/0.4397, 1/0.9048]) #loss_weights=[0.9048, 0.4397, 6.0888, 1.1175]) 
    val_loss_checkpoint = ModelCheckpoint(val_loss_min_fpath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='min')
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=early_stop_val_patience)

    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, val_loss_checkpoint],
                        shuffle=True)

    model.save(final_model_output_fpath)
    with open(history_output_fpath, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    return history, model

print("Starting training.")
begin_time = datetime.datetime.now()
history, model = train((RESIZED_WIDTH, RESIZED_HEIGHT, 10), 4,
                       PREPROCESSED_IMAGE_ABSOLUTE_PATH,
                       PREPROCESSED_MASK_ABSOLUTE_PATH,
                       MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH,
                       FINAL_MODEL_ABSOLUTE_FPATH,
                       HISTORY_ABSOLUTE_FPATH,
                       BATCH_SIZE,
                       NUM_EPOCHS,
                       LEARNING_RATE,
                       EARLY_STOP_VAL_PATIENCE)
lapsed_time = datetime.datetime.now() - begin_time                       
print("Ending training. Elapsed time: ", lapsed_time)
with open(EXP_RESULTS+'/time.txt', 'w') as out_file:
     out_file.write(lapsed_time)

# Clear GPU memory
from numba import cuda
cuda.select_device(0)
cuda.close()
print("Memory clean.")