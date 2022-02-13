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
                          concatenate,
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
expName = "P001-03"


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
BATCH_SIZE=10
NUM_EPOCHS=125 #1500
LEARNING_RATE=3e-4
EARLY_STOP_VAL_PATIENCE=30
act = "relu"
dropout_rate = 0.5
SUPERVISION = True

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

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def UNetPlusPlus(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    #if K.image_dim_ordering() == 'tf':
    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    #else:
    #  bn_axis = 1
    #  img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same',kernel_regularizer=regularizers.l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(img_input, nestnet_output_4)

    return model


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

        #axis = -1 #if channels last 
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
    model = UNetPlusPlus(256,256,10, num_classes, deep_supervision=SUPERVISION)

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
     out_file.write(str(lapsed_time))

# Clear GPU memory
from numba import cuda
cuda.select_device(0)
cuda.close()
print("Memory clean.")