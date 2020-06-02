from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt

import argparse

import cv2
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from keras import backend as K

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

K.tensorflow_backend._get_available_gpus()


##
#Se toman los argumentos de entrada
parser = argparse.ArgumentParser(description='Codigo principal')
parser.add_argument('--mode', type=str, default=None,
                    help='Define si se prueba el codigo o se entrena (None, demo o test)')
parser.add_argument('--img', type=str, default=None,
                    help='Imagen a probar en modo demo')

args = parser.parse_args()

##
#Se establecen los paths
path_csv = '/media/user_home2/vision2020_01/Data/Proyectos_finales/iWildCam2019/train.csv'
path_train = '/media/user_home2/vision2020_01/Data/Proyectos_finales/iWildCam2019/train_images'
path_pesos = '/media/user_home2/vision2020_01/Data/Proyectos_finales/iWildCam2019/Pesos'

##
#Se plantan seeds
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

##
#Se crea el modelo

# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(14, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

##
#Se cargan los datos
train_df = pd.read_csv(path_csv)
train_df['category_id'] = train_df['category_id'].astype(str)

batch_size=32
img_size = 299
nb_epochs = 20

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = path_train,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',shuffle=False)

validation_generator  = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = path_train,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',shuffle=False)

set(train_generator.class_indices)
nb_classes = 14

##Se entrena el modelo usando fine-tune

# train the model on the new data for a few epochs
#model.fit(...)

# Train model
# history = model.fit_generator(
#             train_generator,
# #             steps_per_epoch = train_generator.samples // batch_size,
#             steps_per_epoch = 100,
#             validation_data = validation_generator,
# #             validation_steps = validation_generator.samples // batch_size,
#             validation_steps = 50,
#             epochs = nb_epochs,
#             verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True


##
#Metrica final
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy',f1])

##
#Se saca la matriz de confusion

#Se evalua el modelo
model.load_weights(path_pesos+'/pesos_inicial.h5')
model.evaluate_generator(validation_generator, steps=200, verbose=2)
pred = model.predict_generator(validation_generator,steps=200,verbose=2)
predicted = np.argmax(pred, axis=1)

#Se muestra la matriz de confusion
cm = confusion_matrix(validation_generator.classes[0:(200*32)], np.argmax(pred, axis=1),normalize='all')
fig = plt.figure(figsize = (30,20))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
fig.savefig('matriz.png')


#Reporte de clasificacion
class_names = ['Empty','Deer','Fox','Coyote','Racoon','Skunk','Bobcat','Cat','Dog','Opposum','Mountain Lion',
               'Squirrel','Rodent','Rabbit']
print(classification_report(validation_generator.classes[0:(200*32)], predicted, target_names=class_names))