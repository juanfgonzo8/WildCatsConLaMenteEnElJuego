from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

import cv2
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input

import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from keras import backend as K

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

K.tensorflow_backend._get_available_gpus()

##
#Se establecen los paths
path_csv = '/media/user_home2/vision2020_01/Data/iWildCam2019/train.csv'
path_train = '/media/user_home2/vision2020_01/Data/iWildCam2019/train_images'

##
#Se crea el modelo

# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet', include_top=False)
# , input_tensor=Input(shape=(4,), name="input")
pre_model = InceptionResNetV2(weights=None, input_shape=(299,299,4), include_top=False)

for new_layer, layer in zip(pre_model.layers[2:], base_model.layers[2:]):
    new_layer.set_weights(layer.get_weights())

# input_layer = Input(shape=(299, 299, 4), name="input")
# base_model.layers[0] = input_layer

# add a global spatial average pooling layer
x = pre_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(14, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=pre_model.input, outputs=predictions)

#model = Model(inputs=input_layer, outputs=model1[1:])

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False
print('Out')
for layer in model.layers[0:5]:
    print(layer.name)
print('Base')
for layer in base_model.layers[0:5]:
    print(layer.name)
print('Pre')
for layer in pre_model.layers[0:5]:
    print(layer.name)

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

##
#Se cargan los datos
train_df = pd.read_csv(path_csv)
train_df['category_id'] = train_df['category_id'].astype(str)
train_df['location'] = train_df['location'].astype(int)

##
# Anadir capa con contenido del lugar

def cuartaCapa(im):
    global train_df
    loca = train_df['location'][cuartaCapa.pos]
    cap = np.zeros((299,299,1))
    cap.fill(np.float32(loca))
    im_nueva = np.concatenate((im,cap),axis=2)
    cuartaCapa.pos += 1
    # print('promedio '+str(np.mean(cap)))
    # print('locacion '+str(loca))
    # print('posi '+str(cuartaCapa.pos))
    return im_nueva

cuartaCapa.pos = 0

##
batch_size=16
img_size = 299
nb_epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)  #,preprocessing_function=cuartaCapa)
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
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy',f1])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
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


for epoch in range(nb_epochs):

    print('Epoch '+str(epoch+1)+'/10')
    cont = int(1)
    steps_train = int(train_generator.samples // batch_size)
    for image_batch, label_batch in train_generator:
        new_batch = np.zeros((batch_size, 299, 299, 4))
        for i,im in enumerate(image_batch):
            im_new = cuartaCapa(im)
            new_batch[i,:,:,:] = im_new
        if cont == steps_train:
            print('Metricas train')
            print(model.train_on_batch(np.float32(new_batch),y=label_batch,reset_metrics=False))
        print('Batch '+str(cont)+'/'+str(steps_train))
        cont += 1
        model.train_on_batch(np.float32(new_batch),y=label_batch,reset_metrics=False)

    steps_test = validation_generator.samples // batch_size
    cont = int(1)
    for image_batch, label_batch in train_generator:
        new_batch = np.zeros((batch_size, 299, 299, 4))
        for i,im in enumerate(image_batch):
            im_new = cuartaCapa(im)
            new_batch[i,:,:,:] = im_new
        if cont == steps_test:
            print(model.test_on_batch(np.float32(new_batch),label_batch,reset_metrics=False))
        model.test_on_batch(np.float32(new_batch),label_batch,reset_metrics=False)
        cont += 1
    print('Metricas test')
    print('..')

    model.reset_metrics()



# image_batch, label_batch = next(train_generator)
# print(image_batch.shape)
# print(label_batch.shape)