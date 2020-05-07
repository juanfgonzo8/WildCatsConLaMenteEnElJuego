from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

import cv2
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

##
#Se establecen los paths
path_csv = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train.csv'
path_train = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images'

##
#Se crea el modelo

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

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
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

##
#Se cargan los datos
train_df = pd.read_csv(path_csv)
train_df['category_id'] = train_df['category_id'].astype(str)

batch_size=32
img_size = 299
nb_epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = path_train,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

validation_generator  = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = path_train,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

set(train_generator.class_indices)
nb_classes = 14

##Se entrena el modelo usando fine-tune

# train the model on the new data for a few epochs
#model.fit(...)

# Train model
history = model.fit_generator(
            train_generator,
#             steps_per_epoch = train_generator.samples // batch_size,
            steps_per_epoch = 100,
            validation_data = validation_generator,
#             validation_steps = validation_generator.samples // batch_size,
            validation_steps = 50,
            epochs = nb_epochs,
            verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit(...)

# Train model
history = model.fit_generator(
            train_generator,
#             steps_per_epoch = train_generator.samples // batch_size,
            steps_per_epoch = 100,
            validation_data = validation_generator,
#             validation_steps = validation_generator.samples // batch_size,
            validation_steps = 50,
            epochs = nb_epochs,
            verbose=2)