from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
print(tf.__version__)

import keras
from keras import layers
import os
from keras.models import load_model
from keras.callbacks import EarlyStopping

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train = ImageDataGenerator(rescale = 1/255)
val =ImageDataGenerator(rescale=1/255)

# train_path = 'C:/Users/minju/flag_data/train'
# val_path = 'C:/Users/minju/flag_data/val'

train_path = '../data/train'
# val_path = os.path.join('C:/', 'Users', 'minju', 'flag_data', 'val')
val_path = '../data/val'

train_dataset = train.flow_from_directory(train_path,
                                          target_size = (200,200),
                                          batch_size = 32,
                                          shuffle=True,
                                          class_mode='binary')

val_dataset = train.flow_from_directory(val_path,
                                          target_size = (200,200),
                                          batch_size = 32,
                                          shuffle=True,
                                          class_mode='binary')



xception= keras.applications.Xception(weights='imagenet',include_top=False, input_shape=(200,200,3))

x = xception.output

flatten_layer = layers.Flatten()  # instantiate the layer
x = flatten_layer(x)

x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
prediction = layers.Dense(1, activation='sigmoid')(x)

model =keras.Model(xception.input, prediction)

model.compile(loss= keras.losses.BinaryCrossentropy() , optimizer= keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

model.summary()

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 2,
                      epochs= 50,
                      callbacks= [EarlyStopping(monitor='val_loss',patience=10)],
                      validation_data = val_dataset)

model.save('model/xception.h5')
