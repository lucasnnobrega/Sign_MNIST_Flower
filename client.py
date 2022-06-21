import os

import flwr as fl
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import pandas as pd
import argparse

import constantes as CTES

parser = argparse.ArgumentParser(description='file_name')
parser.add_argument('file_path', metavar='N', type=str,
                    help='filepath')

args = parser.parse_args()
print(args.file_path)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
# model = tf.keras.applications.MobileNetV2((28, 28, 1), classes=10, weights=None)


def convolutional_model():
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3, 3), activation='relu', input_shape=(28,28,1)),
    # MaxPooling2D calcula o valor máximo de uma matriz 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=26, activation='softmax')
    ])
    
    return model 

def flatten_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(13, activation='softmax')
    tf.keras.layers.Dense(26, activation='softmax')
    ])

    return model 

model = convolutional_model()
#model = flatten_model()

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def load_data(file_path):
    
    df = pd.read_csv(file_path+"_train.csv")
    if "client" in file_path:
        # removendo a primeira coluna
        df = df.iloc[: , 1:]

    print(df.columns)

    y_train = df.iloc[: , 0].values
    y_train = y_train.reshape(y_train.shape[0],1)
    
    x_train = df.iloc[: , 1:]
    images = []
    for i, row in x_train.iterrows():
        img_format = row.values.reshape(28,28)
        images.append(img_format)
    x_train = np.array(images)
    x_train = x_train/255.0
    
    
    df = pd.read_csv(file_path+"_test.csv")
    if "client" in file_path:
        # removendo a primeira coluna
        df = df.iloc[: , 1:]
    
    print(df.columns)

    y_test = df.iloc[: , 0].values
    y_test = y_test.reshape(y_test.shape[0],1)
    
    x_test = df.iloc[: , 1:]
    images = []
    for i, row in x_test.iterrows():
        img_format = row.values.reshape(28,28)
        images.append(img_format)
    x_test = np.array(images)
    x_test = x_test/255.0

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    return (x_train, y_train), (x_test, y_test)

# Adicionar conjunto de validação para hiperparametros
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = load_data(args.file_path)

print(CTES.GREEN, "Dataset Antes", CTES.RESET)
for i in [x_train, y_train, x_test, y_test]:
    print(i.shape)

# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # aleatoriamente rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # aleatoriamente zoom image 
        width_shift_range=0.1,  # aleatoriamente shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # aleatoriamente shift images vertically (fraction of total height)
        horizontal_flip=False,  # aleatoriamente inverter as imagens - nao implementado devido a mudanca de significado 
        vertical_flip=False)  # aleatoriamente inverter as imagens - nao implementado devido a mudanca de significado 

datagen.fit(x_train)

#print(y_train[1:4])
#exit()

# Define Flower client
class MyClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(datagen.flow(x_train,y_train, batch_size = 32, subset="training"),
                  epochs=5,
                  validation_data= datagen.flow(x_train, y_train,
                                    batch_size=8, subset='validation'))
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        print("EVALUATE")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=MyClient())
