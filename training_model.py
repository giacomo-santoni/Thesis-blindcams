import numpy as np
import Preprocessing_new as prep
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models

file_drdf_list = ["/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/initial_data_good/response.drdf","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data/response_pde25.drdf","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data2/response.drdf"]
file_root_list = ["/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/initial_data_good/sensors.root","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data/sensors_pde25.root","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data2/sensors.root"]

#AUGMENTATION
def Augmentation(rname: list, fname: list):
    input_shape = [32,32,32,1]
    data_augmentation = keras.Sequential()
    data_augmentation.add(layers.RandomFlip("horizontal", input_shape=input_shape[1:]))
    data_augmentation.add(layers.RandomRotation(0.2))
    data_augmentation.add(layers.RandomTranslation(0.2,0.2))
    
    nr_blind_events1, blind_events1 = prep.FindBlindEvents(rname[0], fname[0])
    nr_blind_events2, blind_events2 = prep.FindBlindEvents(rname[1], fname[1])
    nr_blind_events3, blind_events3 = prep.FindBlindEvents(rname[2], fname[2])
    
    all_blind_events = np.concatenate((blind_events1,blind_events2,blind_events3), axis=0)

    augmented_ds = []
    for blind_im in all_blind_events:
        for i in range(6):
            augmented_image = data_augmentation(blind_im)
            augmented_image_sq = np.squeeze(augmented_image, axis=3)
            augmented_ds.append(augmented_image_sq)
    augmented_ds = np.asarray(augmented_ds)

    augmented_labels = len(augmented_ds)*[1]
    return augmented_ds, augmented_labels

def DatasetwithAugmentation(rname: list, fname: list):
    train_ds, val_ds, test_ds,train_labels, val_labels,test_labels = prep.WholeDataset(rname, fname)
    augmented_ds, augmented_labels = Augmentation(rname, fname)
    train_ds_aug = np.concatenate((train_ds,augmented_ds),axis=0)
    train_labels_aug = np.concatenate((train_labels,augmented_labels),axis=0)
    return train_ds_aug, train_labels_aug


#CLASS WEIGHTS
def DatasetWeights(rname: list, fname: list):
    nr_blind_events1, blind_events1 = prep.FindBlindEvents(rname[0],fname[0])
    nr_blind_events2, blind_events2 = prep.FindBlindEvents(rname[1],fname[1])
    nr_blind_events3, blind_events3 = prep.FindBlindEvents(rname[2],fname[2])

    nr_events1, nr_pixels1, nr_cams1, nr_tot_events1, nr_pixels_1_ev1 = prep.DimensionsData(fname[0])
    nr_events2, nr_pixels2, nr_cams2, nr_tot_events2, nr_pixels_1_ev2 = prep.DimensionsData(fname[1])
    nr_events3, nr_pixels3, nr_cams3, nr_tot_events3, nr_pixels_1_ev3 = prep.DimensionsData(fname[2])

    augmented_ds, augmented_labels = Augmentation(rname, fname)
    
    nr_blind_ev = len(nr_blind_events1) + len(nr_blind_events2) + len(nr_blind_events3) + len(augmented_ds)
    nr_tot_ev = nr_tot_events1 + nr_tot_events2 + nr_tot_events3 + len(augmented_ds)
    nr_not_blind_ev = nr_tot_ev - nr_blind_ev

    initial_bias = np.log(nr_blind_ev/nr_not_blind_ev)

    weights_0 = (1/nr_not_blind_ev)*(nr_tot_ev/2)
    weights_1 = (1/nr_blind_ev)*(nr_tot_ev/2)
    weights_classes = {0: weights_0, 1: weights_1}
    return initial_bias, weights_classes

def Training(rname: list, fname: list):
    train_ds_aug, train_labels_aug = DatasetwithAugmentation(rname, fname)
    train_ds, val_ds, test_ds,train_labels, val_labels,test_labels = prep.WholeDataset(rname, fname)
    initial_bias, weights_classes = DatasetWeights(rname, fname)

    input_shape = [32,32,32,1]
    epochs = 10
    #output_bias = keras.initializers.Constant(initial_bias)
    model = models.Sequential()
    #model.add(data_augmentation)
    model.add(layers.Conv2D(8,3,padding='same', activation='ReLU', input_shape=input_shape[1:]))
    model.add(layers.Conv2D(16,3,padding='same', activation='ReLU', input_shape=input_shape[1:]))
    model.add(layers.Conv2D(16,3,padding='same', activation='ReLU', input_shape=input_shape[1:]))
    #model.add(layers.Conv2D(32,3,padding='same', activation='sigmoid', input_shape=input_shape[1:]))
    model.add(layers.MaxPooling2D((32,32)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))#bias_initializer=output_bias

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
   
    #callbacks = [keras.callbacks.ModelCheckpoint("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/ultima_versione_buoni/tf_model_new/cams_model_new.keras")]
    history = model.fit(train_ds_aug, train_labels_aug, validation_data= (val_ds, val_labels), epochs=epochs, batch_size=32, class_weight=weights_classes)
    return model, history
    
model, history = Training(file_root_list, file_drdf_list)

saved_model = model.save("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/ultima_versione_buoni/tf_model_new/cams_model_new.keras")









































