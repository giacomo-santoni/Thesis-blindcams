import numpy as np
import Preprocessing_new as prep
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models

train_ds,val_ds,test_ds,train_labels,val_labels,test_labels = prep.PrepareDataTraining("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/sensors.root", "/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/response.drdf")

#weights for imbalanced dataset
initial_bias = np.log(756/53244)
weights_0 = (1/53244)*(54000/2)
weights_1 = (1/756)*(54000/2)
weights_classes = {0: weights_0, 1: weights_1}
weights_classes

#augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(31,31,1)),
    layers.RandomRotation(0.2),
    #layers.RandomZoom(0.1),
    layers.RandomTranslation(0.2,0.2)
  ]
)

#model
input_shape = [32,31,31,1]
#output_bias = keras.initializers.Constant(initial_bias)
model = models.Sequential()
#model.add(data_augmentation)
model.add(layers.Conv2D(8,3,padding='same', activation='ReLU', input_shape=input_shape[1:]))
model.add(layers.Conv2D(16,3,padding='same', activation='ReLU', input_shape=input_shape[1:]))
# model.add(layers.Conv2D(31,3,padding='same', activation='sigmoid', input_shape=input_shape[1:]))
# model.add(layers.Conv2D(31,3,padding='same', activation='sigmoid', input_shape=input_shape[1:]))
model.add(layers.MaxPooling2D((31,31)))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))#bias_initializer=output_bias


#model.build(input_shape=input_shape)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

epochs = 10
history = model.fit(train_ds, train_labels, validation_data= (val_ds, val_labels), epochs=epochs, batch_size=32, class_weight=weights_classes)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

results = model.evaluate(test_ds,test_labels)
print("test loss, test acc:", results)