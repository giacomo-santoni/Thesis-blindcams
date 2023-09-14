import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models
import Preprocessing as prep

saved_model = models.load_model("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/tf_model/cams_model.keras")
weights = saved_model.get_weights()

preprocessed_data = prep.Preprocessing("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/response.drdf")
preprocessed_data = np.asarray(preprocessed_data)

predictions = saved_model.predict(preprocessed_data)

preprocessed_data_matrix = preprocessed_data.reshape(1000,54,31,31)
predictions_matrix = predictions.reshape(1000,54)

file = prep.load_drdf("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/response.drdf")
cam_list = prep.CamList("/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/response.drdf")

#creo un dictionary come i dati drdf
ev_cams_predictions = []
for i in range(len(preprocessed_data_matrix)):
    cams_predictions = dict(zip(cam_list,predictions_matrix[i]))
    ev_cams_predictions.append((file[i][0],cams_predictions))

#scrivo il dict sul file
new_file = open("events_cams_predictions", "w")
new_file.write(str(ev_cams_predictions))
new_file.close()