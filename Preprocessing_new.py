import drdf
import matplotlib.pyplot as plt
import numpy as np
import RootPreprocessing_new as rooprep
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_drdf(fname):
  reader = drdf.DRDF()
  reader.read(fname)
  events = []
  for run in reader.runs:
    for event in reader.runs[run]:
      hits_map = dict()
      for cam, img in reader.runs[run][event].items():
        amplitude = img.pixels[:, :, 0] 
        time = img.pixels[:, :, 1]
        hits_map[cam] = amplitude
      events.append((event, hits_map))
  return events

def CamList(fname):
    cam_list = []
    file = load_drdf(fname)
    for cam in file[0][1]:
        cam_list.append(cam)
    return cam_list

def AllImages(fname): 
    all_images = []
    cam_list = CamList(fname)
    file = load_drdf(fname)
    for i in range(len(file)):
        ph_matrix = []
        for cam in cam_list:
            ph_matrix.append(file[i][1][cam])
        all_images.append(ph_matrix)
    return all_images

def FlattenSingleCam(fname):
    all_photons_single_cam = []
    all_images = AllImages(fname)
    for i in range(len(all_images)):
        photons_single_cam = []
        for j in range(len(all_images[0])):
            photons_single_cam.append(all_images[i][j].flatten())
        all_photons_single_cam.append(photons_single_cam)
    return all_photons_single_cam

#applico lo scaling su vettori colonne che sono le camere 31x31 appiattite
def ScalingData(fname):
    all_photons_single_cam_scaled = []
    all_images = AllImages(fname)
    all_photons_single_cam = FlattenSingleCam(fname)
    for i in range(len(all_images)):
        ph_matrix_scaled = []
        for j in range(len(all_images[0])):
            transformer = RobustScaler().fit(all_photons_single_cam[i][j].reshape(-1,1))
            scaled_data = transformer.transform(all_photons_single_cam[i][j].reshape(-1,1))
            ph_matrix_scaled.append(scaled_data)
        all_photons_single_cam_scaled.append(ph_matrix_scaled)
    return all_photons_single_cam_scaled

def MatricesList(fname):
    all_images = AllImages(fname)
    dim_matrix = len(all_images[0][0])
    all_photons_single_cam_scaled = ScalingData(fname)
    #dentro la matrice 1000righex54colonne, prendo un elemento, che è un array di 961 pixels, e lo trasformo in una matrice 31x31
    all_images_scaled = []
    index = 0
    for i in range(len(all_images)):
        all_images_scaled_per_ev = []
        for j in range(len(all_images[0])):
            matrix = np.array(all_photons_single_cam_scaled[i][j][index:index + dim_matrix * dim_matrix]).reshape((dim_matrix, dim_matrix))
            all_images_scaled_per_ev.append(matrix)
        all_images_scaled.append(all_images_scaled_per_ev)
    return all_images_scaled

#ora ho una matrice 1000righex54colonne dove un elemento è una matrice 31x31. Per essere consistente con i dati di root la voglio trasformare in un array di 54000 elementi
def Flattening(fname):
    all_images_scaled = MatricesList(fname)
    all_images_scaled_1d = []
    for sublist in all_images_scaled:
        all_images_scaled_1d.extend(sublist)
    return all_images_scaled_1d

def Reshaping(fname):
    all_images_scaled_1d_reshaped = []
    all_images = AllImages(fname)
    all_images_scaled_1d = Flattening(fname)
    for data in all_images_scaled_1d:
        data_new = data.reshape(len(all_images[0][0]),len(all_images[0][0]),1)
        all_images_scaled_1d_reshaped.append(data_new)
    return all_images_scaled_1d_reshaped

def Preprocessing(fname):
    load_drdf(fname)
    CamList(fname)
    AllImages(fname)
    FlattenSingleCam(fname)
    ScalingData(fname)
    MatricesList(fname)
    Flattening(fname)
    return Reshaping(fname)

def SplitDataset(rname,fname):
    all_images_scaled_1d = Flattening(fname)
    inner_photons_list = rooprep.InnerPhotonsList(rname,fname)
    #ev_cam_state = rooprep.CamStatesRootList(rname, fname)
    t_ds, val_ds, t_inner_ph, val_inner_ph = train_test_split(all_images_scaled_1d, inner_photons_list, train_size=0.9, random_state=42)
    train_ds, test_ds, train_inner_ph, test_inner_ph = train_test_split(t_ds, t_inner_ph, train_size=0.9, random_state=42)

    train_ds = np.asarray(train_ds)
    val_ds = np.asarray(val_ds)
    train_inner_ph = np.asarray(train_inner_ph) 
    val_inner_ph = np.asarray(val_inner_ph)
    test_ds = np.asarray(test_ds) 
    test_inner_ph = np.asarray(test_inner_ph)

    return train_ds, val_ds, test_ds, train_inner_ph, val_inner_ph, test_inner_ph

def PrepareDataTraining(rname, fname):
    train_ds, val_ds, test_ds, train_inner_ph, val_inner_ph, test_inner_ph = SplitDataset(rname,fname)
    all_images_scaled_1d = Flattening(fname)

    train_labels_bool = train_inner_ph > 5
    train_labels = train_labels_bool.astype(int)

    val_labels_bool = val_inner_ph > 5
    val_labels = val_labels_bool.astype(int)

    test_labels_bool = test_inner_ph > 5
    test_labels = test_labels_bool.astype(int)

    train_ds = train_ds.reshape(int(0.9*0.9*len(all_images_scaled_1d)), 31, 31, 1)
    val_ds = val_ds.reshape(int(0.1*len(all_images_scaled_1d)), 31, 31, 1)
    test_ds = test_ds.reshape(int(0.1*0.9*len(all_images_scaled_1d)),31,31,1)
    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels