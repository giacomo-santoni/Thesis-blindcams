import drdf
import numpy as np
import RootPreprocessing_new as rooprep
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


#DRDF FILE - SIMULATED DATA

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

file_drdf_list = ["/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/initial_data_good/response.drdf","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data/response_pde25.drdf","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data2/response.drdf"]
file_root_list = ["/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/initial_data_good/sensors.root","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data/sensors_pde25.root","/Users/giacomosantoni/Desktop/TESI/Progetto_ML/blindcams/data/new_data2/sensors.root"]

def CamList(fname):
    cam_list = []
    file = load_drdf(fname)
    for cam in file[0][1]:
        cam_list.append(cam)
    return cam_list

#creo una matrice 1000 righe x 54 colonne. ogni elemento è una camera 31x31
def AllImages(fname): 
    all_images_list = []
    cam_list = CamList(fname)
    file = load_drdf(fname)
    for i in range(len(file)):
        ph_matrix = []
        for cam in cam_list:
            ph_matrix.append(file[i][1][cam])
        all_images_list.append(ph_matrix)
    all_images = np.array(all_images_list)
    return all_images

def DimensionsData(fname):
    all_images = AllImages(fname)
    cam_list = CamList(fname)
    nr_events = len(all_images)
    nr_pixels = len(all_images[0][0])
    nr_cams = len(cam_list)
    nr_tot_events = nr_cams*nr_events
    nr_pixels_1_ev = nr_pixels*nr_pixels*nr_cams
    return nr_events, nr_pixels, nr_cams, nr_tot_events, nr_pixels_1_ev

#creo una matrice di 1000 righe e 51894 colonne. Così per ogni evento, la riga corrispondente presenta tutti i pixel di tutte le camere
def PixelsAllCamsPerEvents(fname):
    all_images = AllImages(fname)
    nr_events, nr_pixels, nr_cams, nr_tot_events, nr_pixels_1_ev = DimensionsData(fname)
    pixels_to_scale = []
    for images_all_cams_per_ev in all_images: 
        pixels_all_cams_per_ev = images_all_cams_per_ev.flatten()
        pixels_to_scale.append(pixels_all_cams_per_ev)
    pixels_to_scale = np.asarray(pixels_to_scale)
    pixels_to_scale_matrix = pixels_to_scale.reshape(nr_events,nr_pixels_1_ev,1)
    return pixels_to_scale_matrix

#applico lo scaling su vettori colonne che sono le camere 31x31 appiattite
def ScalingData(fname):
    pixels_to_scale_matrix = PixelsAllCamsPerEvents(fname)
    nr_events, nr_pixels, nr_cams, nr_tot_events, nr_pixels_1_ev = DimensionsData(fname)
    photons_scaled_all_ev = []
    for pixels in pixels_to_scale_matrix:
        transformer = RobustScaler().fit(pixels)
        photons_scaled_in_cam = transformer.transform(pixels)
        photons_scaled_all_ev.append(photons_scaled_in_cam)
    photons_scaled_all_ev_matrix = np.asarray(photons_scaled_all_ev)
    all_images_scaled = photons_scaled_all_ev_matrix.reshape(nr_events,nr_cams,nr_pixels,nr_pixels)
    return all_images_scaled

#ora ho una matrice 1000righex54colonne dove un elemento è una matrice 31x31. Per essere consistente con i dati di root la voglio trasformare in un array di 54000 elementi
def Flattening(fname):
    all_images_scaled = ScalingData(fname)
    all_images_scaled_1d = []
    for sublist in all_images_scaled:
        all_images_scaled_1d.extend(sublist)
    return all_images_scaled_1d

def Reshaping(fname):
    all_images_scaled_1d_reshaped = []
    all_images = AllImages(fname)
    all_images_scaled_1d = Flattening(fname)
    nr_events, nr_pixels, nr_cams, nr_tot_events, nr_pixels_1_ev = DimensionsData(fname)
    for data in all_images_scaled_1d:
        data_new = data.reshape(nr_pixels,nr_pixels,1)
        all_images_scaled_1d_reshaped.append(data_new)
    return all_images_scaled_1d_reshaped

def Preprocessing(fname):
    load_drdf(fname)
    CamList(fname)
    AllImages(fname)
    PixelsAllCamsPerEvents(fname)
    ScalingData(fname)
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

def PrepareDataSingleFile(rname, fname):
    train_ds, val_ds, test_ds, train_inner_ph, val_inner_ph, test_inner_ph = SplitDataset(rname,fname)
    nr_events, nr_pixels, nr_cams, nr_tot_events, nr_pixels_1_ev = DimensionsData(fname)

    train_labels_bool = train_inner_ph > 5
    train_labels = train_labels_bool.astype(int)

    val_labels_bool = val_inner_ph > 5
    val_labels = val_labels_bool.astype(int)

    test_labels_bool = test_inner_ph > 5
    test_labels = test_labels_bool.astype(int)

    train_ds = train_ds.reshape(train_ds.shape[0], nr_pixels,nr_pixels, 1)
    val_ds = val_ds.reshape(val_ds.shape[0], nr_pixels, nr_pixels, 1)
    test_ds = test_ds.reshape(test_ds.shape[0],nr_pixels,nr_pixels,1)
    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels

def WholeDataset(rname: list,fname: list):
    #dataset1
    train_ds1, val_ds1, test_ds1, train_labels1, val_labels1, test_labels1 = PrepareDataSingleFile(rname[0],fname[0])
    #dataset2
    train_ds2, val_ds2, test_ds2, train_labels2, val_labels2, test_labels2 = PrepareDataSingleFile(rname[1],fname[1])
    #dataset3
    train_ds3, val_ds3, test_ds3, train_labels3, val_labels3, test_labels3 = PrepareDataSingleFile(rname[2],fname[2])

    train_ds = np.concatenate((train_ds1, train_ds2, train_ds3), axis=0)
    val_ds = np.concatenate((val_ds1, val_ds2, val_ds3), axis=0)
    test_ds = np.concatenate((test_ds1, test_ds2, test_ds3), axis=0)

    train_labels = np.concatenate((train_labels1, train_labels2, train_labels3), axis=0)
    val_labels = np.concatenate((val_labels1, val_labels2, val_labels3), axis=0)
    test_labels = np.concatenate((test_labels1, test_labels2, test_labels3), axis=0)
    return train_ds, val_ds, test_ds,train_labels, val_labels,test_labels

def FindBlindEvents(rname,fname):
    ev_cam_state = rooprep.RootPreprocessing(rname,fname)
    data_prep = Preprocessing(fname)
    n = 0
    nr_blind_events = []
    blind_events = []
    for i in ev_cam_state:
        if i == 1:
            nr_blind_events.append(n)
        n+=1
    for i in nr_blind_events:
        if np.max(i)>100:
            blind_events.append(data_prep[i])
    return nr_blind_events, blind_events