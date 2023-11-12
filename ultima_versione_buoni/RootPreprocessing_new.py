import ROOT as root
import numpy as np
import Preprocessing_aug_prova2 as prep

def OpenRootFile(rname,fname):
    #sensor1.root è il file; ogni camera è un TTree
    input_file = root.TFile.Open(rname, "READ")
 
    cam_list = prep.CamList(fname)

    nr_photons_list_all_cams_list = []
    for cam in cam_list:
        nr_photons_list = []
        tree = input_file.Get(cam)
        entries = tree.GetEntries()
        for i in range(entries):
            n = tree.GetEntry(i)
            inner_photons = tree.innerPhotons
            nr_photons_list.append(inner_photons)
        nr_photons_list_all_cams_list.append(nr_photons_list)
        nr_photons_list_all_cams = np.array(nr_photons_list_all_cams_list)#ho creato una matrice 1000 colonne (nr eventi) x 54 righe (nr camere). I numeri che vediamo sono i numeri di fotoni che arrivano alla camera
    return nr_photons_list_all_cams

def EventListNumber(fname):
    file = prep.load_drdf(fname)
    ev_list = []
    for i in range(len(file)):
        ev_list.append(file[i][0])
    return ev_list

def column(matrix, i):
    column = [row[i] for row in matrix]
    return column

def RootDataDictionary(rname,fname):
    nr_photons_list_all_cams = OpenRootFile(rname,fname)
    ev_list = EventListNumber(fname)
    all_photons_in_ev = []
    for i in ev_list:
        nr_photons_in_ev = column(nr_photons_list_all_cams, i)#lista di fotoni in un determinato evento/sono le colonne delle matrici
        all_photons_in_ev.append(nr_photons_in_ev)#faccio una lista di queste liste di fotoni

    cam_list = prep.CamList(fname)

    final_root_data = []
    for i in range(len(all_photons_in_ev)):
        nr_photon_in_cam = []
        for j in range(len(all_photons_in_ev[i])):
            nr_photon_in_cam.append(all_photons_in_ev[i][j])
        dict_cam_ev = dict(zip(cam_list, nr_photon_in_cam))
        final_root_data.append((ev_list[i], dict_cam_ev))
    return final_root_data

def InnerPhotonsList(rname,fname):
    final_root_data = RootDataDictionary(rname, fname)
    inner_photons_list = []
    for i in range(len(final_root_data)):
        for cam in final_root_data[i][1].keys():
            inner_photons_list.append(final_root_data[i][1][cam])  
    inner_photons_list = np.asarray(inner_photons_list)
    return inner_photons_list

def CamStatesRootList(rname,fname):
    inner_photons_list = InnerPhotonsList(rname,fname)
    tot_photons = prep.SumPhotons(fname)
    ev_cam_state = []
    for i in range(len(inner_photons_list)):
        np.seterr(divide='ignore', invalid='ignore')
        ratio = inner_photons_list[i]/tot_photons[i]
        if ratio >= 0.1:
            value = 0
        elif ratio < 0.1:
            value = 1
        ev_cam_state.append(value)
    return ev_cam_state

def RootPreprocessing(rname, fname):
    return CamStatesRootList(rname, fname)