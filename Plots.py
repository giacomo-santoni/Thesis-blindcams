import Preprocessing2 as prep
import matplotlib.pyplot as plt

def PlotCamsImages(fname):
    file = prep.load_drdf(fname)
    all_images = prep.AllImages(fname)
    cam_list = prep.CamList(fname)
    for i in range(len(all_images)):
        #for cam in cam_list:
        for j in range(len(all_images[i])):
            plt.imshow(all_images[i][j], interpolation='none')
            plt.colorbar()
            plt.title(f"event {file[i][0]} on {cam_list[j]}")
            plt.show()

def PlotCamsScaledImages(fname):
    file = prep.load_drdf(fname)
    all_images_scaled = prep.MatricesList(fname)
    cam_list = prep.CamList()
    for i in range(len(all_images_scaled)):
        for j in range(len(all_images_scaled[i])):
            plt.imshow(all_images_scaled[i][j], interpolation='none')
            plt.colorbar()
            plt.title(f"event {file[i][0]} on {cam_list[j]}")
            plt.show()