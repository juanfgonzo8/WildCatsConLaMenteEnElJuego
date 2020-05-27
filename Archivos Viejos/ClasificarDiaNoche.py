##
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# Path JF Dia 1: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0affae-23d2-11e8-a6a3-ec086b02610b.jpg
# Path JF Noche 1: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0affb1-23d2-11e8-a6a3-ec086b02610b.jpg

# Path JF Dia 2: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0b01b5-23d2-11e8-a6a3-ec086b02610b.jpg
# Path JF Noche 2: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0b01b3-23d2-11e8-a6a3-ec086b02610b.jpg

# Path JF Dia 3: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0b02fa-23d2-11e8-a6a3-ec086b02610b.jpg
# Path JF Noche 3: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/5a0b02e6-23d2-11e8-a6a3-ec086b02610b.jpg

#Se copian los paths de las imagenes
def diaONoche():
    path_dia1 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0affae-23d2-11e8-a6a3-ec086b02610b.jpg'
    path_noche1 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0affb1-23d2-11e8-a6a3-ec086b02610b.jpg'

    path_dia2 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0b01b5-23d2-11e8-a6a3-ec086b02610b.jpg'
    path_noche2 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0b01b3-23d2-11e8-a6a3-ec086b02610b.jpg'

    path_dia3 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0b02fa-23d2-11e8-a6a3-ec086b02610b.jpg'
    path_noche3 = '/media/user_home2/vision2020_01/Data/iWildCam2019/iwildcam-2019-fgvc6/train_images/5a0b02e6-23d2-11e8-a6a3-ec086b02610b.jpg'

    #Se leen las imagenes
    imDia1 = cv2.imread(path_dia1)
    imNoche1 = cv2.imread(path_noche1)

    imDia2 = cv2.imread(path_dia2)
    imNoche2 = cv2.imread(path_noche2)

    imDia3 = cv2.imread(path_dia3)
    imNoche3 = cv2.imread(path_noche3)

    #Se cambian las imagenes a HSV
    imDia1 = cv2.cvtColor(imDia1,cv2.COLOR_BGR2HSV)
    imNoche1 = cv2.cvtColor(imNoche1,cv2.COLOR_BGR2HSV)

    imDia2 = cv2.cvtColor(imDia2,cv2.COLOR_BGR2HSV)
    imNoche2 = cv2.cvtColor(imNoche2,cv2.COLOR_BGR2HSV)

    imDia3 = cv2.cvtColor(imDia3,cv2.COLOR_BGR2HSV)
    imNoche3 = cv2.cvtColor(imNoche3,cv2.COLOR_BGR2HSV)

    #Se sacan los histogramas de Hue y Saturation
    histHueDia1 = np.histogram(imDia1[:,:,0],bins=100)
    histHueDia2 = np.histogram(imDia2[:,:,0],bins=100)
    histHueDia3 = np.histogram(imDia3[:,:,0],bins=100)

    histHueNoche1 = np.histogram(imNoche1[:,:,0],bins=100)
    histHueNoche2 = np.histogram(imNoche2[:,:,0],bins=100)
    histHueNoche3 = np.histogram(imNoche3[:,:,0],bins=100)

    histSatDia1 = np.histogram(imDia1[:,:,1],bins=100)
    histSatDia2 = np.histogram(imDia2[:,:,1],bins=100)
    histSatDia3 = np.histogram(imDia3[:,:,1],bins=100)

    histSatNoche1 = np.histogram(imNoche1[:,:,1],bins=100)
    histSatNoche2 = np.histogram(imNoche2[:,:,1],bins=100)
    histSatNoche3 = np.histogram(imNoche3[:,:,1],bins=100)

    histDia1 = np.concatenate((histHueDia1[0],histSatDia1[0]),0)
    histDia2 = np.concatenate((histHueDia2[0],histSatDia2[0]),0)
    histDia3 = np.concatenate((histHueDia3[0],histSatDia3[0]),0)

    histNoche1 = np.concatenate((histHueNoche1[0],histSatNoche1[0]),0)
    histNoche2 = np.concatenate((histHueNoche2[0],histSatNoche2[0]),0)
    histNoche3 = np.concatenate((histHueNoche3[0],histSatNoche3[0]),0)

    #Se entrena un SVM. Dia = 0, Noche = 1.
    x = [histDia1,histDia2,histDia3,histNoche1,histNoche2,histNoche3]
    y = [0,0,0,1,1,1]
    clf = svm.SVC()
    clf.fit(x,y)
    return clf

def probarDia(path_prueba,clf):
    #Se prueba una imagen
    imPrueba = cv2.imread(path_prueba)
    imPrueba = cv2.cvtColor(imPrueba,cv2.COLOR_BGR2HSV)

    histHuePrueba = np.histogram(imPrueba[:,:,0],bins=100)
    histSatPrueba = np.histogram(imPrueba[:,:,1],bins=100)
    histPrueba = np.concatenate((histHuePrueba[0],histSatPrueba[0]),0)

    pred = clf.predict([histPrueba])
    return pred

#Se grafican los histogramas
# plt.figure()
# plt.hist(imDia1[:,:,0].ravel())
# plt.hist(imDia2[:,:,0].ravel())
# plt.hist(imDia3[:,:,0].ravel())
#
# plt.figure()
# plt.hist(imNoche1[:,:,0].ravel())
# plt.hist(imNoche2[:,:,0].ravel())
# plt.hist(imNoche3[:,:,0].ravel())

# w = np.concatenate((imNoche1[:,:,0].ravel(),imNoche2[:,:,0].ravel(),imNoche3[:,:,0].ravel()))
# x = np.concatenate((imDia1[:,:,0].ravel(),imDia2[:,:,0].ravel(),imDia3[:,:,0].ravel()))

# plt.figure()
# # # plt.hist(w,bins=150)
# # # plt.hist(x,bins=150)

# w = np.concatenate((imNoche1[:,:,2].ravel(),imNoche2[:,:,2].ravel(),imNoche3[:,:,2].ravel()))
# x = np.concatenate((imDia1[:,:,2].ravel(),imDia2[:,:,2].ravel(),imDia3[:,:,2].ravel()))
#
# plt.figure()
# plt.hist(w,bins=150)
# plt.hist(x,bins=150)