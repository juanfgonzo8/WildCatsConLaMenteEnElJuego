import numpy as np
import cv2
from matplotlib import pyplot as plt

# Funcion para CLAHE
# Recibe BGR y saca BGR

def miCLAHE(im):
    elClahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    labs = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    labs[:,:,0] = elClahe.apply(np.uint16(labs[:,:,0]))
    clahed = cv2.cvtColor(labs,cv2.COLOR_Lab2BGR)
    clahed = cv2.cvtColor(clahed, cv2.COLOR_BGR2RGB)
    return clahed

def imclahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = clahe.apply(np.uint16(l))
    l1 = np.zeros((l.shape[0],l.shape[1],1))
    a1 = np.zeros((a.shape[0], a.shape[1], 1))
    b1 = np.zeros((b.shape[0], b.shape[1], 1))
    l1[:,:,0] = l
    a1[:, :, 0] = a
    b1[:, :, 0] = b
    lab = cv2.merge((l1,a1,b1))
    bgr = cv2.cvtColor(lab.astype('float32'), cv2.COLOR_LAB2BGR)
    return bgr
#Funcion para Simple white balance
# Recibe BGR y saca BGR
def miWBsimple(im):
    elWB = cv2.xphoto.createSimpleWB()
    elWB.setP(0.4) #Se puede cambiar
    WBed = elWB.balanceWhite(im)
    WBed = cv2.cvtColor(WBed, cv2.COLOR_BGR2RGB)
    return WBed

# Funcion para Gray-world white balance
# Recibe BGR y saca BGR
def miWBgrayworld(im):
    elWB = cv2.xphoto.createGrayworldWB()
    elWB.setSaturationThreshold(0.9) #Se puede cambiar
    WBed = elWB.balanceWhite(im)
    WBed = cv2.cvtColor(WBed, cv2.COLOR_BGR2RGB)
    return WBed

# Funcion para learning-based automatic white balance
# Recibe BGR y saca BGR
def miWB_LB(im):
    elWB = cv2.xphoto.createLearningBasedWB()
    elWB.setSaturationThreshold(0.99) #Se puede cambiar
    WBed = elWB.balanceWhite(im)
    WBed = cv2.cvtColor(WBed, cv2.COLOR_BGR2RGB)
    return WBed

# path_nombre = 'D:/VISION/iwildcam-2019-fgvc6-NUEVO/train_images/5a0affc7-23d2-11e8-a6a3-ec086b02610b.jpg'
# elClahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # puede probarse (8,8)
# plt.figure()
# plt.imshow(miCLAHE(cv2.imread(path_nombre),elClahe))
# plt.title('CLAHE')
# plt.figure()
# plt.imshow(imclahe(cv2.imread(path_nombre),elClahe))
# plt.title('CLAHE2')
# plt.figure()
# plt.imshow(miWBsimple(cv2.imread(path_nombre)))
# plt.title('WB Simple')
#
# plt.figure()
# plt.imshow(miWBgrayworld(cv2.imread(path_nombre)))
# plt.title('WB Grayworld')
#
# plt.figure()
# plt.imshow(miWB_LB(cv2.imread(path_nombre)))
# plt.title('WB LB')