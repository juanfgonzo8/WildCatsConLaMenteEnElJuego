import numpy as np
import cv2

# Funcion para CLAHE
# Recibe BGR y saca BGR
def miCLAHE(im):
    elClahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16)) #puede probarse (8,8)
    labs = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    labs[0] = elClahe.apply(labs[0])
    clahed = cv2.cvtColor(labs,cv2.COLOR_Lab2BGR)
    clahed = cv2.cvtColor(clahed, cv2.COLOR_BGR2RGB)
    return clahed

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