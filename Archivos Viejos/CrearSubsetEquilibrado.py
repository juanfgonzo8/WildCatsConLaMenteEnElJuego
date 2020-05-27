import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#Path JF anots: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train.csv
#Path JF train: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/
def sacarPaths(path_csv,path_train1):
    csv_train = pd.read_csv(path_csv)
    path_train = path_train1
    anots = csv_train['category_id'].to_numpy()
    arch = csv_train['file_name'].tolist()
    unics, conts = np.unique(anots, return_counts=True)
    num_min = np.min(conts[:-1])
    subset = []
    subsetcat=[]
    for i in unics:
        pos = np.where(anots==i)[0]
        #print(pos)
        #selec = pos[0:num_min]
        selec = pos[0:90]
        noms = [arch[j] for j in selec]
        cats=[anots[j] for j in selec]
        subsetcat.append(cats)
        subset.append(noms)
    lista = []

    for i in subset:
        for j in i:
            lista.append(path_train+j)
    listnom=[]
    listcat=[]
    listnom=[]
    for i in subset:
        for j in i:
             listnom.append(j)
    for i in subsetcat:
        for j in i:
             listcat.append(j)
    X_train, X_test, y_train, y_test = train_test_split(listnom, listcat, test_size = 0.3, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test
