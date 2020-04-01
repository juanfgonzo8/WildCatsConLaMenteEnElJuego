import pandas as pd
import numpy as np

#Path JF anots: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train.csv
#Path JF train: /Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/
csv_train = pd.read_csv('/Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train.csv')
path_train = '/Users/johngonzalez/Desktop/Trabajos JF/Documents/U/Octavo Semestre/Vision Artificial/Proyecto/iwildcam-2019-fgvc6/train_images/'
anots = csv_train['category_id'].to_numpy()
arch = csv_train['file_name'].tolist()
unics, conts = np.unique(anots, return_counts=True)
num_min = np.min(conts[:-1])
subset = []
for i in unics:
    pos = np.where(anots==i)[0]
    #print(pos)
    selec = pos[0:num_min]
    noms = [arch[j] for j in selec]
    subset.append(noms)
lista = []
for i in subset:
    for j in i:
        lista.append(path_train+j)