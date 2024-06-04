import torch
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

#Creación de los datos y estandarización de los datos predictores
X,Y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

#Función de re-escalamiento en el intervalo [-pi, pi]
def escalamiento_datos(x, x_minimo, x_maximo):
    rescal_int = lambda s: 2*np.pi*s / (x_maximo-x_minimo)
    if rescal_int(x_maximo) < 3.15159:
        return rescal_int(x) + rescal_int(x_maximo)-np.pi
    else:
        return rescal_int(x) - rescal_int(x_maximo)+np.pi
np_escalamiento_datos = np.vectorize(escalamiento_datos)


#Re-escalamiento de los datos
series =  [[columna[j] for columna in X] for j in range(4)]
datos_rescalados = pd.DataFrame(X)
datos_rescalados['target']=Y
data = torch.tensor(datos_rescalados.values)
