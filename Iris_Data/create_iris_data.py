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
datos_rescalados = pd.DataFrame(data=Y, columns=["target"])
for j in range(4):
    serie = np.array(series[j])
    datos_rescalados[load_iris().feature_names[j]] = np_escalamiento_datos(serie, serie.min(), serie.max())
    
#Convert to tensors
def convert_to_tensors(train_test_split_result):
    """
    Convierte los conjuntos de entrenamiento y prueba de train_test_split a tensores de PyTorch.

    Args:
        train_test_split_result: Resultado de train_test_split que incluye X_train, X_test, Y_train, Y_test.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): Tensores de PyTorch para X_train, X_test, Y_train y Y_test.
    """
    X_train, X_test, Y_train, Y_test = train_test_split_result
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)  
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)    

    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor


#División en datos de prueba y entrenamiento (total)
X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(
                                                datos_rescalados.drop(columns="target"), datos_rescalados["target"], train_size=0.7, random_state=42)




X_train, X_test, Y_train, Y_test = train_test_split( datos_rescalados.drop(columns="target").values,datos_rescalados["target"].values,train_size=0.7, random_state=42)


#División en datos para la clasificación One-to-One
#Clase 0 vs 1
clase_0_vs_1 = pd.concat([datos_rescalados[datos_rescalados.target == 0], datos_rescalados[datos_rescalados.target == 1]], ignore_index=True)
X_train_01, X_test_01, Y_train_01, Y_test_01 =convert_to_tensors(
    train_test_split(clase_0_vs_1.drop(columns="target").values, clase_0_vs_1["target"].values, train_size=0.7, random_state=42)
    )
#Clase 1 vs 2
clase_1_vs_2 = pd.concat([datos_rescalados[datos_rescalados.target == 1], datos_rescalados[datos_rescalados.target == 2]], ignore_index=True)
X_train_12, X_test_12, Y_train_12, Y_test_12 =convert_to_tensors(
    train_test_split(clase_1_vs_2.drop(columns="target").values, clase_1_vs_2["target"].values, train_size=0.7, random_state=42)
    )

#Clase 2 vs 0
clase_2_vs_0 = pd.concat([datos_rescalados[datos_rescalados.target == 2], datos_rescalados[datos_rescalados.target == 0]], ignore_index=True)
X_train_20, X_test_20, Y_train_20, Y_test_20 =convert_to_tensors(
    train_test_split(clase_2_vs_0.drop(columns="target").values, clase_2_vs_0["target"].values, train_size=0.7, random_state=42)
    )
