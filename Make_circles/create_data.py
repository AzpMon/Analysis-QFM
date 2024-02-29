from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

#Generate the data and plot it
n_samples=100
x_array, y_array = make_circles(n_samples,noise=0.07,random_state=42)
y_array = 2*y_array -1
y_array = np.array(y_array)
x_array = np.pi * x_array 
x_array = np.array(x_array)
res = input("Print data? [Y/N]")
if res == "Y":
    plt.scatter(x=x_array[:, 0],  y=x_array[:, 1], c=y_array)
    plt.title("Data"); plt.grid()


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

X_train, X_test, Y_train, Y_test = convert_to_tensors(train_test_split(x_array,y_array,train_size = 0.7, random_state=42))

#Makes a split for the dataset
X_train_array, X_test_array, Y_train_array, Y_test_array = train_test_split(x_array,y_array,train_size=0.7)




#DataFrame
datos = pd.DataFrame(data=x_array, columns = ["X1", "X2"])
datos["Y"]= y_array
features = datos[["X1", "X2"]]
labels = datos["Y"] 

etiqueta1 = pd.DataFrame(data = datos[datos.Y == 1]); etiqueta1= etiqueta1.reset_index()
features1 = etiqueta1[["X1", "X2"]]
labels1 = etiqueta1["Y"]

etiqueta2 = pd.DataFrame(data = datos[datos.Y == -1]); etiqueta2= etiqueta2.reset_index()
features2 = etiqueta2[["X1", "X2"]]
labels2 = etiqueta2["Y"]


datos_originales = datos.copy()
clase1 = datos_originales[datos_originales.Y == 1]
clase2 = datos_originales[datos_originales.Y == -1]
X_class1 = torch.tensor(clase1[["X1", "X2"]].values)
X_class2 = torch.tensor(clase2[["X1", "X2"]].values)
