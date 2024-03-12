import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

# Generar datos
n_samples = 100
x_array, y_array = make_circles(n_samples, noise=0.07, random_state=42)
y_array = 2 * y_array - 1
y_array = np.array(y_array)
x_array = np.pi * x_array
x_array = np.array(x_array)

# Convertir a tensores de PyTorch
def convert_to_tensors(train_test_split_result):
    X_train, X_test, Y_train, Y_test = train_test_split_result
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor

X_train, X_test, Y_train, Y_test = convert_to_tensors(train_test_split(x_array, y_array, train_size=0.7, random_state=42))

# Realizar una división para el conjunto de datos
X_train_array, X_test_array, Y_train_array, Y_test_array = train_test_split(x_array, y_array, train_size=0.7)

# Crear DataFrame
datos = pd.DataFrame(data=x_array, columns=["X1", "X2"])
datos["Y"] = y_array
features = datos[["X1", "X2"]]
labels = datos["Y"]

etiqueta1 = pd.DataFrame(data=datos[datos.Y == 1])
etiqueta1 = etiqueta1.reset_index()
features1 = etiqueta1[["X1", "X2"]]
labels1 = etiqueta1["Y"]

etiqueta2 = pd.DataFrame(data=datos[datos.Y == -1])
etiqueta2 = etiqueta2.reset_index()
features2 = etiqueta2[["X1", "X2"]]
labels2 = etiqueta2["Y"]

datos_originales = datos.copy()
clase1 = datos_originales[datos_originales.Y == 1]
clase2 = datos_originales[datos_originales.Y == -1]
X_class1 = torch.tensor(clase1[["X1", "X2"]].values)
X_class2 = torch.tensor(clase2[["X1", "X2"]].values)

# Visualización de datos
res = input("Print data? [Y/N]")
if res == "Y":
    plt.figure(figsize=(8, 6))
    plt.scatter(x=features1["X1"], y=features1["X2"], c='green', label='Label 1')
    plt.scatter(x=features2["X1"], y=features2["X2"], c='cyan', label='Label -1')
    plt.title("Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid()
    plt.show()

