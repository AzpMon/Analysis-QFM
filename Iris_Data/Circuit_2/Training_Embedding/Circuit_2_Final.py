import pennylane as qml
from pennylane import numpy as np
import torch

###############################################################################################
#Ansatz y kernel (generales)
kernel_v2_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits)

#Ansatz Circuit (Quantum Feature Map)
def ansatz_v2(qubits, layers,params,  dato):
    r = np.linalg.norm(dato)        
    for layer in range(layers):
        for qubit in range(qubits):
            qml.RY(phi = dato[qubit], wires = qubit)  
        for qubit in range(qubits-1):
                qml.CRY(phi = params[layer][qubit]*r, wires = [qubit, qubit+1])
        qml.CRY(phi = params[layer][qubit+1]*r, wires = [qubit+1, 0])
  
#Kernel Circuit
def kernel_v2(qubits, layers,params, dato1, dato2, draw = False):
    """
    Args:
        qubits : Número de qubits, coincide con la dimensión de los vectores de datos
        layers : Número de capas
        params : Tensos con los parámetros del circuito (Kernel)
        data1  : dato 1 (para la creación del Kernel)
        data2  : dato 2 (para la creación del kernel)
        draw   : Para dibujar el circuito con los parámetros y datos dados, False por default.
    """
    @qml.qnode(device = kernel_v2_dev(qubits), interface = "torch")
    def calculate_kernel_v2():
        ansatz_v2(qubits, layers,params, dato1)
        qml.adjoint(ansatz_v2)(qubits, layers,params, dato2)
        return qml.probs(wires = range(qubits))
        
    return calculate_kernel_v2() 

#Make Measurment
def make_measurment(layers, params,ansatz,Pauli_measurment,  x1, x2):
    @qml.qnode(device = kernel_v2_dev(qubits=4), interface = "torch")
    def calculate_kernel_v3():
        ansatz(qubits=4, layers=layers, params=params, dato = x1)
        qml.adjoint(ansatz)(qubits=4, layers=layers, params=params,dato = x2)
        return qml.probs(wires=range(4), op = Pauli_measurment)
    return calculate_kernel_v3()[0]
###############################################################################################



######################################## Kernel 1 capa ########################################
#Kernel 01
params_1layers_01  = torch.tensor([-1.3482,  0.3438,  0.0083,  6.9057], requires_grad=False)
def ansatz_1layers_02(x):
    return  ansatz_v2(qubits=4, layers=1,params=params_1layers_01, dato = x)
#Kernel 12
params_1layers_12  = torch.tensor([-0.1496, -0.1398, -1.2082, -0.8108], requires_grad=False)
def ansatz_1layers_20(x):
    return  ansatz_v2(qubits=4, layers=1,params=params_1layers_12, dato = x)
#Kernel 20
params_1layers_20 = torch.tensor([-1.6985, -0.1673, -1.2978,  0.9400], requires_grad=False)
def ansatz_1layers_20(x):
    return  ansatz_v2(qubits=4, layers=1,params=params_1layers_20, dato = x)


######################################## Kernel 2 capas ########################################
#Kernel 01
params_2layers_01 = torch.tensor([[-1.8356,  0.1275, -3.2358,  2.3315],
         [ 2.4469,  0.2466, -3.0476, -1.3541]], requires_grad=False)
def ansatz_2layers_01(x):
    return  ansatz_v2(qubits=4, layers=2,params=params_2layers_01, dato = x)

#Kernel 12
params_2layers_12 = torch.tensor([[-1.8932,  2.3739,  1.7020, -2.0017],
         [-0.4359,  0.3168,  0.7401, -0.3023]], requires_grad=False)
def ansatz_2layers_12(x):
    return  ansatz_v2(qubits=4, layers=2,params=params_2layers_12, dato = x)

#Kernel 20
params_2layers_20 = torch.tensor([[-2.3540, -0.3696, -0.6999, -0.3996],
         [-1.2611, -1.1652, -0.6331, -0.8964]], requires_grad=False)
def ansatz_2layers_20(x):
    return  ansatz_v2(qubits=4, layers=2,params=params_2layers_20, dato = x)
################################################################################################


######################################## Kernel 3 capas ########################################
#Kernel 01
params_3layers_01 = torch.tensor([[ 2.9623, -0.2286, -1.3422,  2.3956],
         [ 3.0711,  2.9154, -0.0306, -2.1503],
         [-1.7235, -1.9817,  0.1852,  0.0263]], requires_grad=False)
def ansatz_3layers_01(x):
    return  ansatz_v2(qubits=4, layers=3,params=params_3layers_01, dato = x)

#Kernel 12
params_3layers_12 = torch.tensor([[-1.4025,  0.7426, -1.2352, -1.7183],
         [ 2.5227,  2.1812,  0.5861, -0.9121],
         [ 0.0653, -0.1261, -0.2155, -0.7657]], requires_grad=False)
def ansatz_3layers_12(x):
    return  ansatz_v2(qubits=4, layers=3,params=params_3layers_12, dato = x)

#Kernel 20
params_3layers_20=torch.tensor([[ 2.0612,  1.9870, -1.5497, -2.2230],
         [-0.3397,  0.0154, -1.3658,  0.2703],
         [-2.8656, -0.8892,  1.6216,  0.1945]], requires_grad=False)
def ansatz_3layers_20(x):
    return  ansatz_v2(qubits=4, layers=3,params=params_3layers_20, dato = x)
################################################################################################

######################################## Kernel 4 capas ########################################
#Kernel 01
params_4layers_01 = torch.tensor([[ 1.6127, -0.5819, -0.5743,  0.9243],
         [-0.1322,  0.3557,  1.5869,  0.2718],
         [-1.8247, -0.3342,  2.0580,  3.3858],
         [-0.0888,  1.5733, -0.2166, -0.5001]], requires_grad=False)
def ansatz_4layers_01(x):
    return  ansatz_v2(qubits=4, layers=4,params=params_4layers_01, dato = x)

#Kernel 12
params_4layers_12 = torch.tensor([[-1.8875, -1.8171,  1.0721,  0.0945],
         [-2.7552,  1.1530, -0.6033,  0.0553],
         [ 2.1404, -2.6744,  2.0857,  0.1950],
         [ 0.1954, -0.2516, -0.8069,  0.0248]], requires_grad=True)
def ansatz_4layers_12(x):
    return  ansatz_v2(qubits=4, layers=4,params=params_4layers_12, dato = x)

#Kernel 20  
params_4layers_20=torch.tensor([[ 0.0321, -2.9655,  2.1133,  1.2895],
         [-0.0715,  2.5225,  0.1952, -0.8970],
         [ 0.4744, -0.0593, -0.4529,  1.7432],
         [ 3.5083, -1.3877, -1.9547, -0.7768]], requires_grad=False)
def ansatz_4layers_20(x):
    return  ansatz_v2(qubits=4, layers=4,params=params_4layers_20, dato = x)
################################################################################################
