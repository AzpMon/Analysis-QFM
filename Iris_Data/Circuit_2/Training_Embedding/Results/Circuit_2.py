import pennylane as qml
from pennylane import numpy as np
import torch


kernel_v2_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits)
create_params_k2 = lambda  layers,qubits : torch.randn(layers, qubits, requires_grad = True)

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
def kernel_circuit_v2(qubits, layers,params, dato1, dato2, draw = False):
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

#Kernel function
kernel_v2 = lambda qubits,layers, params, dato1,dato2 : kernel_circuit_v2(qubits, layers,params,dato1, dato2)[0]
