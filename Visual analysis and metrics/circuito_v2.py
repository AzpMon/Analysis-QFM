import pennylane as qml
from pennylane import numpy as np
import torch


kernel_v2_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits)
create_params_k2 = lambda  layers,qubits : torch.randn(layers, qubits, requires_grad = True)
layers_v2 = 2
params_v2 = torch.tensor([[-0.1816, -0.5608],
         [ 0.7123,  1.5811]], requires_grad=False)
#Ansatz Circuit (Quantum Feature Map)
def ansatz_v2(qubits, layers,params,  dato):
    r = np.linalg.norm(dato)        
    for layer in range(layers):
        for qubit in range(qubits):
            qml.RY(phi = dato[qubit], wires = qubit)  
        for qubit in range(qubits-1):
                qml.CRY(phi = params[layer][qubit]*r, wires = [qubit, qubit+1])
        qml.CRY(phi = params[layer][qubit+1]*r, wires = [qubit+1, 0])

        
def make_measurment(layers, params,ansatz,Pauli_measurment,  x1, x2):
    @qml.qnode(device = kernel_v2_dev(qubits=4), interface = "torch")
    def calculate_kernel_v3():
        ansatz(qubits=4, layers=layers, params=params, dato = x1)
        qml.adjoint(ansatz)(qubits=4, layers=layers, params=params,dato = x2)
        return qml.probs(wires=range(4), op = Pauli_measurment)
    return calculate_kernel_v3()[0]