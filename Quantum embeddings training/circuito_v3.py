import pennylane as qml
from pennylane import numpy as np
import torch

#Pennylane device
kernel_v3_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits)

#Parameter creation
create_params_k3 = lambda layers: torch.rand(layers, requires_grad=True)

#Ansatz Circuit (Quantum Feature Map)
def ansatz_v3(qubits, layers, params, dato):
    op = qml.PauliZ(0)
    for qubit in range(1, qubits):
        op = op @ qml.PauliZ(qubit)
    for layer in range(layers):
        for qubit in range(qubits):
            qml.RY(phi = dato[qubit], wires = qubit)
        qml.exp(op, -1j*np.linalg.norm(dato)*params[layer])
            
#Kernel Circuit
def kernel_circuit_v3(qubits, layers, params, dato1, dato2, draw=False):
    @qml.qnode(device = kernel_v3_dev(qubits), interface = "torch")
    def calculate_kernel_v3():
        ansatz_v3(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz_v3)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel_v3)()
    return calculate_kernel_v3()


#Kernel function
kernel_v3 = lambda qubits,layers,params, dato1,dato2 : kernel_circuit_v3(qubits, layers, params, dato1, dato2)[0]
