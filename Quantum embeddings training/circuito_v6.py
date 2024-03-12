import pennylane as qml
from pennylane import numpy as np
import torch

#Pennylane device
kernel_v6_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits )

#Parameter creation
create_params_k6 = lambda layers,qubits: torch.rand(2,layers,qubits, requires_grad=True)

#Ansatz Circuit (Quantum Feature Map)
def ansatz_v6(qubits, layers,params,  dato):
    for qubit in range(qubits):
        qml.Hadamard(wires=qubit)
    
    r = torch.linalg.vector_norm(dato)
    for layer in range(layers):
        #Entanglement via CNOTS
        for qubit in range(qubits):
            #Entanglement via CNOTS
            if qubit < qubits-1:
                qml.CNOT(wires=[qubit, qubit+1])
            else:
                qml.CNOT(wires=[qubits-1, 0])
        
        #RZ(r)-parametrized rotations    
        for qubit in range(qubits):
            param_rotY = params[0][layer][qubit]
            qml.RZ(phi =torch.tanh(r * param_rotY), wires = qubit)
            
        #Parametrized Phase-Shift(x_1^2, x_2^2)
        for qubit in range(qubits):
            param_shift = params[1][layer][qubit]
            qml.PhaseShift(phi=param_shift*pow(dato[qubit],2), wires=qubit)
        
#Kernel Circuit
def kernel_circuit_v6(qubits, layers, params, dato1, dato2, draw=False):
    @qml.qnode(device = kernel_v6_dev(qubits), interface = "torch")
    def calculate_kernel_v6():
        ansatz_v6(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz_v6)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel_v6)()
    return calculate_kernel_v6()


#Kernel function
kernel_v6 = lambda qubits,layers,params, dato1,dato2 : kernel_circuit_v6(qubits, layers, params, dato1, dato2)[0]