import pennylane as qml
from pennylane import numpy as np
import torch

#Pennylane device
kernel_v5_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits )

#Parameter creation
create_params_k5 = lambda layers,qubits: torch.rand(2,layers,qubits, requires_grad=True)

          
        
        
#Ansatz Circuit (Quantum Feature Map)
def ansatz_v5(qubits, layers,params,  dato):
    for qubit in range(qubits):
        qml.Hadamard(wires=qubit)
    
    r = np.linalg.norm(dato)
    for layer in range(layers):
        #Entanglement via CNOTS
        for qubit in range(qubits):
            #Entanglement via CNOTS
            if qubit < qubits-1:
                qml.CNOT(wires=[qubit, qubit+1])
            else:
                qml.CNOT(wires=[qubits-1, 0])
        
        #RY-parametrized rotatinos        
        for qubit in range(qubits):
            param_rotY = params[0][layer][qubit]
            qml.RZ(phi = r * param_rotY, wires = qubit)
            
        #Parametrized Phase-Shift  
        for qubit in range(qubits):
            param_shift = params[1][layer][qubit]
            qml.PhaseShift(phi=param_shift*pow(dato[qubit],2), wires=qubit)
        
#Kernel Circuit
def kernel_circuit_v5(qubits, layers, params, dato1, dato2, draw=False):
    dev = qml.device(name = "default.qubit", wires = qubits)

    @qml.qnode(device = kernel_v5_dev(qubits), interface = "torch")
    def calculate_kernel_v5():
        ansatz_v5(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz_v5)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel_v5)()
    return calculate_kernel_v5()


#Kernel function
kernel_v5 = lambda qubits,layers,params, dato1,dato2 : kernel_circuit_v5(qubits, layers, params, dato1, dato2)[0]
