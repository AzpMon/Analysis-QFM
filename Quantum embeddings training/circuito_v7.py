import pennylane as qml
import torch

#Pennylane device
kernel_v7_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits )

#Parameter creation
create_params_k7 = lambda layers,qubits: torch.rand(3,layers,qubits, requires_grad=True)

#Ansatz Circuit (Quantum Feature Map)
def ansatz_v7(qubits, layers,params,  dato):
    for qubit in range(qubits):
        qml.Hadamard(wires=qubit)
    
    r = torch.linalg.vector_norm(dato)
    for layer in range(layers):
        
        #CRY Entanglement (via sigmoid)
        for qubit in range(qubits):
            param_rotCRY = params[0][layer][qubit]
            #Entanglement via CRY
            if qubit < qubits-1:
                qml.CRY(phi = torch.pi * torch.sigmoid(r * param_rotCRY), wires = [qubit, qubit+1] )
            else:
                qml.CRY(phi = torch.pi * torch.sigmoid(r * param_rotCRY)  , wires=[qubits-1, 0])
        
        #RY rotations
        for qubit in range(qubits):
            param_RY = params[1][layer][qubit]
            qml.RY(phi = param_RY * r, wires = qubit)
    
        #Parametrized Phase-Shift  
        for qubit in range(qubits):
            param_shift = params[2][layer][qubit]
            qml.PhaseShift(phi=param_shift*pow(dato[qubit],2), wires=qubit)
        
#Kernel Circuit
def kernel_circuit_v7(qubits, layers, params, dato1, dato2, draw=False):
    @qml.qnode(device = kernel_v7_dev(qubits), interface = "torch")
    def calculate_kernel_v7():
        ansatz_v7(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz_v7)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel_v7)()
    return calculate_kernel_v7()


#Kernel function
kernel_v7 = lambda qubits,layers,params, dato1,dato2 : kernel_circuit_v7(qubits, layers, params, dato1, dato2)[0]