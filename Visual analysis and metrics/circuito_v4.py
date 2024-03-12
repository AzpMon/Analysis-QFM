import pennylane as qml
from pennylane import numpy as np
import torch


#Optimized params
layers_v4 = 2
params_v4 = torch.tensor([[[0.5869, 1.2154],
          [0.4606, 1.9823]],
 
         [[0.3906, 0.0915],
          [0.4860, 0.8554]]], requires_grad=False)


#Pennylane device
kernel_v4_dev = lambda qubits: qml.device(name="default.qubit", wires = qubits )

#Parameter creation
create_params_k4 = lambda layers,qubits: torch.rand(2,layers,qubits, requires_grad=True)

          
        
        
#Ansatz Circuit (Quantum Feature Map)
def ansatz_v4(qubits, layers,params,  dato):
    for qubit in range(qubits):
        qml.Hadamard(wires=qubit)
    
    r = np.linalg.norm(dato)
    for layer in range(layers):
        for qubit in range(qubits):
            #Circuit Params
            param_entangl = params[0][layer][qubit]
            param_rotZ = params[1][layer][qubit]
            
            if qubit<qubits-1:
                #Q. Gates
                qml.CRY(phi = r *param_entangl, wires = [qubit, qubit+1])
                qml.RZ(phi = dato[qubit]*param_rotZ, wires = qubit+1)
            else:
                qml.CRY(phi = r*param_entangl , wires = [qubit, 0])
                qml.RZ(phi = dato[qubit]*param_rotZ, wires = 0)

    

            
#Kernel Circuit
def kernel_circuit_v4(qubits, layers, params, dato1, dato2, draw=False):
    @qml.qnode(device = kernel_v4_dev(qubits), interface = "torch")
    def calculate_kernel_v4():
        ansatz_v4(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz_v4)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel_v4)()
    return calculate_kernel_v4()


#Kernel function
kernel_v4 = lambda qubits,layers,params, dato1,dato2 : kernel_circuit_v4(qubits, layers, params, dato1, dato2)[0]
