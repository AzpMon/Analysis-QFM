from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import pennylane as qml
import numpy as np

def Kernel_OneQubit(ansatz,qubits, layers, params, dato1, dato2, kernel_qubit):
    """_summary_

    Args:
        ansatz (function): Optimized Quantum Feature Map
        qubits (int): Num. Qubits
        layers (int): Num. Layers (for the QFM)
        params (torch.tensor): Optimized params (...required_grad = False)
        dato1 (torch.tensor): x1
        dato2 (torch.tensor): x2
        kernel_qubit (int): The qubit that will be considered for the construction of the Kernel


    """
    dev_2Qubits = qml.device(name = "default.qubit", wires = qubits)   
    
    #Ansatz Circuit for density matrix(Quantum Feature Map)
    def density_matrix(x):
        ansatz(qubits, layers, params, x)
        return qml.density_matrix(wires = kernel_qubit)
    
    rho_1 = qml.QNode(func= lambda : density_matrix(dato1), device = dev_2Qubits, interface="torch")
    rho_2 = qml.QNode(func= lambda : density_matrix(dato2), device = dev_2Qubits, interface="torch") 
    
    #Se calcula el producto interno entre ambas matrices de densidad (Producto interno de Forbenius)
    kernel = 0
    for i in range(qubits):
        for j in range(qubits):
            kernel += rho_1()[i][j] * rho_2()[j][i]
    return kernel


def kernelMatrix_OneQubit(ansatz, qubits, layers, params,A,B,kernel_qubit):
    """Creates the Kernel matrix associated to the kernel_fn for the datasets A & B.
    Args:
        qubits (int): Num. Qubit
        layers (int): Num. layers
        params (torch.tensor): Kernel fn. params
        kernel_fn (function)
        A (torch.tensor): Dataset to make the Kernel
        B (torch.tensor): Dataset to make the kernel
    """
    return np.array([[Kernel_OneQubit(ansatz,qubits, layers, params, a, b, kernel_qubit) for b in B] for a in A])
    
    


def SVC_OneQubit(ansatz, qubits, layers,params,kernel_qubit,X_train,X_test, Y_train,Y_test, print_results = True):
    """Returns a classififer associated to kernel matrix  (created with the fn. create_kernel_matrix)

    Args:
        qubits (int): Num. Qubits
        layers (int): Num. Layers
        params (torch.tensor): Kernel's parameters.
        X (torch.tensor): Input Data Set
        Y (torch.tensor): Label Data Set
    """
    clf = SVC(kernel=lambda X1, X2: kernelMatrix_OneQubit(ansatz, qubits, layers, params,X1,X2,kernel_qubit)).fit(X_train, Y_train)
    y_predict = clf.predict(X_test)
    
    acc = accuracy_score(Y_test, y_predict)
    
    
    dicc = {'Clf':clf, "test_accuracy":acc}
    
    if print_results == True:
        print(f'Test_accuracy (for qubit {kernel_qubit}) = {round(acc,5)}')
        
    return dicc
