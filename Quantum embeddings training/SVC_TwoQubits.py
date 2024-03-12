from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import pennylane as qml
import numpy as np

qml_device = lambda qubits: qml.device(name="default.qubit", wires = qubits)


def kernel_circuit(ansatz, qubits, layers, params, dato1, dato2, draw=False):
    @qml.qnode(device = qml_device(qubits), interface = "torch")
    def calculate_kernel():
        ansatz(qubits, layers, params, dato = dato1)
        qml.adjoint(ansatz)(qubits, layers, params, dato = dato2)
        return qml.probs(wires=range(qubits))
    if draw == True:
        qml.draw_mpl(calculate_kernel)()
    return calculate_kernel()

kernel_fn = lambda ansatz,qubits,layers,params,x1, x2 : kernel_circuit(ansatz, qubits, layers, params, x1, x2, draw=False)[0]

#Creates the Kernel matrix (associated to the )
def create_kernel_matrix(qubits, layers, params,ansatz,A,B):
    """Creates the Kernel matrix associated to the kernel_fn for the datasets A & B.
    Args:
        qubits (int): Num. Qubit
        layers (int): Num. layers
        params (torch.tensor): Kernel fn. params
        kernel_fn (function)
        A (torch.tensor): Dataset to make the Kernel
        B (torch.tensor): Dataset to make the kernel
    """
    
    
    return np.array([[kernel_fn(ansatz,qubits,layers,params, a, b) for b in B] for a in A])

def accuracy(clf, X_test, Y_test):
    return accuracy_score(y_true=Y_test, y_pred=clf.predict(X_test))

#Creates the associates SVM Classifier (sklearn.SVM.SVC)
def SVM_2Qubits(qubits, layers,params,ansatz,X_train,X_test, Y_train,Y_test, print_resutls=True):
    """Returns a classififer associated to kernel matrix  (created with the fn. create_kernel_matrix)

    Args:
        qubits (int): Num. Qubits
        layers (int): Num. Layers
        params (torch.tensor): Kernel's parameters.
        X (torch.tensor): Input Data Set
        Y (torch.tensor): Label Data Set
    """
    clf = SVC(kernel=lambda X1, X2: create_kernel_matrix(qubits, layers,params,ansatz,X1, X2)).fit(X_train, Y_train)
    acc = accuracy(clf, X_test, Y_test)
    dicc = {'Clf':clf, "test_accuracy":acc}
    
    if print_resutls == True:
        print(f'Test_accuracy (2 qubits) = {round(acc, 5)}')

    return dicc
