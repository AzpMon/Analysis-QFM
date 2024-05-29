import pennylane as qml
from pennylane import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"]=True
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


###################################################################################################################################################
def perform_optimization_clf(learning_rate, epochs, batch_size,qubits,layers,params,kernel_fn,
                             X_train, X_test, Y_train, Y_test,
                             title = "",
                             plot_kta = True, plot_accuracy = True):
    init_params = params.clone()
    print("   Optimization:")

    opt_res = optimization(learning_rate, epochs, batch_size,qubits,layers,params,
                                 kernel_fn,
                                 X=X_train,Y=Y_train, 
                                 title=title, 
                                 plot_kta=plot_kta)

    clf = SVM_classifier(qubits, layers,params.detach(),kernel_fn,X_train,X_test, Y_train,Y_test)
    init_params = params.clone()
    dicc = {'clf': clf[0],'params_init':init_params,'params_opt':opt_res[0], 'latest_kta':opt_res[1], 'test_acc':clf[1]}

    return dicc



###################################################################################################################################################
###################################################################################################################################################
############################################################### Forward function ###############################################################
def forward(X,qubits, layers, params,kernel_fn, assume_normalized_kernel=True):
    """Computes the Kernel Matrix associated with the respective circuit.

    Args:
        params (torch.tensor): Parameters of the Kernel circuit.
        kernel_fn (function): Kernel function.
        qubits (int): Number of qubits; the same as the dimension of the feature space.
        layers (int): Number of layers for the Quantum Kernel Circuit.
        assume_normalized_kernel (bool, optional): Normalization of the kernel (True/False).

    Returns:
        torch.tensor: The Kernel Matrix associated with the respective circuit, using the specified params, layers, and qubits."""
    Kernel = qml.kernels.square_kernel_matrix(X,
                                              kernel = lambda x1, x2 : kernel_fn(qubits=qubits, layers=layers, params=params, dato1=x1, dato2=x2),
                                            assume_normalized_kernel=assume_normalized_kernel)
    
    return Kernel
###################################################################################################################################################


############################################################### Cost function (KTA) ###############################################################
def cost_fn(Y_data,_kernel,rescale_class_labels=True):
    """Computes the Kernel-Target Alignment given the respective Kernel Matrix and "Ideal Kernel" for the label data.
    
    Args:
        Y_data (labels): Data used to create the Kernel and the "Ideal Kernel".
        _kernel (torch.tensor): Associated Kernel Matrix (Forward pass)
        rescale_class_labels (bool, optional): _description_. Defaults to True.

    Return:
        Scalar value representing the Kernel-Target Alignment.
    """
    #Se corrige el desbalanceo de las clases en caso de estarlo.

    classes = torch.unique(Y_data)
    if rescale_class_labels:
        nplus = torch.count_nonzero(Y_data == classes[0]).item()
        nminus = len(Y_data) - nplus
        # Ensure that neither nor nminous are cero
        if nplus != 0 and nminus != 0:
            _Y = torch.where(Y_data == classes[0], Y_data / nplus, Y_data / nminus)
        else:
            #If the Y_data consists only in one class
            raise ValueError("Error: Solo hay una clase.")
    else:
        _Y = Y_data

    Kernel_ideal = torch.outer(_Y, _Y)

    inner_product = torch.sum(_kernel * Kernel_ideal)
    norm = torch.sqrt(torch.sum(_kernel * _kernel) * torch.sum(Kernel_ideal * Kernel_ideal))
    kta = inner_product / norm
    return kta
###################################################################################################################################################


############################################################## Optimization function ##############################################################
def optimization(learning_rate, epochs, batch_size,qubits,layers,kernel_fn,params,X,Y, plot_kta = False, title=""):
    """Performs the optimization of the parameters with the KTA using the Adam optimizer.
    Args:
        learning_rate (float): Learning Rate.
        epochs (int): Num. epochs for the training loop.
        batch_size (int): Batch size.
        qubits (int): Num. Qubits; the same as the dimension of the Feature Space.
        layers (int): Num. Layers for the Kernel Circuit.
        params (torch.tensor): Parameters for the Kernel Circuit.
        kernel_fn (function): Kernel (classic) associated with the Kernel Circuit.
        X (torch.tensor): Input (feature) data for the training loop.
        Y (torch.tensor): Input labels for the training loop.
        plot_kta (bool, optional): Plot the KTA per epoch after the training loop. Defaults to False.

    Returns:
        array: [optimized parameters, max. KTA]"""
    kta_calculado=[]
    train_kta = []
    optimizer = torch.optim.Adam([params],lr = learning_rate)
    init_params = params.clone()

    

    for epoch in range(epochs):
        # Select batch_idx    
        batch_idx_np = np.random.choice(list(range(len(Y))), batch_size, requires_grad = False)
        batch_idx = torch.tensor(batch_idx_np, requires_grad=False)

        X_batch = X[batch_idx]
        Y_batch = Y[batch_idx]
        
        if len(Y_batch.unique()) == 2:
            # 1. Forward pass: Computes the predicted Kernel
            Kernel_pred = forward(X=X_batch, qubits=qubits, layers=layers, params=params, kernel_fn=kernel_fn, assume_normalized_kernel=True)
            
            # 2.Calculates loss
            loss = -cost_fn(Y_batch, Kernel_pred) #Negative, because the Pytorch's minimization
            kta_calculado.append(-loss.detach().item())
            
            # 3.Computes the Gradients
            loss.backward()
            
            # 4.Update the params
            optimizer.step()
            
            # 5.Reload the gradients
            optimizer.zero_grad()
            
            
            current_kernel = forward(X,qubits, layers, params,kernel_fn)
            current_kta = cost_fn(Y,current_kernel)
            train_kta.append(current_kta.detach())
            
            if (epoch+1) % 10 == 0:
                print(f"Layer(s) = {layers} ---- Epochs = {epoch+1} ---- Train KTA = {current_kta:.8f}")

    #Plot the KTA
    if plot_kta == True:
        plt.plot(train_kta,lw=0.8, color = "red", label = f'learning rate = {learning_rate}\nepochs = {epochs}\nbatch size = {batch_size}\nlayers = {layers}')
        plt.ylabel("KTA")
        plt.xlabel("Epochs")
        plt.title("KTA vs Epochs")
        plt.legend()
        plt.title(title)
        plt.grid()        

    dicc = {'params_init':init_params,'params_opt':params, 'latest_kta':current_kta}

    return dicc
###################################################################################################################################################


#Create the associated Kernel Matrix
def create_kernel_matrix(qubits, layers, params,kernel_fn,A,B):
    """Creates the Kernel matrix associated to the kernel_fn for the datasets A & B.
    Args:
        qubits (int): Num. Qubit
        layers (int): Num. layers
        params (torch.tensor): Kernel fn. params
        kernel_fn (function)
        A (torch.tensor): Dataset to make the Kernel
        B (torch.tensor): Dataset to make the kernel
    """
    return np.array([[kernel_fn(qubits,layers,params, a, b) for b in B] for a in A])


def accuracy(clf, X_test, Y_test):
    return accuracy_score(y_true=Y_test, y_pred=clf.predict(X_test))


#Creates the associates SVM Classifier (sklearn.SVM.SVC)
def SVM_classifier(qubits, layers,params,kernel_fn,X_train,X_test, Y_train,Y_test):
    """Returns a classififer associated to kernel matrix  (created with the fn. create_kernel_matrix)

    Args:
        qubits (int): Num. Qubits
        layers (int): Num. Layers
        params (torch.tensor): Kernel's parameters.
        X (torch.tensor): Input Data Set
        Y (torch.tensor): Label Data Set
    """
    clf = SVC(kernel=lambda X1, X2: create_kernel_matrix(qubits, layers,params,kernel_fn,X1, X2)).fit(X_train, Y_train)
    acc = accuracy(clf, X_test, Y_test)
    dicc = {'Clf':clf, "test_accuracy":acc}

          
    return dicc