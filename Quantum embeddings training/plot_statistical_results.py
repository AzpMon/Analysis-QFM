import matplotlib.pyplot as plt 
import seaborn as sns 
from pennylane import numpy as np
import torch
from make_measurements import *

def plot_stats_results(ansatz,
                       qubits,
                       layers,
                       params,
                       Pauli_measurement,
                       Xclass,
                       measure_qubit,
                       Class,
                       observable,
                       color, 
                       GralTitle,
                       plot = True):
    equal_measurements=[]
    distinct_measurements=[]
    
    #Computes the measurments and inner product via the observable
    for x1 in Xclass:
        for x2 in Xclass:
            if torch.equal(x1, x2)==True:
                equal_measurements.append(make_measurements(ansatz,qubits, layers, params,Pauli_measurement, x1, x2).detach().numpy())
            else:
                distinct_measurements.append(make_measurements(ansatz,qubits, layers, params,Pauli_measurement, x1, x2).detach().numpy())
    equal_measurements = np.array(equal_measurements)
    distinct_measurements = np.array(distinct_measurements)
    
    res = {'equal_data': equal_measurements, 'distinict_data':distinct_measurements}
    
    
    if plot == True:
        #Plot histograms of the results
        title = f'Observable {observable}, Qubit {measure_qubit},Class {Class}\n{GralTitle} '
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        
        sns.histplot(equal_measurements, kde = True, color = color, ax = ax[0])
        ax[0].set_xlabel("Expected values \n (Measurements)")
        ax[0].grid()
        
        
        sns.histplot(distinct_measurements, kde = True, color = color, ax = ax[1])
        ax[1].set_xlabel("Inner Products \n (Different data points)")
        ax[1].grid()
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.show()    
        plt.close()
        
    return res


def plot_expected_values(ansatz,
                       qubits,
                       layers,
                       params,
                       Pauli_measurement,
                       Xclass,
                       measure_qubit,
                       Class,
                       observable,
                       color, 
                       GralTitle,
                       plot = True):
    exp_values=[]
        
    #Computes the measurments and inner product via the observable
    for x1 in Xclass:
        for x2 in Xclass:
            if torch.equal(x1, x2)==True:
                exp_values.append(make_measurements(ansatz,qubits, layers, params,Pauli_measurement, x1, x2).detach().numpy())
    exp_values = np.array(exp_values)
    
    if plot == True:
        title = f'Observable {observable}, Qubit {measure_qubit},Class {Class}\n{GralTitle} '
        sns.histplot(exp_values, kde = True, color = color)
        plt.title(title)
        plt.grid()

   
    res = {'exp_values': exp_values}
