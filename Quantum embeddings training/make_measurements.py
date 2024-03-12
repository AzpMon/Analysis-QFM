import pennylane as qml
import numpy as np
def make_measurements(ansatz,qubits, layers, params, dato):
    dev = qml.device(name = "default.qubit", wires = qubits)
    #QUBIT 0
    def circuito_expX_Q0():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliX(0))
    expX_Q0 = qml.QNode(func = circuito_expX_Q0, device =dev, interface="torch")


    def circuito_expY_Q0():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliY(0))
    expY_Q0  = qml.QNode(func = circuito_expY_Q0, device =dev, interface="torch")


    def circuito_expZ_Q0():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliZ(0))
    expZ_Q0  = qml.QNode(func = circuito_expZ_Q0, device =dev, interface="torch")

    #QUBIT 1
    def circuito_expX_Q1():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliX(1))
    expX_Q1  = qml.QNode(func = circuito_expX_Q1, device =dev, interface="torch")


    def circuito_expY_Q1():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliY(1))
    expY_Q1 = qml.QNode(func = circuito_expY_Q1, device =dev, interface="torch")


    def circuito_expZ_Q1():
        ansatz(qubits, layers,params,dato)
        return qml.expval(qml.PauliZ(1))
    expZ_Q1 = qml.QNode(func = circuito_expZ_Q1, device =dev, interface="torch")
    
    
    res = {'qubit0':[expX_Q0(), expY_Q0(), expZ_Q0()], 'qubit1':[expX_Q1(), expY_Q1(), expZ_Q1()]}
    
    return res   


def bloch_states(ansatz,qubits, layers, params, class1, class2):
    
    #Mediciones Clase 1
    mediciones_x0_1=[];mediciones_y0_1=[];mediciones_z0_1=[]
    mediciones_x1_1=[];mediciones_y1_1=[];mediciones_z1_1=[]
    
    
    #Mediciones Clase 2
    mediciones_x0_2=[];mediciones_y0_2=[];mediciones_z0_2=[]
    mediciones_x1_2=[];mediciones_y1_2=[];mediciones_z1_2=[]
    
    

    for dato in class1:
        res = make_measurements(ansatz,qubits, layers, params, dato)
        #Qubit 0
        mediciones_x0_1.append(res['qubit0'][0])
        mediciones_y0_1.append(res['qubit0'][1])
        mediciones_z0_1.append(res['qubit0'][2])
                
        #Qubit 1
        mediciones_x1_1.append(res['qubit1'][0])
        mediciones_y1_1.append(res['qubit1'][1])
        mediciones_z1_1.append(res['qubit1'][2])
            
    for dato in class2:
        #Qubit 0
        res2 = make_measurements(ansatz,qubits, layers, params, dato)
        mediciones_x0_2.append(res2['qubit0'][0])
        mediciones_y0_2.append(res2['qubit0'][1])
        mediciones_z0_2.append(res2['qubit0'][2])
                
        #Qubit 1
        mediciones_x1_2.append(res2['qubit1'][0])
        mediciones_y1_2.append(res2['qubit1'][1])
        mediciones_z1_2.append(res2['qubit1'][2])
                
            
            
                    
    clase1 = {'qubit0':{'x':np.array(mediciones_x0_1), 'y':np.array(mediciones_y0_1),'z': np.array(mediciones_z0_1)},
              'qubit1':{'x':np.array(mediciones_x1_1), 'y': np.array(mediciones_y1_1),'z': np.array(mediciones_z1_1)}}
    
    clase2 = {'qubit0':{'x':np.array(mediciones_x0_2), 'y':np.array(mediciones_y0_2),'z': np.array(mediciones_z0_2)},
              'qubit1':{'x':np.array(mediciones_x1_2), 'y': np.array(mediciones_y1_2),'z': np.array(mediciones_z1_2)}}
        
            

    return {'label1':clase1, 'label2':clase2}
