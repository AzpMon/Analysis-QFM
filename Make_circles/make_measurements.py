import pennylane as qml
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
