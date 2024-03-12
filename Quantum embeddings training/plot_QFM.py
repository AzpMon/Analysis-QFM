import plotly.graph_objects as go
import pandas as pd
import numpy as np
from make_measurements import *





def plot_QFM(ansatz, qubits, layers, params,class1,class2):

    ############################Creación estados cuánticos####################
    #CLASE 1 (edos cuánticos)
    datos_clase1_Q0=[]
    datos_clase1_Q1=[]
    for x in class1:
        res = make_measurements(ansatz,qubits, layers, params, x)
        datos_clase1_Q0.append(res["qubit0"])
        datos_clase1_Q1.append(res["qubit1"])
        
        
    #CLASE 2 (edos cuánticos)
    datos_clase2_Q0=[]
    datos_clase2_Q1=[]
    for x in class2:
        res = make_measurements(ansatz,qubits, layers, params, x)
        datos_clase2_Q0.append(res["qubit0"])
        datos_clase2_Q1.append(res["qubit1"])

        
    ############################### QUBIT 0 ###############################
    # Componentes qubit 0, clase 1
    X_C1_Q0 = np.array([datos_clase1_Q0[j][0] for j in range(len(class1))])
    Y_C1_Q0 = np.array([datos_clase1_Q0[j][1] for j in range(len(class1))])
    Z_C1_Q0 = np.array([datos_clase1_Q0[j][2] for j in range(len(class1))])
    label_1 = np.ones(len(X_C1_Q0))

    # Componentes qubit 0, clase 2
    X_C2_Q0 = np.array([datos_clase2_Q0[j][0] for j in range(len(class2))])
    Y_C2_Q0 = np.array([datos_clase2_Q0[j][1] for j in range(len(class2))])
    Z_C2_Q0 = np.array([datos_clase2_Q0[j][2] for j in range(len(class2))])
    label_2 = -1 * np.ones(len(X_C2_Q0))

    #Dataframe Datos Qubit 0
    Q0_C1 = {'<x>': X_C1_Q0, '<y>': Y_C1_Q0, '<z>': Z_C1_Q0, 'class': label_1}
    Q0_C2 = {'<x>': X_C2_Q0, '<y>': Y_C2_Q0, '<z>': Z_C2_Q0, 'class': label_2}
    Q0_C1_df = pd.DataFrame(data=Q0_C1)
    Q0_C2_df = pd.DataFrame(data=Q0_C2)
    qubit_0 = pd.concat([Q0_C1_df, Q0_C2_df])

    ############################### QUBIT 1 ###############################
    # Componentes qubit 1, clase 1
    X_C1_Q1 = np.array([datos_clase1_Q1[j][0] for j in range(len(class1))])
    Y_C1_Q1 = np.array([datos_clase1_Q1[j][1] for j in range(len(class1))])
    Z_C1_Q1 = np.array([datos_clase1_Q1[j][2] for j in range(len(class1))])
    label_1 = np.ones(len(X_C1_Q1))

    # Componentes qubit 1, clase 2
    X_C2_Q1 = np.array([datos_clase2_Q1[j][0] for j in range(len(class2))])
    Y_C2_Q1 = np.array([datos_clase2_Q1[j][1] for j in range(len(class2))])
    Z_C2_Q1 = np.array([datos_clase2_Q1[j][2] for j in range(len(class2))])
    label_2 = -1 * np.ones(len(X_C2_Q1))

    #Dataframe Datos Qubit 0
    Q1_C1 = {'<x>': X_C1_Q1, '<y>': Y_C1_Q1, '<z>': Z_C1_Q1, 'class': label_1}
    Q1_C2 = {'<x>': X_C2_Q1, '<y>': Y_C2_Q1, '<z>': Z_C2_Q1, 'class': label_2}
    Q1_C1_df = pd.DataFrame(data=Q1_C1)
    Q1_C2_df = pd.DataFrame(data=Q1_C2)
    qubit_1 = pd.concat([Q1_C1_df, Q1_C2_df])

    ######################################## PLOT ########################################
    ############### QUBIT 1  ###############
    fig_Q0= go.Figure()

    # Añadir scatter plot
    colors = {1: "cyan", -1: "green"}
    fig_Q0.add_trace(go.Scatter3d(
        x=qubit_0['<x>'],
        y=qubit_0['<y>'],
        z=qubit_0['<z>'],
        mode='markers',
        marker=dict(
            size=5,
            color=qubit_0['class'].map(colors),  # Mapear colores según la clase
            opacity=0.5,
            symbol=qubit_0['class'].map({1: "circle", -1: "diamond"})  # Asignar símbolo según la clase
        )
    ))

    # Añadir cáscara de esfera de radio unitario
    phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    x_sphere = np.sin(theta) * np.cos(phi)
    y_sphere = np.sin(theta) * np.sin(phi)
    z_sphere = np.cos(theta)

    fig_Q0.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale = "YlGnBu",
        opacity=0.2,
        showscale=False     
    ))

    # Configurar diseño de la escena
    fig_Q0.update_layout(
        scene=dict(
            xaxis=dict(title='<x>'),  # Título del eje X
            yaxis=dict(title='<y>'),  # Título del eje Y
            zaxis=dict(title='<z>'),  # Título del eje Z
        ),
        template="plotly_dark",
        annotations=[
            dict(
                text="QUBIT 0",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.99,  # Ajustar las coordenadas x e y para centrar el título
                font=dict(size=14)
            )
        ]
    )

    ############### QUBIT 1  ###############
    fig_Q1 = go.Figure()

    #Añadir estados cuánticos asociados según el mapeo
    fig_Q1.add_trace(go.Scatter3d(
        x=qubit_1['<x>'],
        y=qubit_1['<y>'],
        z=qubit_1['<z>'],
        mode='markers',
        marker=dict(
            size=5,
            color=qubit_0['class'].map(colors),  # Mapear colores según la clase
            opacity=0.5,
            symbol=qubit_0['class'].map({1: "circle", -1: "diamond"})  # Asignar símbolo según la clase
        )
    ))

    #Añadir esfera de Bloch
    fig_Q1.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale = "YlGnBu",
        opacity=0.2,
        showscale=False     
    ))

    fig_Q1.update_layout(
        scene=dict(
            xaxis=dict(title='<x>'),  # Título del eje X
            yaxis=dict(title='<y>'),  # Título del eje Y
            zaxis=dict(title='<z>'),  # Título del eje Z
        ),
        template="plotly_dark",
        annotations=[
            dict(
                text="QUBIT 1",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.99,  # Ajustar las coordenadas x e y para centrar el título
                font=dict(size=14)
            )
        ]
    )

    fig_Q0.show()
    fig_Q1.show()
