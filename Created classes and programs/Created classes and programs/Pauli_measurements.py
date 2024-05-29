import pennylane as qml
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import dark_palette,light_palette
import plotly.graph_objects as go
from scipy.stats import entropy
from sklearn.svm import SVC 
import torch 
import  sin_prob_dist # type: ignore
import itertools

class Pauli_measurements:
    def __init__(self,tensorDataset, ansatz, params, qubits, layers,mesh_samples = 10000):
        """Initialize the Pauli_measurements class.
        Args:
            tensorDataset (torch.tensor): Dataset with labels in the last columnn 
            colors (list): List of colors to use for each label 

        """ 
        
        ######################### SEPARAR DATOS ############################
        # Dictionary for storing the separated datasets
        self._dictLabelsData = {}
        
        # Get the labels
        lastColumnIndex = tensorDataset.shape[1] -1
        self.labelsDataset = tensorDataset[:, lastColumnIndex]
        self._unique_labels = torch.unique(self.labelsDataset)
        self.__intUniqueLabels = [int(label) for label in self._unique_labels]
        
        
    
        # Separate the data
        for label in self._unique_labels:
            mask = self.labelsDataset == label.item()  # Create a mask for the label
            dataBlock = tensorDataset[mask][:, :-1]    # Filtering data by label
            self._dictLabelsData[f'DataLabel{int(label.item())}'] = dataBlock 


        ############################### ATRIBUTOS DEL CIRCUITO ###################        
        self.ansatz = ansatz
        self.params = params 
        self.qubits = qubits
        self.layers = layers 
        
        ######################## CREAR ESTADOS CUÁNTICOS ######################
        # Dictionary for storing the traced quantum states 
        self._dictQuantumStates =  self.__quantumStates()
        # self._dictQuantumStates[qubitQ][labelL]
        
        

        ############################### ESPACIOS MUESTRALES ###############################
        #Grid design for Sample Space's design with physical restriction 2D... [-1,1]x[-1,1]
        self._real_samples = pow( round(np.sqrt(mesh_samples)), 2)
        self._len_samples = round(np.sqrt(mesh_samples)) 
        max_grid = 1  - (2/self._len_samples - 1/self._len_samples) 
        min_grid = -max_grid 
                
        #Sample Space [-1,1]x[-1,1] of expected values for Pauli Observables 2D
        self._u_mesh = np.linspace(min_grid, max_grid, self._len_samples)
        self._v_mesh = np.linspace(min_grid, max_grid, self._len_samples)
        self._U, self._V = np.meshgrid(self._u_mesh, self._v_mesh)
        self.sample_space = np.column_stack((self._U.flatten(), self._V.flatten()))
        self._sample_space_u = self.sample_space[:,0]
        self._sample_space_v = self.sample_space[:,1]
        self.mesh_area = 4 / self._real_samples
        
        
        # Sample Space for the bloch sphare
        self.mesh_samples = mesh_samples
        self._phi = np.random.uniform(0, 2 * np.pi, self.mesh_samples)
        self._r = np.cbrt(np.random.uniform(0, 1, self.mesh_samples))


        # Dada la descripción del espacio según las coordenadas esféricas se sample de forma distinta
        sin_sampler = sin_prob_dist.sin_prob_dist(a=0, b=np.pi)
        self._theta =  sin_sampler.rvs(size=self.mesh_samples)

        self._sphere_x = self._r * np.sin(self._theta) * np.cos(self._phi)
        self._sphere_y = self._r * np.sin(self._theta) * np.sin(self._phi)
        self._sphere_z = self._r * np.cos(self._theta)

        # Create sphere sample space
        self._sphere_sample_space = np.column_stack((self._sphere_x, self._sphere_y, self._sphere_z))
        self._sphere_vol_diff = (4 * np.pi / 3) / self.mesh_samples  #Aprox.



        ######################### CREAR LAS DISTRIBUCIONES DE LOS DATOS PARA CADA QUBIT Y ETIQUETA #####################
        # Dictionary for storing the KDE PDFS for each qubit and label (complete)
        self._dictPdfsFullQubit={}
        # self._dictPdfsFullQubit['qubitQ']['labelL']
        for qubit in range(self.qubits):
            dictLabels = {}
            for label in self.__intUniqueLabels:
                dictLabels[f'label{label}'] = self._kde_pdf_qubit(qubit, label)
            self._dictPdfsFullQubit[f'qubit{qubit}'] = dictLabels



        ######################### CREAR LAS DISTRIBUCIONES DE LOS DATOS PARA CADA PAR DE COORDENADAS EN UN QUBIT Y ETIQUETA #####################
        # Dictionary for storing the KDE_2D PDFS for each qubit and label (complete)
        self._dictPdfs2D={}
        # self._dictPdfs2D[qubitQ][labelN][cord1cord2]
        for qubit in range(self.qubits):
            dictLabels = {}
            
            for label in self.__intUniqueLabels:
                dictCordinates = {}
                for cordinates in [('X','Y'), ('X','Z'), ('Y','Z')]:
                    coord1 = cordinates[0]
                    coord2 = cordinates[1]
                    dictCordinates[f'{coord1}{coord2}'] = self._kde_pdf_2D(qubit,label, coord1, coord2)
                dictLabels[f'label{label}'] = dictCordinates
            self._dictPdfs2D[f'qubit{qubit}'] = dictLabels
            
            
            
        self.__labelCombinations = list(itertools.combinations(self.__intUniqueLabels ,2))
        ########################## CALCULAR KL PARA CADA QUBIT Y PARA CADA CLASE  ############################
        self._dictKlFullQubit = {}
        # _dictKlFullQubit[qubitQ][labelN_labelM]  
        for qubit in range(self.qubits):
            dictKL = {}
            for combination in self.__labelCombinations:
                label1 = combination[0]
                label2 = combination[1]
                dictKL[f'label{label1}_label{label2}'] = self.kl_divergence_qubit(qubit,label1, label2)
            
            self._dictKlFullQubit[f'qubit{qubit}'] = dictKL 
            
        
        ########################## CALCULAR KL PARA CADA QUBIT Y PROYECCIÓN  ###################################
        # _dictKlProjectedQubit[qubitQ][labelN_labelM]['UV']
        self._dictKlProjectedQubit={}
        for qubit in range(self.qubits):
            dictKlProjected = {}
            for combination in self.__labelCombinations:
                label1 = combination[0]
                label2 = combination[1]
                
                dictKlCoordinates={}
                for coordinate in [('X','Y'),('X','Z'),('Y','Z')]:
                    u = coordinate[0]
                    v = coordinate[1]
                    
                    dictKlCoordinates[f'{u}{v}'] = self.kl_divergence_2D(u,v,qubit,label1, label2)
                dictKlProjected[f'label{label1}_label{label2}'] = dictKlCoordinates
            self._dictKlProjectedQubit[f'qubit{qubit}'] = dictKlProjected 
            
        
                
    
    
    def __expValueOneQubit(self, qubit, observable, ansatz,qubits,layers, params, dato):
        """ Computes the quantum projected state of the optimized encoding cirucit in the 
        indicated qubit, then computes the expected value for the given observable.
        Args:
            qubit (int): Qubit of interest
            observable (string): X,Y or Z
            ansatz (qml.function): Quantum circuit
            qubits (int): Number of qubits of the circuit
            layers (int): Number of layers of the circuit
            params (torch.tensor): Optimized circuit's params
            dato (torch.tensor): Data to be considered

        Raises:
            ValueError: Observable must be 'X', 'Y', or 'Z

        Returns:
            torch.tensor: The trace of the observable with the projected quantum state
        """
        
        dev = qml.device(name = "default.qubit")        # Pennylane device
        @qml.qnode(device = dev, interface="torch")     # Pennylane qnode
        def traceOutFromAnsatz():       
            """
            Implement the circuit and then trace out to the 
            interested qubit, return the density matrix
            """                
            ansatz(qubits,layers, params, dato)
            return qml.density_matrix(wires=qubit)
        
        rho_qubit = traceOutFromAnsatz()                # Obtain the traced state
        
        # Make the measurements in the Pauli Observables
        if observable == 'X':
            observableMatrix = qml.matrix(op = qml.PauliX(qubit))
        elif observable == 'Y':
            observableMatrix = qml.matrix(op = qml.PauliY(qubit))
        elif observable == 'Z':
            
            observableMatrix = qml.matrix(op = qml.PauliZ(qubit))
        else:
             raise ValueError("Observable must be 'X', 'Y', or 'Z'.")  
         
         
        return np.real(np.trace(np.dot(rho_qubit, observableMatrix)))   
    

    def makeMeasurements(self, qubit,data):
        """ Implementing meausrements in each Pauli Observable, computes the bloch's vector
        of the mapped data

        Args:
            qubit (int): In wich qubit the measurement will be done
            dato (torch.tensor): Type of data

        Returns:
            torch.tensor: 
        """
        measurement = []
        for obs in 'X', 'Y', 'Z':
            measurement.append(self.__expValueOneQubit(qubit, obs, self.ansatz,self.qubits,self.layers, self.params, data))
        return  measurement
    

    
    def make_measurement(self,obs, qubit, dato):
        """ Given an observabe an a qubit of interest, computes the expected value
        Args:
            obs (str): It must be X,Y or Z
            qubit (int): Qubit of interest
            dato (torch.tensor): Classic data point

        Returns:
            torch.tensor: Expected value
        """
        return self.__expValueOneQubit(self, qubit, obs, self.ansatz,self.qubits,self.layers, self.params, dato)
        
        
        
        
    
    def __quantumStates(self):
        """Get the quantum states of all the data set
        Returns:
            dict: A dictionary with measurements for each class-data; dict[labeln][qubitm]['component'] for 
                    n label (1,2) and m qubit(0,1)
        """
        
        # Dictionary for storing the quantum states for each qubit an label
        quantumStateS = {}
        
        
        # For each qubit     
        for qubiT in range(self.qubits):
            # For each Label
            dataLabelDict = {}
            for labeL in self._unique_labels:
                res = []
                # Computes the states for each label
                for dato in self._dictLabelsData[f'DataLabel{int(labeL.item())}']:
                    res.append(self.makeMeasurements(qubiT,dato))
                dataLabelDict[f'label{int(labeL.item())}'] = torch.tensor(res) 
                
                
                # Storing it in the dictionary  
                quantumStateS[f'qubit{qubiT}']= dataLabelDict
                
        return quantumStateS
    

    ####################################################### FULL QUBIT ##################################
    def _kde_pdf_qubit(self, qubit, label):
        """Using grid search for the bandwidht computes the KernelDensity for the  labeled dataset.Then
        evaluates the pdf in the sample space, returning the pdf to be evaluate in a single datapoint (quantum state)
        Args:
            qubit (int): Index of qubit
            label (int): Label associated to the data that will be considered
        Returns:
            function: PDF of the associated quantum states computed by a KDE using grid search 
        """
        
        # Grid  search (for the bw)  for the dataset
        
        params = {'bandwidth': np.linspace(0.001, 2, 30)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'])
        
        # Kernel Density Estimation 
        best_kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'])
        best_kde.fit(self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'])  
        

        
        #Sample probabilities bloch_sphere sample space
        sampleProbsSum = self._sphere_vol_diff * np.sum(np.exp(best_kde.score_samples(self._sphere_sample_space)))
        
        
        #PDF normalized and with physical restrictions
        def calculate_probability_label(data):
            data = data.reshape(1,-1)
            if np.all(data <= 1) and np.all(data>= -1):
                return np.exp(best_kde.score_samples(data)) / sampleProbsSum
            else:
                return 0

        return  calculate_probability_label 

    def kl_divergence_qubit(self,qubit,label1, label2):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in the correspond qubit.
        Args:
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """        
        pdf1 = self._dictPdfsFullQubit[f'qubit{qubit}'][f'label{label1}']
        pdf2 = self._dictPdfsFullQubit[f'qubit{qubit}'][f'label{label2}']

        p1_probs = np.array([pdf1(sample) for sample in self._sphere_sample_space])
        p2_probs = np.array([pdf2(sample) for sample in self._sphere_sample_space])
        
        p1_probs = p1_probs[p1_probs > 0]
        p2_probs = p2_probs[p2_probs > 0]
        
        #KL divergence using scipy.stats.entropy
        kl_bloch_sphere = entropy(p1_probs, p2_probs)

        return kl_bloch_sphere 

    




    ######################### Computes the preNormalized PDF calculated from the Grid-Search KernelDensity ##########################
    def _kde_pdf_estimation_2D(self, data):
        """Using grid search fot the bandwidht computes the KernelDensity for the given dataset.Then
        evaluates the pdf in the sample space, returning the sum of its probabilities
        Args:
            dataset (torch.tensor): Dataset to obtain the bw and kde.
        Returns:
            numpy.float64: 
        """
        #Grid  search (for the bw) 
        params = {'bandwidth': np.linspace(0.001, 2, 30)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(data)
        
        #KDE 
        best_kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'])
        best_kde.fit(data)
        
        #Sample probabilities vol 
        sample_probs = np.sum(np.exp(best_kde.score_samples(self.sample_space)))
        
        def calculate_probability(data):
            data = data.reshape(1,-1)
            if np.all(data <= 1) and np.all(data>= -1):
                return np.exp(best_kde.score_samples(data)) / sample_probs
            else:
                return 0
        return  calculate_probability
        




    
    ############################## Computes the pdf using KDE for a determine qubit and pair of coordinates ##########################################
    def _kde_pdf_2D(self, qubit,label, coord1, coord2):
        """Private method to obtain  the pdf in the selected qubit corresponded to a label for the coord1, coord2 for all the classical dataset
        e.g: qubit=1,label=1, coord1='x', coord2='y' computes the pdf_xy for label1 in the qubit 1
        Args:
            qubit  (int): Considered qubit for the KDE pdf
            label  (int): Label to be considered in the computation
            coord1 (str): Pauli_expected_value1 to be consider ('x','y','z')
            coord2 (str): Pauli_expected_value2 to be consider ('x','y','z')
        Returns:
            list: kde_pdf_label1, kde_pdf_label2 
        """
        column1 = 0
        column2 = 1
        # To make the calculations in the tensors (cordinate 1)
        if coord1 == 'x':
            column1 = 0
        elif coord1 == 'y':
            column1 = 1
        elif coord1 == 'z':
            column1=2

        # To make the calculations in the tensors (cordinate 1)
        if coord2 == 'x':
            column2 = 0
        elif coord2 == 'y':
            column2 = 1
        elif coord2 == 'z':
            column2 = 2

        # Projection of the states (u,v)
        data_u = self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'][:, column1]
        data_v = self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'][:, column2]
        data_2D   = np.column_stack((data_u,data_v ))
        

        #KDE estimation using grid search for label1 data, return the pdf to be evaluate in a datapoint
        kde_pdf_2D = self._kde_pdf_estimation_2D(data_2D)
        
        
        return kde_pdf_2D  
    
    
    
    def kl_divergence_2D(self, coord1, coord2, qubit, label1, label2):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in 2D.
        Args:
            observable_1 (str): Pauli observable for the first distribution.
            observable_2 (str): Pauli observable for the second distribution.
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """
        pdf1 =  self._dictPdfs2D[f'qubit{qubit}'][f'label{label1}'][f'{coord1}{coord2}']
        pdf2 =  self._dictPdfs2D[f'qubit{qubit}'][f'label{label2}'][f'{coord1}{coord2}']


        p1_probs = np.array([pdf1(sample) for sample in self.sample_space])
        p2_probs = np.array([pdf2(sample) for sample in self.sample_space])
        
        p1_probs = p1_probs[p1_probs > 0]
        p2_probs = p2_probs[p2_probs > 0]
        
        #KL divergence using scipy.stats.entropy
        kl_bloch_sphere = entropy(p1_probs, p2_probs)

        return kl_bloch_sphere 
    
    
    
    
    
    
    ##################################### VISUALIZACIÓN 2D #################################
    def plot_quantum_pdf(self, observable_1, observable_2, qubit, label1, label2, colors=('cyan', 'green'), figsize=(14, 6), levels=20, save_fig=False, plot_name="ex", sns_palette='light_palette'):
        """Plot quantum probability density functions (2D)
        Args:
            observable_1 (str): Pauli observable for the horizontal axis.
            observable_2 (str): Pauli observable for the vertical axis.
            qubit (int): Index of the qubit.
            figsize (tuple, optional): Figure size. Defaults to (14, 6).
            levels (int, optional): Number of contour levels. Defaults to 25.
            save_fig (bool, optional): Save the figure. Defaults to False.
            plot_name (str, optional): Name for the saved plot. Defaults to "ex".
        """

        # Cmap configuration
        cmap1 = ''
        cmap2 = ''
        if sns_palette == 'dark_palette':
            cmap1 = dark_palette(f'{colors[0]}', as_cmap=True)
            cmap2 = dark_palette(f'{colors[1]}', as_cmap=True)
        elif sns_palette == 'light_palette':
            cmap1 = light_palette(f'{colors[0]}', as_cmap=True)
            cmap2 = light_palette(f'{colors[1]}', as_cmap=True)

        # self._dictPdfs2D[qubitQ][labelN][cord1cord2]
        # KDE-pdfs
        pdf1 = self._dictPdfs2D[f'qubit{qubit}'][f'label{label1}'][f'{observable_1}{observable_2}']
        pdf2 = self._dictPdfs2D[f'qubit{qubit}'][f'label{label2}'][f'{observable_1}{observable_2}']

        # Probabilities
        probs1 = np.array([pdf1(sample) for sample in self.sample_space])
        probs2 = np.array([pdf2(sample) for sample in self.sample_space])

        # Reshape probabilities for contour plots
        probs1 = probs1.reshape(self._len_samples, self._len_samples)
        probs2 = probs2.reshape(self._len_samples, self._len_samples)

        # KL Divergence
        kl = self._dictKlProjectedQubit[f'qubit{qubit}'][f'label{label1}_label{label2}'][f'{observable_1}{observable_2}']

        # Subplot configuration
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # ContourPlot label1
        contour1 = axs[0].contourf(self._U, self._V, probs1, levels=levels, cmap=cmap1)
        axs[0].set_title('Quantum KDE-PDF (label 1)')
        axs[0].set_xlabel(f'$\\langle {observable_1.upper()} \\rangle$')
        axs[0].set_ylabel(f'$\\langle {observable_2.upper()} \\rangle$')
        axs[0].set_xlim([-1, 1])
        axs[0].set_ylim([-1, 1])
        axs[0].grid(True)  # Add grid

        # ContourPlot label2
        contour2 = axs[1].contourf(self._U, self._V, probs2, levels=levels, cmap=cmap2)
        axs[1].set_title('Quantum KDE-PDF (label 2)')
        axs[1].set_xlabel(f'$\\langle {observable_1.upper()} \\rangle$')
        axs[1].set_ylabel(f'$\\langle {observable_2.upper()} \\rangle$')
        axs[1].set_xlim([-1, 1])
        axs[1].set_ylim([-1, 1])
        axs[1].grid(True)  # Add grid

        plt.suptitle(f'QUBIT {qubit}\n KL = {round(kl, 5)}', fontsize=16, fontweight='bold', y=1.01)

        # Save the figure
        if save_fig:
            plt.savefig(f'{plot_name}.eps', format='eps', bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    
    
    


    ################################# VISUALIZACIÓN 3D ###########################################
    # Plottly interactive plot of the optimized circuit applied to the dataset
    def plot_interactive_quantumSpace(self, qubit, label1, label2, colors=('cyan', 'green')):
        """Make an interactive plot of the Hilbert spaces for the qubit of interest
        """
        kl_0 = self._dictKlFullQubit[f'qubit{qubit}'][f'label{label1}_label{label2}']  
        

        labels_1 =  label1*np.ones(len(self._dictQuantumStates[f'qubit{qubit}'][f'label{label1}'][:,0]))
        labels_2 =  label2*np.ones(len(self._dictQuantumStates[f'qubit{qubit}'][f'label{label2}'][:,0]))
        
        # Qubit0 DataFrame
        Q0_C1 = {
            '<x>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label1}'][:,0],
            '<y>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label1}'][:,1],
            '<z>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label1}'][:,2],
            'label': labels_1
        }
        Q0_C2 = {
            '<x>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label2}'][:,0],
            '<y>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label2}'][:,1],
            '<z>': self._dictQuantumStates[f'qubit{qubit}'][f'label{label2}'][:,2],
            'label': labels_2
        }
        qubit_0 = pd.concat([pd.DataFrame(data=Q0_C1), pd.DataFrame(data=Q0_C2)])

        ######################################## PLOT ########################################
        fig_Q0 = go.Figure()

        # Añadir scatter plot
        colors = {label1: colors[0], label2: colors[1]}
        fig_Q0.add_trace(go.Scatter3d(
            x=qubit_0['<x>'],
            y=qubit_0['<y>'],
            z=qubit_0['<z>'],
            mode='markers',
            marker=dict(
                size=5,
                color=qubit_0['label'].map(colors),  # Mapear colores según la clase
                opacity=0.5,
                symbol=qubit_0['label'].map({label1: "circle", label2: "circle"})  # Asignar símbolo según la clase
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
            colorscale="YlGnBu",
            opacity=0.2,
            showscale=False     
        ))

        # Configurar diseño de la escena
        fig_Q0.update_layout(
            scene=dict(
                xaxis=dict(title='<X>'),  # Título del eje X
                yaxis=dict(title='<Y>'),  # Título del eje Y
                zaxis=dict(title='<Z>'),  # Título del eje Z
            ),
            template="plotly",
            annotations=[
                dict(
                    text=f'QUBIT {qubit}, KL={round(kl_0, 5)}',
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.99,  # Ajustar las coordenadas x e y para centrar el título
                    font=dict(size=14)
                )
            ]
        )

        fig_Q0.show()