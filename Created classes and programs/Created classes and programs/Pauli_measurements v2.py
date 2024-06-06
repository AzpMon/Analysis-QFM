import pennylane as qml
import torch 
from scipy.stats import entropy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from seaborn import dark_palette,light_palette
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from  separateData import *
import sin_prob_dist # type: ignore

class Pauli_measurements:
    def __init__(self,dataset,label1,label2, ansatz,params, qubits, layers,mesh_samples = 10000, colors=('blue', 'green')):
        """Initialize the Pauli_measurements class.
        Args:
            tensorDataset (torch.tensor): Dataset with labels in the last columnn 
            colors (list): List of colors to use for each label 
        """ 


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


        
        
        
        ############################### CIRCUIT ATRIBUTES ###########################        
        self.ansatz = ansatz
        self.params = params 
        self.qubits = qubits
        self.layers = layers 
 


        ############################### DATASET ATRIBUTES ###########################     
        self._label1 = label1
        self._label2 = label2
        self._labels = [self._label1, self._label2] 
        self.__dataSet = separateData(dataset) 
        self._dataLabel1 = self.__dataSet.separarDatos(self._label1)
        self._dataLabel2 = self.__dataSet.separarDatos(self._label2)
        self._dictLabelsData = {f'DataLabel{self._label1}':self._dataLabel1,f'DataLabel{self._label2}':self._dataLabel2 }
        self.colors = colors


        
        
        ######################## CREATE QUANTUM STATES ######################
        # Dictionary for storing the traced quantum states 
        self._dictQuantumStates =  self.__quantumStates()
        # self._dictQuantumStates[qubitQ][labelL]


    
        ######################### CREATE PROBABILIRY DENSITY FUNCTIONS FOR THE  DATA FOR EACH QUBIT AND LABEL #####################
        # Dictionary for storing the KDE PDFS for each qubit and label (complete)
        self._dictPdfsFullQubit={}                           # Dict for save the associated PDFs for each qubit
        # self._dictPdfsFullQubit['qubitQ']['labelL']
        for qubit in range(self.qubits):
            dictLabels = {}
            for label in self._labels:
                quantumData = self._dictQuantumStates[f'qubit{qubit}'][f'label{label}']
                dictLabels[f'label{label}'] = self._kde_pdf_FullQubit(quantumData)
            self._dictPdfsFullQubit[f'qubit{qubit}'] = dictLabels

        
        ########################## COMPUTE KL FOR EACH QUBIT  ############################
        self._dictKlFullQubit = {}
        # _dictKlFullQubit[qubitQ]  
        for qubit in range(self.qubits):
            self._dictKlFullQubit[f'qubit{qubit}'] = self._kl_divergence_qubit(qubit)[0]



        






        ######################### COMPUTE THE PDFS FOR EACH QUBIT AND PAIR OF COORDINATES #####################
        self._dictPdfs2D={}
        # self._dictPdfs2D[qubitQ][labelN][cord1cord2]
        for qubit in range(self.qubits):
            dictLabels = {}
            
            for label in self._labels:
                dictCordinates = {}
                for cordinates in [('X','Y'), ('X','Z'), ('Y','Z')]:
                    coord1 = cordinates[0]
                    coord2 = cordinates[1]
                    dictCordinates[f'{coord1}{coord2}'] = self._kde_pdf_2D(qubit,label, coord1, coord2)
                dictLabels[f'label{label}'] = dictCordinates
            self._dictPdfs2D[f'qubit{qubit}'] = dictLabels

        ########################## CALCULAR KL PARA CADA QUBIT Y PROYECCIÓN  ###################################
        # _dictKlProjectedQubit[qubitQ]['UV']
        self._dictKlProjectedQubit={}
        for qubit in range(self.qubits):
            dictKlCoordinates={}
            for coordinate in [('X','Y'),('X','Z'),('Y','Z')]:
                u = coordinate[0]
                v = coordinate[1]
                
                dictKlCoordinates[f'{u}{v}'] = self.kl_divergence_2D(u,v,qubit)[0]
            self._dictKlProjectedQubit[f'qubit{qubit}'] = dictKlCoordinates 
            


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
            ansatz(qubits,layers, params, dato)         # Implement the ansatz
            return qml.density_matrix(wires=qubit)      # Trace out for the qubit of interest
        
        rho_qubit = traceOutFromAnsatz()                # Obtain the traced state
        
        # Make the measurements in the corresponded Pauli Observable
        if observable == 'X':
            observableMatrix = qml.matrix(op = qml.PauliX(qubit))
        elif observable == 'Y':
            observableMatrix = qml.matrix(op = qml.PauliY(qubit))
        elif observable == 'Z':
            observableMatrix = qml.matrix(op = qml.PauliZ(qubit))
        else:
             raise ValueError("Observable must be 'X', 'Y', or 'Z'.")  

        # Compute the expected value for the corresponded observable via density operators
        expectedValue = np.real(np.trace(np.dot(rho_qubit, observableMatrix))) 
        
        return expectedValue

    

    def _makeMeasurements(self, qubit,data):
        """ Implementing meausrements in each Pauli Observable, computes the bloch's vector
        of the mapped data

        Args:
            qubit (int): In wich qubit the measurement will be done
            dato (torch.tensor): Type of data

        Returns:
            torch.tensor: 
        """
        measurement = []                    # List for the bloch vectors
        for obs in 'X', 'Y', 'Z':
            # Compute the expected value for the qubit of interest
            expValue_traced = self.__expValueOneQubit(qubit, obs, self.ansatz,self.qubits,self.layers, self.params, data)
            measurement.append(expValue_traced)
        return  measurement

    
    def __quantumStates(self):
        """Get the quantum states of all the data set
        Returns:
            dict: A dictionary with measurements for each class-data; dict[qubitQ][labelL] for 
                    qubit Q and label L
        """
        # Dictionary for storing the quantum states for each qubit an label
        quantumStates = {}
        # For each qubit   
        
        for qubit in range(self.qubits):
            # For each Label
            dataLabelDict = {}
            for label in self._labels:                
                res = []
                # Computes the states for each label
                for dataPoint in self._dictLabelsData[f'DataLabel{label}']:
                    res.append(self._makeMeasurements(qubit, dataPoint))
                
                dataLabelDict[f'label{label}'] = torch.tensor(res, requires_grad=False)
            quantumStates[f'qubit{qubit}']=dataLabelDict

        # Storing it in the dictionary  
        return quantumStates


    

    
    ####################################################### FULL QUBIT  PDFs##################################
    def _kde_pdf_FullQubit(self, quantumData):
        """Using grid search for the bandwidth, computes the Kernel Density Estimation (KDE) for the labeled quantum dataset.
        Then evaluates the probability density function (PDF) in the sample space, returning the PDF to be evaluated for a single data point (quantum state).
        Args:
            quantumData (torch.tensor): Set of quantum states
        Returns:
            function: PDF of the associated quantum states computed by a KDE using grid search 
        """
    
        # Convert the tensor to a Numpy array
        quantumData_np = quantumData.numpy()
        
        # Grid search (for bandwidth) for the dataset
        paramsKDE = {'bandwidth': np.linspace(0.005, 1, 15)}
        grid = GridSearchCV(KernelDensity(), paramsKDE, cv=5)
        grid.fit(quantumData_np)
        
        # Kernel Density Estimation 
        best_kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'])
        best_kde.fit(quantumData_np)  
    
        # Sample probabilities for the Bloch sphere sample space
        sampleProbsSum = self._sphere_vol_diff * np.sum(np.exp(best_kde.score_samples(self._sphere_sample_space)))
        
        # PDF normalized and with physical restrictions
        def calculate_probability_label(data):
            data = data.reshape(1, -1)
            if np.all(data <= 1) and np.all(data >= -1):
                if sampleProbsSum != 0:  # Handling the case of division by zero
                    return np.exp(best_kde.score_samples(data)) / sampleProbsSum
                else:
                    return 0
            else:
                return 0
                    
        return calculate_probability_label


    def _kl_divergence_qubit(self, qubit):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in the corresponding qubit.
        Args:
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """
        # The associated pdfs for the correspondend qubit and label
        pdf1 = self._dictPdfsFullQubit[f'qubit{qubit}'][f'label{self._label1}']
        pdf2 = self._dictPdfsFullQubit[f'qubit{qubit}'][f'label{self._label2}']





        p1_probs = np.array([pdf1(sample) for sample in self._sphere_sample_space])
        p2_probs = np.array([pdf2(sample) for sample in self._sphere_sample_space])
        return entropy(p1_probs,p2_probs)





    ####################################################### PROJECTED  QUBIT  PDFs ##########################################
    def _kde_pdf_ProjectedQubit(self, quantumProjectedData):
        """Using grid search for the bandwidth, computes the Kernel Density Estimation (KDE) for the projected quantum dataset.
        Then evaluates the probability density function (PDF) in the sample space, returning the PDF to be evaluated for a single data point (quantum state).
        Args:
            quantumData (numpy.array): Set of quantum projected  states
        Returns:
            function: PDF of the associated quantum states computed by a KDE using grid search 
        """        
        # Grid search (for bandwidth) for the dataset
        paramsKDE = {'bandwidth': np.linspace(0.005, 1, 15)}
        grid = GridSearchCV(KernelDensity(), paramsKDE, cv=5)
        grid.fit(quantumProjectedData)
        
        # Kernel Density Estimation 
        best_kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'])
        best_kde.fit(quantumProjectedData)  
    
        # Sample probabilities for the Bloch sphere sample space
        sampleProbsSum = self.mesh_area  * np.sum(np.exp(best_kde.score_samples(self.sample_space)))
        
        # PDF normalized and with physical restrictions
        def calculate_probability_label(data):
            data = data.reshape(1, -1)
            if np.all(data <= 1) and np.all(data >= -1):
                if sampleProbsSum != 0:  # Handling the case of division by zero
                    return np.exp(best_kde.score_samples(data)) / sampleProbsSum
                else:
                    return 0
            else:
                return 0
                    
        return calculate_probability_label
    
    ############################## Computes the pdf using KDE for a determine qubit and pair of coordinates ##########################################
    def _kde_pdf_2D(self, qubit,label, coord1, coord2):
        """Private method to obtain  the pdf in the selected qubit corresponded to a label for the coord1, coord2 for all the classical dataset
        e.g: qubit=1,label=1, coord1='x', coord2='y' computes the pdf_xy for label1 in the qubit 1
        Args:
            qubit  (int): Considered qubit for the KDE pdf
            label  (int): Label to be considered in the computation
            coord1 (str): Pauli_expected_value1 to be consider ('X','Y','Z')
            coord2 (str): Pauli_expected_value2 to be consider ('X','Y','Z')
        Returns:
            list: kde_pdf_label1, kde_pdf_label2 
        """
        column1 = 0
        column2 = 0
        # To make the calculations in the tensors (cordinate 1)
        if coord1 == 'X':
            column1 = 0
        elif coord1 == 'Y':
            column1 = 1
        elif coord1 == 'Z':
            column1=2

        # To make the calculations in the tensors (cordinate 1)
        if coord2 == 'X':
            column2 = 0
        elif coord2 == 'Y':
            column2 = 1
        elif coord2 == 'Z':
            column2 = 2

        # Projection of the states (u,v)
        data_u = self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'][:, column1]
        data_v = self._dictQuantumStates[f'qubit{qubit}'][f'label{label}'][:, column2]
        data_2D   = np.column_stack((data_u,data_v ))
        

        #KDE estimation using grid search for label1 data, return the pdf to be evaluate in a datapoint
        kde_pdf_2D = self._kde_pdf_ProjectedQubit(data_2D)
        
        
        return kde_pdf_2D  



    def kl_divergence_2D(self, coord1, coord2, qubit):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in 2D.
        Args:
            observable_1 (str): Pauli observable for the first distribution.
            observable_2 (str): Pauli observable for the second distribution.
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """
        pdf1 =  self._dictPdfs2D[f'qubit{qubit}'][f'label{self._label1}'][f'{coord1}{coord2}']
        pdf2 =  self._dictPdfs2D[f'qubit{qubit}'][f'label{self._label2}'][f'{coord1}{coord2}']


        p1_probs = np.array([pdf1(sample) for sample in self.sample_space])
        p2_probs = np.array([pdf2(sample) for sample in self.sample_space])



        #KL divergence using scipy.stats.entropy
        kl_projected = entropy(p1_probs, p2_probs)

        return kl_projected 



    ##################################### VISUALIZACIÓN 2D #################################
    def plot_quantum_pdf_2D(self, observable_1, observable_2, qubit, figsize=(14, 6), levels=10, save_fig=False, plot_name="ex", sns_palette='light_palette'):
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
            cmap1 = dark_palette(f'{self.colors[0]}', as_cmap=True)
            cmap2 = dark_palette(f'{self.colors[1]}', as_cmap=True)
        elif sns_palette == 'light_palette':
            cmap1 = light_palette(f'{self.colors[0]}', as_cmap=True)
            cmap2 = light_palette(f'{self.colors[1]}', as_cmap=True)

        pdf1 = self._dictPdfs2D[f'qubit{qubit}'][f'label{self._label1}'][f'{observable_1}{observable_2}']
        pdf2 = self._dictPdfs2D[f'qubit{qubit}'][f'label{self._label2}'][f'{observable_1}{observable_2}']

        # Probabilities
        probs1 = np.array([pdf1(sample) for sample in self.sample_space])
        probs2 = np.array([pdf2(sample) for sample in self.sample_space])

        # Reshape probabilities for contour plots
        probs1 = probs1.reshape(self._len_samples, self._len_samples)
        probs2 = probs2.reshape(self._len_samples, self._len_samples)

        # KL Divergence
        kl = self._dictKlProjectedQubit[f'qubit{qubit}'][f'{observable_1}{observable_2}']

        # Subplot configuration
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # ContourPlot label1
        sns.kdeplot(x=self._sample_space_u, y=self._sample_space_v,cmap = cmap1, weights=probs1.flatten(), fill=True, ax=axs[0],alpha=0.8,  cbar=False, levels = levels)
        axs[0].set_title(f'Quantum KDE (label {self._label2})')
        axs[0].set_xlabel(f'$\\langle {observable_1.upper()} \\rangle$')
        axs[0].set_ylabel(f'$\\langle {observable_2.upper()} \\rangle$')
        axs[0].set_xlim([-1, 1])
        axs[0].set_ylim([-1, 1])
        axs[0].grid(True)  # Add grid

        sns.kdeplot(x=self._sample_space_u, y=self._sample_space_v,cmap = cmap2, weights=probs2.flatten(), fill=True, ax=axs[1],alpha=0.8,  cbar=False, levels = levels)
        axs[1].set_title(f'Quantum KDE (label {self._label2})')
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
        
    def plot_all_quantum_pdfs_2D(self):
        """For all the qubits  and each projected space plot the associated pdf  and the  KL between each label"""
        for qubit in range(self.qubits):
            for observables in [['X','Y'],['X','Z'], ['Y','Z']]:
                observable_1=observables[0]
                observable_2=observables[1]
                self.plot_quantum_pdf_2D(observable_1, observable_2, qubit)     
            print("\n\n\n")
                
                
                



 ################################# VISUALIZACIÓN 3D ###########################################
    # Plottly interactive plot of the optimized circuit applied to the dataset
    def plot_full_bloch_sphere(self, qubit):
        """Make an interactive plot of the Hilbert spaces for the qubit of interest
        """
        kl_0 = self._dictKlFullQubit[f'qubit{qubit}']
        

        labels_1 =  self._label1*np.ones(len(self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label1}'][:,0]))
        labels_2 =  self._label2*np.ones(len(self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label2}'][:,0]))
        
        # Qubit0 DataFrame
        Q0_C1 = {
            '<x>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label1}'][:,0],
            '<y>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label1}'][:,1],
            '<z>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label1}'][:,2],
            'label': labels_1
        }
        Q0_C2 = {
            '<x>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label2}'][:,0],
            '<y>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label2}'][:,1],
            '<z>': self._dictQuantumStates[f'qubit{qubit}'][f'label{self._label2}'][:,2],
            'label': labels_2
        }
        qubit_0 = pd.concat([pd.DataFrame(data=Q0_C1), pd.DataFrame(data=Q0_C2)])

        ######################################## PLOT ########################################
        fig_Q0 = go.Figure()

        # Añadir scatter plot
        colors = {self._label1: self.colors[0], self._label2: self.colors[1]}
        fig_Q0.add_trace(go.Scatter3d(
            x=qubit_0['<x>'],
            y=qubit_0['<y>'],
            z=qubit_0['<z>'],
            mode='markers',
            marker=dict(
                size=5,
                color=qubit_0['label'].map(colors),  # Mapear colores según la clase
                opacity=0.5,
                symbol=qubit_0['label'].map({self._label1: "circle", self._label2: "circle"})  # Asignar símbolo según la clase
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
        
    
    def plot_all_qubits(self):
        """Plot all the interactive bloch spheres with the associates KL"""
        for qubit in range(self.qubits):
            self.plot_full_bloch_sphere(qubit)
            print("\n\n\n")
