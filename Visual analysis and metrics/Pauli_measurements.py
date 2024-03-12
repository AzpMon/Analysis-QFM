import pennylane as qml
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from seaborn import dark_palette,light_palette
import plotly.graph_objects as go
from scipy.stats import entropy
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

#Classic_data : np.columns_stack((input_array, target_array))

#Corregir el uso de los QNodes de pennylane para los cirucitos de los Kernels
class Pauli_measurements:
    def __init__(self, ansatz, layers, params, class1, class2, qubits=2, label_color1='cyan', label_color2="green",mesh_samples = 10000):
        """Initialize the Pauli_measurements class.
        Args:
            ansatz (qml.function): Quantum circuit ansatz.
            layers (int): Number of layers in the ansatz.
            params (torch.tensor): Parameters for the quantum circuit.
            class1 (torch.tensor): Data for class 1.
            class2 (torch.tensor): Data for class 2.
            qubits (int, optional): Number of qubits. Defaults to 2.
            label_color1 (str, optional): Color for class 1. Defaults to 'cyan'.
            label_color2 (str, optional): Color for class 2. Defaults to 'green'.
            mesh_samples (int, optional): Number of samples for mesh grid. Defaults to 10000.
            sns_palette (str, optional): Seaborn color palette ('dark_palette' or 'light_palette'). Defaults to 'dark_palette'.
            plt_style (str, optional): Matplotlib style. Defaults to 'dark_background'.
        """
        # Data 
        self.classic_dataset1 = class1
        self.classic_dataset2 = class2
        self.label_color1 = label_color1
        self.label_color2 = label_color2
        self.len_class1 = self.classic_dataset1.shape[0]
        self.len_class2 = self.classic_dataset2.shape[0]
        
        #Plot configuration
        self.label_color1 = label_color1
        self.label_color1 = label_color1

        # Circuit params
        self.ansatz = ansatz
        self.qubits = qubits
        self.layers = layers
        self.params = params
        
        

        # QNodes to use
        self.qml_device = qml.device(name = "default.qubit", wires = self.qubits)



        self.__expX_Q0 = qml.QNode(func=self.__circuito_expX_Q0, device=self.qml_device, interface="torch")
        self.__expY_Q0 = qml.QNode(func=self.__circuito_expY_Q0, device=self.qml_device, interface="torch")
        self.__expZ_Q0 = qml.QNode(func=self.__circuito_expZ_Q0, device=self.qml_device, interface="torch")
        self.__expX_Q1 = qml.QNode(func=self.__circuito_expX_Q1, device=self.qml_device, interface="torch")
        self.__expY_Q1 = qml.QNode(func=self.__circuito_expY_Q1, device=self.qml_device, interface="torch")
        self.__expZ_Q1 = qml.QNode(func=self.__circuito_expZ_Q1, device=self.qml_device, interface="torch")
        
        
        #Quantum states of all the dataset
        self.quantum_states= self.quantum_states()
        
        #Quibt0, label1 data
        self._qubit0_label1_data = self.quantum_states['label1']['qubit0']
        self.qubit0_label1 = np.column_stack((self._qubit0_label1_data['x'], self._qubit0_label1_data['y'], self._qubit0_label1_data['z']))
        #Quibt0, label2 data
        self._qubit0_label2_data = self.quantum_states['label2']['qubit0']
        self.qubit0_label2 = np.column_stack((self._qubit0_label2_data['x'], self._qubit0_label2_data['y'], self._qubit0_label2_data['z']))
        #Quibt1, label1 data
        self._qubit1_label1_data = self.quantum_states['label1']['qubit1']
        self.qubit1_label1 = np.column_stack((self._qubit1_label1_data['x'], self._qubit1_label1_data['y'], self._qubit1_label1_data['z']))
        #Quibt1, label2 data
        self._qubit1_label2_data = self.quantum_states['label2']['qubit1']
        self.qubit1_label2 = np.column_stack((self._qubit1_label2_data['x'], self._qubit1_label2_data['y'], self._qubit1_label2_data['z']))

        #Dictionary with all the data, per qubit and label
        self._qubits_data = {
            'qubit0': {'label1':self.qubit0_label1,'label2':self.qubit0_label2},
            'qubit1': {'label1':self.qubit1_label1,'label2':self.qubit1_label2},
                    }
        
        
        
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
        
        
        
        #All the posible KDE_pdf for each qubit, label and pair of Pauli Operators. (Evaluetes in a single 2D data)
        self.pdf_qubit0_label1_xy = self._kde_pdfs(qubit=0, coord1='x', coord2='y')[0]      #Qubit0, label1, xy
        self.pdf_qubit0_label1_xz = self._kde_pdfs(qubit=0, coord1='x', coord2='z')[0]      #Qubit0, label1, xz
        self.pdf_qubit0_label1_yz = self._kde_pdfs(qubit=0, coord1='y', coord2='y')[0]      #Qubit0, label1, yz
        self.pdf_qubit0_label2_xy = self._kde_pdfs(qubit=0, coord1='x', coord2='y')[1]      #Qubit0, label2, xy
        self.pdf_qubit0_label2_xz = self._kde_pdfs(qubit=0, coord1='x', coord2='z')[1]      #Qubit0, label2, xz
        self.pdf_qubit0_label2_yz = self._kde_pdfs(qubit=0, coord1='y', coord2='y')[1]      #Qubit0, label2, yz
    
        self.pdf_qubit1_label1_xy = self._kde_pdfs(qubit=1, coord1='x', coord2='y')[0]      #Qubit1, label1, xy
        self.pdf_qubit1_label1_xz = self._kde_pdfs(qubit=1, coord1='x', coord2='z')[0]      #Qubit1, label1, xz
        self.pdf_qubit1_label1_yz = self._kde_pdfs(qubit=1, coord1='y', coord2='y')[0]      #Qubit1, label1, yz
        self.pdf_qubit1_label2_xy = self._kde_pdfs(qubit=1, coord1='x', coord2='y')[1]      #Qubit1, label2, xy
        self.pdf_qubit1_label2_xz = self._kde_pdfs(qubit=1, coord1='x', coord2='z')[1]      #Qubit1, label2, xz
        self.pdf_qubit1_label2_yz = self._kde_pdfs(qubit=1, coord1='y', coord2='y')[1]      #Qubit1, label2, yz
        
        #Dictionary with all the KDE_pdf 2D
        self._pdfs = {
            'qubit0': {
                'label1': {'xy':self.pdf_qubit0_label1_xy,'xz':self.pdf_qubit0_label1_xz,'yz':self.pdf_qubit0_label1_yz },
                'label2': {'xy':self.pdf_qubit0_label2_xy,'xz':self.pdf_qubit0_label2_xz,'yz':self.pdf_qubit0_label2_yz }
                        },
            'qubit1': {
                'label1': {'xy':self.pdf_qubit1_label1_xy,'xz':self.pdf_qubit1_label1_xz,'yz':self.pdf_qubit1_label1_yz },
                'label2': {'xy':self.pdf_qubit1_label2_xy,'xz':self.pdf_qubit1_label2_xz,'yz':self.pdf_qubit1_label2_yz }
                        }
                    }
    
        
        #All the 2D-Kullback Leilbler divergences for each qubit and pair of Pauli Operators.
        self.kl_qubit0_xy = self.kl_divergence_2D('x', 'y', 0)[0]          #KL,  Qubit0, xy
        self.kl_qubit0_xz = self.kl_divergence_2D('x', 'z', 0)[0]          #KL,  Qubit0, xz
        self.kl_qubit0_yz = self.kl_divergence_2D('y', 'z', 0)[0]          #KL,  Qubit0, yz
        
        self.kl_qubit1_xy = self.kl_divergence_2D('x', 'y', 1)[0]          #KL,  Qubit1, xy
        self.kl_qubit1_xz = self.kl_divergence_2D('x', 'z', 1)[0]          #KL,  Qubit1, xz
        self.kl_qubit1_yz = self.kl_divergence_2D('y', 'z', 1)[0]          #KL,  Qubit1, yz
        
        #Dictionary with all the 2D-Kullback Leibler divergences
        self._kl_2D = {
            'qubit0': {'xy':self.kl_qubit0_xy,'xz':self.kl_qubit0_xz,'yz':self.kl_qubit0_yz },
            'qubit1': {'xy':self.kl_qubit1_xy,'xz':self.kl_qubit1_xz,'yz':self.kl_qubit1_yz }     
                    }
        
        
        
        
        # Sample Space for the bloch sphare
        self.mesh_samples = mesh_samples
        self._phi = np.random.uniform(0, 2 * np.pi, self.mesh_samples)
        self._theta = np.arccos(2 * np.random.uniform(0, 1, self.mesh_samples) - 1)
        self._r = np.cbrt(np.random.uniform(0, 1, self.mesh_samples))

        self._sphere_x = self._r * np.sin(self._theta) * np.cos(self._phi)
        self._sphere_y = self._r * np.sin(self._theta) * np.sin(self._phi)
        self._sphere_z = self._r * np.cos(self._theta)

        # Create sphere sample space
        self._sphere_sample_space = np.column_stack((self._sphere_x, self._sphere_y, self._sphere_z))
        self._sphere_vol_diff = (4 * np.pi / 3) / self.mesh_samples  #Aprox.
        
        #All the posible KDE_pdf for each qubit and label (Evaluetes in a single 3D data)
        self.pdf_qubit0_label1_bloch = self._kde_pdfs_qubit(qubit=0)[0]
        self.pdf_qubit0_label2_bloch = self._kde_pdfs_qubit(qubit=0)[1]
        self.pdf_qubit1_label1_bloch = self._kde_pdfs_qubit(qubit=1)[0]
        self.pdf_qubit1_label2_bloch = self._kde_pdfs_qubit(qubit=1)[1]
        
        #Dictionary with all the KDE_pdf 3D
        self._bloch_sphere_pdfs={
            'qubit0':{
                'label1':  self.pdf_qubit0_label1_bloch,
                'label2': self.pdf_qubit0_label2_bloch
                },
            'qubit1':{
                'label1': self.pdf_qubit1_label1_bloch,
                'label2': self.pdf_qubit1_label2_bloch
                    }
                        }
        
        
        #All the 3D-Kullback Leilbler divergences for each qubit and pair of Pauli Operators.
        self.kl_qubit0 = self.kl_divergence_qubit(qubit=0)        #KL,  Qubit0
        self.kl_qubit1 = self.kl_divergence_qubit(qubit=1)        #KL,  Qubit1
        
        #Dictionary with all the 2D-Kullback Leibler divergences
        self._kl_3D = {'qubit0': self.kl_qubit0,'qubit1': self.kl_qubit1}
        
        






    #Computations of the different Pauli expected values in different qubits
    def __circuito_expX_Q0(self, dato):
        """Private method to perform a PauliX quantum measurement on qubit 0.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""    
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliX(0))
    def __circuito_expY_Q0(self, dato):
        """Private method to perform a PauliY quantum measurement on qubit 0.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliY(0))
    def __circuito_expZ_Q0(self, dato):
        """Private method to perform a PauliZ quantum measurement on qubit 0.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliZ(0))
    def __circuito_expX_Q1(self, dato):
        """Private method to perform a PauliY quantum measurement on qubit 1.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliX(1))
    def __circuito_expY_Q1(self, dato):
        """Private method to perform a PauliZ quantum measurement on qubit 1.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliY(1))
    def __circuito_expZ_Q1(self, dato):
        """Private method to perform a PauliZ quantum measurement on qubit 1.
        Args:
             dato (torch.tensor): Data for performing the measurement.
        Returns:
            torch.tensor: The result of the measurement."""
        self.ansatz(self.qubits, self.layers, self.params, dato)
        return qml.expval(qml.PauliZ(1))
    
    
    
    #Compute the measurements of the final mapped data point.
    def make_measurements(self, dato):
        """Perform quantum measurements and return the results.
        Args:
            dato (torch.tensor): Data for performing the measurements.

        Returns:
            dict: A dictionary with measurements for qubit0 and qubit1: 
        """
        return {'qubit0': [self.__expX_Q0(dato), self.__expY_Q0(dato), self.__expZ_Q0(dato)],
                'qubit1': [self.__expX_Q1(dato), self.__expY_Q1(dato), self.__expZ_Q1(dato)]}
    
 
 
    #Computes the quantum states associated to all the dataset after the QFM    
    def quantum_states(self):
        """Get the quantum states of all the data set
        Returns:
            dict: A dictionary with measurements for each class-data; dict[labeln][qubitm]['component'] for 
                    n label (1,2) and m qubit(0,1)
        """
        #Measurements class1
        mediciones_x0_1=[];mediciones_y0_1=[];mediciones_z0_1=[]
        mediciones_x1_1=[];mediciones_y1_1=[];mediciones_z1_1=[]
        #Measurements class2
        mediciones_x0_2=[];mediciones_y0_2=[];mediciones_z0_2=[]
        mediciones_x1_2=[];mediciones_y1_2=[];mediciones_z1_2=[]
        #Make measurements class1
        for dato in self.classic_dataset1:
            res = self.make_measurements(dato)
            #Qubit 0
            mediciones_x0_1.append(res['qubit0'][0])
            mediciones_y0_1.append(res['qubit0'][1])
            mediciones_z0_1.append(res['qubit0'][2])           
            #Qubit 1
            mediciones_x1_1.append(res['qubit1'][0])
            mediciones_y1_1.append(res['qubit1'][1])
            mediciones_z1_1.append(res['qubit1'][2])     
        #Make measurements class2 
        for dato in self.classic_dataset2:
            res2 = self.make_measurements(dato)
            #Qubit 0
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
    
   
    
    # 1QUBIT, 2 PAULI MEASUREMENTS DIMENSIONS
    #Computes the preNormalized PDF calculated from the Grid-Search KernelDensity, 
    def _kde_pdf_estimation_2D(self, dataset):
        """Using grid search fot the bandwidht computes the KernelDensity for the given dataset.Then
        evaluates the pdf in the sample space, returning the sum of its probabilities
        Args:
            dataset (torch.tensor): Dataset to obtain the bw and kde.
        Returns:
            numpy.float64: 
        """
        #Grid  search (for the bw) 
        params = {'bandwidth': np.linspace(0.01, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(dataset)
        
        #KDE 
        best_kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'])
        best_kde.fit(dataset)
        
        #Sample probabilities vol 
        sample_probs = np.sum(np.exp(best_kde.score_samples(self.sample_space)))
        
        def calculate_probability(data):
            data = data.reshape(1,-1)
            if np.all(data <= 1) and np.all(data>= -1):
                return np.exp(best_kde.score_samples(data)) / sample_probs
            else:
                return 0
        return  calculate_probability
      
        
    #Computes the pdf using KDE for a determine qubit and pair of coordinates
    def _kde_pdfs(self, qubit, coord1, coord2):
        """Private method to obtain  the pdf in the selected qubit for the coord1, coord2 for all the classical dataset
        e.g: qubit=1, coord1='x', coord2='y' computes the pdf_xy for label1 and label2
        Args:
            qubit  (int): Considered qubit for the KDE pdf
            coord1 (str): Pauli_expected_value1 to be consider ('x','y','z')
            coord2 (str): Pauli_expected_value2 to be consider ('x','y','z')
        Returns:
            list: kde_pdf_label1, kde_pdf_label2 
        """
        #Label1 dataset in the coord1,coord2 space 
        data_u_label1 = self.quantum_states['label1'][f'qubit{qubit}'][f'{coord1}']
        data_v_label1 = self.quantum_states['label1'][f'qubit{qubit}'][f'{coord2}']
        data_label1   = np.column_stack((data_u_label1,data_v_label1 ))

        #KDE estimation using grid search for label1 data, return the pdf to be evaluate in a datapoint
        kde_pdf_label1 = self._kde_pdf_estimation_2D(data_label1)
        
        
        #Label1 dataset in the coord1_coord2 space 
        data_u_label2 = self.quantum_states['label2'][f'qubit{qubit}'][f'{coord1}']
        data_v_label2 = self.quantum_states['label2'][f'qubit{qubit}'][f'{coord2}']
        data_label2   = np.column_stack((data_u_label2,data_v_label2 ))

        #KDE estimation using grid search for label2 data, return the pdf to be evaluate in a datapoint
        kde_pdf_label2 = self._kde_pdf_estimation_2D(data_label2)
        
        return kde_pdf_label1, kde_pdf_label2       
    
    

    def kl_divergence_2D(self, observable_1, observable_2, qubit):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in 2D.
        Args:
            observable_1 (str): Pauli observable for the first distribution.
            observable_2 (str): Pauli observable for the second distribution.
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """
        pdf1 = self._pdfs[f'qubit{qubit}']['label1'][f'{observable_1+observable_2}']
        pdf2 = self._pdfs[f'qubit{qubit}']['label2'][f'{observable_1+observable_2}']

        kl_divergence = 0.0
        for i in range(self._len_samples):
            for j in range(self._len_samples):
                sample = np.array([self._u_mesh[i], self._v_mesh[j]])
                prob1 = pdf1(sample)
                prob2 = pdf2(sample)
                if prob1 > 0 and prob2 > 0:
                    kl_divergence += prob1 * np.log(prob1 / prob2)

        return kl_divergence
    
    
    def _kde_pdfs_qubit(self, qubit):
        """Using grid search fot the bandwidht computes the KernelDensity for the all the original dataset.Then
        evaluates the pdf in the sample space, returning the pdf to be evaluate in a single datapoint (quantum state)
        Args:
            qubit (int): Index of qubit
        Returns:
            list: kde_pdf_label1, kde_pdf_label2 
        """
        ####################       Label1 Data         ####################
        #Grid  search (for the bw)  fot the dataset 1
        dataset1 = self._qubits_data[f'qubit{qubit}']['label1']
        params1 = {'bandwidth': np.linspace(0.01, 1, 20)}
        grid1 = GridSearchCV(KernelDensity(), params1, cv=5)
        grid1.fit(dataset1)
        
        #KDE label1 data
        best_kde1 = KernelDensity(bandwidth=grid1.best_params_['bandwidth'])
        best_kde1.fit(dataset1)  
        
        #Sample probabilities bloch_sphere sample space
        sample_probs1 = self._sphere_vol_diff * np.sum(np.exp(best_kde1.score_samples(self._sphere_sample_space)))
        
        
        
        ####################       Label2 Data         ####################
        #Grid  search (for the bw)  fot the dataset 2
        dataset2 = self._qubits_data[f'qubit{qubit}']['label2']
        params2 = {'bandwidth': np.linspace(0.01, 1, 20)}
        grid2 = GridSearchCV(KernelDensity(), params2, cv=5)
        grid2.fit(dataset2)
        
        #KDE label2 data
        best_kde2 = KernelDensity(bandwidth=grid2.best_params_['bandwidth'])
        best_kde2.fit(dataset2)  
        
        #Sample probabilities bloch_sphere sample space
        sample_probs2 = self._sphere_vol_diff * np.sum(np.exp(best_kde2.score_samples(self._sphere_sample_space)))
        
        
        #PDF normalized and with physical restrictions
        def calculate_probability_label1(data):
            data = data.reshape(1,-1)
            if np.all(data <= 1) and np.all(data>= -1):
                return np.exp(best_kde1.score_samples(data)) / sample_probs1
            else:
                return 0
            
        def calculate_probability_label2(data):
            data = data.reshape(1,-1)
            if np.all(data <= 1) and np.all(data>= -1):
                return np.exp(best_kde2.score_samples(data)) / sample_probs2
            else:
                return 0
            
            
        return  calculate_probability_label1, calculate_probability_label2   
    
    
    def kl_divergence_qubit(self,qubit):
        """Calculate the Kullback-Leibler divergence between two quantum probability density functions in the correspond qubit.
        Args:
            qubit (int): Index of the qubit.
        Returns:
            float: Kullback-Leibler divergence.
        """        
        pdf1 = self._bloch_sphere_pdfs[f'qubit{qubit}']['label1']
        pdf2 = self._bloch_sphere_pdfs[f'qubit{qubit}']['label2']

        p1_probs = np.array([pdf1(sample) for sample in self._sphere_sample_space])
        p2_probs = np.array([pdf2(sample) for sample in self._sphere_sample_space])
        
        p1_probs = p1_probs[p1_probs > 0]
        p2_probs = p2_probs[p2_probs > 0]
        
        #KL divergence using scipy.stats.entropy
        kl_bloch_sphere = entropy(p1_probs, p2_probs)



        return kl_bloch_sphere 
    
    
    
    
    
    
    
    
    
    
    
        


    def plot_quantum_pdf(self, observable_1, observable_2, qubit,  figsize=(14, 6), levels = 20, save_fig=False, plot_name="ex",
                          sns_palette = 'dark_palette', plt_style = 'dark_background'):
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
       
        plt.style.use(f'{plt_style}')      #Matplotlib style
        #Cmap configuration
        if sns_palette == 'dark_palette':
            cmap1 = dark_palette(f'{self.label_color1}', as_cmap=True)
            cmap2 = dark_palette(f'{self.label_color2}', as_cmap=True)
        if sns_palette == 'ligh_palette':
            cmap1 = light_palette(f'{self.label_color1}', as_cmap=True)
            cmap2 = light_palette(f'{self.label_color1}', as_cmap=True)

        
        #KDE-pdfs
        pdf1 = self._pdfs[f'qubit{qubit}']['label1'][f'{observable_1+observable_2}']
        pdf2 = self._pdfs[f'qubit{qubit}']['label2'][f'{observable_1+observable_2}']
        
        #Probabilities
        probs1 = np.zeros_like(self._U)
        probs2 = np.zeros_like(self._U)

        #Computation of the mesh probabilities 
        kl_divergence = 0.0
        for i in range(self._len_samples):
            for j in range(self._len_samples):
                sample = np.array([self._u_mesh[i], self._v_mesh[j]])
                probs1[i, j] = pdf1(sample)         #Mesh probabilities label1 pdf
                probs2[i, j] = pdf2(sample)         #Mesh probabilities label2 pdf
                
        
        #KL Divergence 
        kl = self._kl_2D[f'qubit{qubit}'][f'{observable_1+observable_2}']        
                
        
        
        #Subplot configuration
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        #ContourPlot label1
        axs[0].contourf(self._U, self._V, probs1, cmap=cmap1, levels=levels)
        axs[0].set_title('Quantum KDE-PDF (label 1)')
        axs[0].set_xlabel(f'$\\langle {observable_1.upper()} \\rangle$')
        axs[0].set_ylabel(f'$\\langle {observable_2.upper()} \\rangle$')
        axs[0].set_xlim([-1, 1])
        axs[0].set_ylim([-1, 1])

        #ContourPlot label1
        axs[1].contourf(self._U, self._V, probs2, cmap=cmap2, levels=levels)
        axs[1].set_title('Quantum KDE-PDF (label 2)')
        axs[1].set_xlabel(f'$\\langle {observable_1.upper()} \\rangle$')
        axs[1].set_ylabel(f'$\\langle {observable_2.upper()} \\rangle$')
        axs[1].set_xlim([-1, 1])
        axs[1].set_ylim([-1, 1])
        
        plt.suptitle(f'QUBIT {qubit}\n KL = {round(kl,5)}', fontsize=16, fontweight='bold', y=1.01)

        
        #Save the figure
        if save_fig == True:
            plt.savefig(f'{plot_name}.eps', format='eps', bbox_inches='tight')

        plt.tight_layout()
        plt.show()
    

   
  
    
    
    
    #Plottly interactive plot of the optimized circuit applied to the dataset
    def plot_interactive_quantumSpace(self):
        """Make an interactive plot of the Hilbert spaces
        """
        
        
        kl_0 =  self._kl_3D['qubit0']
        kl_1 =  self._kl_3D['qubit1']

        labels_1 =  np.ones(self.len_class1)
        labels_2 = -1* np.ones(self.len_class1)
        #Qubit0 DataFrame
        Q0_C1 = {'<x>': self.quantum_states['label1']['qubit0']['x'],
                 '<y>': self.quantum_states['label1']['qubit0']['y'],
                 '<z>': self.quantum_states['label1']['qubit0']['z'],
                 'label': labels_1}
        Q0_C2 = {'<x>': self.quantum_states['label2']['qubit0']['x'],
                 '<y>': self.quantum_states['label2']['qubit0']['y'],
                 '<z>': self.quantum_states['label2']['qubit0']['z'],
                 'label': labels_2}
        qubit_0 = pd.concat([pd.DataFrame(data=Q0_C1), pd.DataFrame(data=Q0_C2)])

        #Qubit1 DataFrame
        Q1_C1 = {'<x>': self.quantum_states['label1']['qubit1']['x'],
                 '<y>': self.quantum_states['label1']['qubit1']['y'],
                 '<z>': self.quantum_states['label1']['qubit1']['z'],
                 'label': labels_1}
        Q1_C2 = {'<x>': self.quantum_states['label2']['qubit1']['x'],
                 '<y>': self.quantum_states['label2']['qubit1']['y'],
                 '<z>': self.quantum_states['label2']['qubit1']['z'],
                 'label': labels_2}
        qubit_1 = pd.concat([pd.DataFrame(data=Q1_C1), pd.DataFrame(data=Q1_C2)])

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
                color=qubit_0['label'].map(colors),  # Mapear colores según la clase
                opacity=0.5,
                symbol=qubit_0['label'].map({1: "circle", -1: "circle"})  # Asignar símbolo según la clase
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
                xaxis=dict(title='<X>'),  # Título del eje X
                yaxis=dict(title='<Y>'),  # Título del eje Y
                zaxis=dict(title='<Z>'),  # Título del eje Z
            ),
            template="plotly_dark",
            annotations=[
                dict(
                    text=f'QUBIT 0, KL={round(kl_0,4)}',
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
                color=qubit_0['label'].map(colors),  # Mapear colores según la clase
                opacity=0.5,
                symbol=qubit_0['label'].map({1: "circle", -1: "circle"})  # Asignar símbolo según la clase
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
                xaxis=dict(title='<X>'),  # Título del eje X
                yaxis=dict(title='<Y>'),  # Título del eje Y
                zaxis=dict(title='<Z>'),  # Título del eje Z
            ),
            template="plotly_dark",
            annotations=[
                dict(
                    text=f'QUBIT 1, KL={round(kl_1,4)}',
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.99,  # Ajustar las coordenadas x e y para centrar el título
                    font=dict(size=14)
                )
            ]
        )
        fig_Q0.show()
        fig_Q1.show() 
        
    def plot_all(self):
        qubits = [0,1]
        coordinates = [['x','y'], ['x','z'], ['y','z']]
        
        for qubit in qubits:
            for coord in coordinates:
                self.plot_quantum_pdf(observable_1=coord[0], observable_2=coord[1], qubit=qubit)
        
        self.plot_interactive_quantumSpace()