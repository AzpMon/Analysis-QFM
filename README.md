# Quantum Machine Learning: Quantum Feature Maps Analysis for classifications tasks.

***Code and results about the analysis   through  Kullback–Leibler divergence for optimized Quantum Feature Maps for Supervised Quantum Machine Learning via Kernel-Target Alignment using Pennylane, Pytorch and Plolty.
.***

## Description

This repository contains the code used for the analysis of classical data mappings to quantum states, optimizing the Quantum Feature Map through Target Kernel Alignment. Based on this mapping, a Kernel Matrix is computed via inner product between quantum states, which is then utilized in a Support Vector Machine trained using the Scikit-Learn library. Detailed analysis is conducted on each qubit by plotting qubit states individually. Subsequently, a sample space associated with each qubit is created, where for each class, a probability distribution is generated to compute the Kullback-Leibler divergence between binary classes. For data with more than two classes, a one-vs-one classification approach is employed, analyzing each qubit for every class pair.




## Contents
This repository is organized into the following folders:

1. Created classes and programs: Contains custom classes and programs developed for various tasks. 
2. Quantum embeddings training: Files for training quantum embeddings. Is based on the 
3. MakeCircles_Data: Data related to the makes_circles dataset.
4. Iris_Data: Data related to the Iris dataset.
5. Visual analysis and metrics: Contains scripts and notebooks for visual analysis and metric calculations.


