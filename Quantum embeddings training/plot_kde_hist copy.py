
import seaborn as sns
import matplotlib.pyplot as plt




def plot_2d_distributions(states, observable_1, observable_2, qubit):
    
    #Estilo oscuro
    plt.style.use('dark_background')


    
    
    # Set up a 1x2 grid for subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x1 = states['label1'][f'qubit{qubit}'][observable_1]
    y1 = states['label1'][f'qubit{qubit}'][observable_2]
    
    x2 = states['label2'][f'qubit{qubit}'][observable_1]
    y2 = states['label2'][f'qubit{qubit}'][observable_2]


    #KDE etiqueta azul (1)
    sns.kdeplot(x=x1, y=y1, cmap="mako", fill=True, label='Label 1', ax=axes[0], alpha = 0.5 )

    #KDE etiqueta verde (2)
    sns.kdeplot(x=x2, y=y2, cmap="viridis", fill=True, label='Label 2', ax=axes[0], alpha = 0.5)

    #Título y ejes
    axes[0].set_xlabel(f'<{observable_1}>')
    axes[0].set_ylabel(f'<{observable_2}>')
    axes[0].set_title(f'2D KDE Plot, qubit {qubit}')
    axes[0].legend()
    axes[0].grid()

    #Histoframa etiqueta azul (1)
    hist1 = axes[1].hist2d(x1, y1, bins=(30, 30), cmap='mako', alpha = 0.5)

    #Histoframa etiqueta verde (2)
    hist2 = axes[1].hist2d(x2, y2, bins=(30, 30), cmap='viridis', alpha=0.5) 

    #Título y ejes
    axes[1].set_xlabel(f'<{observable_1}>')
    axes[1].set_ylabel(f'<{observable_2}>')
    axes[1].set_title(f'2D Histogram, qubit {qubit}')
    axes[1].grid()

    #Colorbar de los histogramas
    cbar1 = plt.colorbar(hist1[3], ax=axes[1], label='Counts - Class 1')
    cbar2 = plt.colorbar(hist2[3], ax=axes[1], label='Counts - Class 2')

    plt.tight_layout()

    plt.show()
