import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import dark_palette

def plot_2d_distributions(states, observable_1, observable_2, qubit):
    # Estilo oscuro
    plt.style.use('dark_background')

    # Configurar una cuadrícula 1x2 para los subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x1 = states['label1'][f'qubit{qubit}'][observable_1]
    y1 = states['label1'][f'qubit{qubit}'][observable_2]

    x2 = states['label2'][f'qubit{qubit}'][observable_1]
    y2 = states['label2'][f'qubit{qubit}'][observable_2]

    # KDE etiqueta azul (1) en el primer subplot (axes[0])
    #sns.kdeplot(x=x1, y=y1,cmap = 'mako', fill=True, ax=axes[0], alpha=0.8,warn_singular=False,levels=80)
    sns.kdeplot(x=x1, y=y1, cmap=dark_palette("cyan", as_cmap=True), fill=True, ax=axes[0], alpha=0.8,warn_singular=False,levels=80,thresh=0)
    # Limitar los ejes al rango [-1, 1]
    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.5])
    # Título y ejes
    axes[0].set_xlabel(f'<{observable_1}>')
    axes[0].set_ylabel(f'<{observable_2}>')
    axes[0].set_title(f'<{observable_1.upper()}> vs <{observable_2.upper()}>   (Class 0)')
    #axes[0].grid()

    # KDE etiqueta verde (2) en el segundo subplot (axes[1])
    #sns.kdeplot(x=x2, y=y2, cmap=dark_palette("green", as_cmap=True), fill=True, ax=axes[1], alpha=0.8,warn_singular=False,levels=80)
    sns.kdeplot(x=x2, y=y2, cmap=dark_palette("seagreen", as_cmap=True), fill=True, ax=axes[1], alpha=0.8,warn_singular=False,levels=80,thresh=0)
    # Limitar los ejes al rango [-1, 1]
    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([-1.5, 1.5])
    # Título y ejes
    axes[1].set_xlabel(f'<{observable_1}>')
    axes[1].set_ylabel(f'<{observable_2}>')
    axes[1].set_title(f' <{observable_1.upper()}> vs <{observable_2.upper()}>  (Class 1)')
    #axes[1].grid()

    plt.tight_layout()
    plt.suptitle(F'KDE PLOTS, QUBIT {qubit}', fontsize=16, y=1.02)

    plt.show()

