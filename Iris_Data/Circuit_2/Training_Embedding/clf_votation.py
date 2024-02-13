
from sklearn.svm import SVC
import numpy as np

import numpy as np
from sklearn.svm import SVC

def predicciones_finales_votación(clasificadores, X_set):


    # Obtener las predicciones de cada clasificador para el conjunto de datos
    predicciones_clasificadores = [clasificador.predict(X_set) for clasificador in clasificadores]

    # Inicializar array para almacenar las predicciones finales
    prediccion_final = np.zeros_like(predicciones_clasificadores[0])

    # Votar por mayoría
    for i in range(len(X_set)):
        votos = [prediccion[i] for prediccion in predicciones_clasificadores]
        clases_posibles = np.unique(votos)
        votos_por_clase = np.zeros(len(clases_posibles))

        for j, clase in enumerate(clases_posibles):
            votos_por_clase[j] = np.sum(np.array(votos) == clase)

        clase_ganadora = clases_posibles[np.argmax(votos_por_clase)]
        prediccion_final[i] = clase_ganadora
        
    return prediccion_final



