import numpy as np
np.set_printoptions(suppress=True, precision=3)

from plot import Plot

class Pca(object):
    
    def __init__(self):
        pass

    def process_data(self, data_set, num_instances, cant_votes_per_candidate):
        # transposición de matriz
        data = np.transpose(data_set)
        print('Matriz original:\n', data, '\n')

        # media
        mean = np.mean(data, axis=1)
        print ("Media:", mean)
        # varianza
        var = np.var(data, axis=1)
        print ("Varianza:", var)
        # desviación estándar
        std = np.std(data, axis=1)
        print ("Desviación estándar:", std, '\n')

        # se reescalan los valores, y el conjunto queda centrado en 0
        data_r = data - data.mean(axis=1, keepdims=True)
        print('Matriz luego de reescalado:\n', data_r, '\n')

        # media
        mean = np.mean(data_r, axis=1)
        print ("Media:", mean)
        # varianza
        var = np.var(data_r, axis=1)
        print ("Varianza:", var)
        # desviación estándar
        std = np.std(data_r, axis=1)
        print ("Desviación estándar:", std, '\n')
        
        # matriz de covarianza
        cov = np.cov(data)
        print('Matriz de covarianza:\n', cov, '\n')

        # valores y vectores propios
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        for i in range(len(eigen_values)):
            eig_vec = eigen_vectors[:,i].reshape(1,26)
            print('Valor propio {} de la matriz de covarianza {}'.format(i+1, eigen_values[i]))
            print('Vector propio:', eig_vec)
        print('\n')

        # lista de tuplas (eigenvalue, eigenvector)
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

        # ordenar tuplas de mayor a menor
        eigen_pairs.sort(reverse=True)
        print('Valores propios con sus respectivos vectores propios, ordenados de forma descendente:')
        for i in eigen_pairs:
            print(i)
        print('\n')

        # Matriz w -> dos dimensiones, con los vectores propios en sus columnas
        matrix_w = np.hstack((eigen_pairs[0][1].reshape(26,1), eigen_pairs[1][1].reshape(26,1)))
        print('Matriz W:\n', matrix_w, '\n')

        # transformar instancias al nuevo subespacio
        instances_transformed = np.dot(data_r.T, matrix_w).transpose()
        print('Instancias transformadas:\n', instances_transformed, '\n')

        # ploteo del resultado final
        plot = Plot()
        plot.plot_pca(instances_transformed, cant_votes_per_candidate)
