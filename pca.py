import numpy as np

class PCA(object):
    
    def __init__(self):
        pass

    def process_data(self, data_set):
        # transposición de matriz
        data = np.transpose(data_set)
        print('Matriz original:\n', data, '\n')

        # media
        mean = np.mean(data, axis=1)
        # varianza
        var = np.var(data, axis=1)
        # desviación estándar
        std = np.std(data, axis=1)
        
        print ("Media:", mean)
        print ("Varianza:", var)
        print ("Desviación estándar:", std, '\n')

        # se reescalan los valores, y el conjunto queda centrado en 0
        data_r = data - data.mean(axis=1, keepdims=True)
        print('Matriz luego de reescalado:\n', data_r, '\n')

        # media
        mean = np.mean(data_r, axis=1)
        # varianza
        var = np.var(data_r, axis=1)
        # desviación estándar
        std = np.std(data_r, axis=1)
        
        print ("Media:", mean)
        print ("Varianza:", var)
        print ("Desviación estándar:", std, '\n')

        # matriz de covarianza
        cov = np.cov(data)
        
        print('Matriz de covarianza:\n', cov, '\n')

        # valores y vectores propios
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        for i in range(len(eigen_values)):
            eigvec_cov = eigen_vectors[:,i].reshape(1,26).T
            print('Valor propio {} de la matriz de covarianza {}'.format(i+1, eigen_values[i]))
            print('Vector propio:')
            print(eigvec_cov)
        print('\n')

        # lista de tuplas (eigenvalue, eigenvector)
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

        # ordenar tuplas de mayor a menor
        eigen_pairs.sort()
        eigen_pairs.reverse()
        print('Valores propios con sus respectivos vectores propios, ordenado de forma descendente:')
        for i in eigen_pairs:
            print(i)
        print('\n')

        # Matriz w -> dos dimensiones, con los vectores propios en sus columnas
        matrix_w = np.hstack((eigen_pairs[0][1].reshape(26,1), eigen_pairs[1][1].reshape(26,1)))
        print('Matrix W:\n', matrix_w, '\n')
        print('\n')

        # transformar instancias al nuevo subespacio
        transformed = np.dot(data_r.T, matrix_w).transpose()
        print('Instancias transformadas:\n', transformed, '\n')

        # ploteo del resultado final
        # plt.plot(transformed[0,0:int(num_instances/2)], transformed[1,0:int(num_instances/2)], 'o', markersize=7, color='blue', alpha=0.5, label='men')
        # plt.plot(transformed[0,int(num_instances/2):num_instances], transformed[1,int(num_instances/2):num_instances], '^', markersize=7, color='red', alpha=0.5, label='women')
        # plt.xlabel('x_values')
        # plt.ylabel('y_values')
        # plt.title('Instancias transformadas (con etiquetas)')

        # plt.show()
