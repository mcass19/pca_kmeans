from data_set import DataSet
from pca import Pca
from k_means import Kmeans

print('*****************************************')
print('PCA Y K-MEANS')
print('*****************************************')

option_algorithm = int(input('Ingrese 1 si desea ejecutar el método PCA \no 2 si desea ejecutar el algoritmo K-MEANS: '))

print('\n')

# manejo de los datos
data_set = DataSet()
data_prepared = data_set.prepare_data()
data_set.load_data_set(data_prepared)

if option_algorithm == 1:
    pca = Pca()
    pca.process_data(data_set.data.values, data_set.num_instances, data_set.cant_votes_per_party)
elif option_algorithm == 2:
    k = int(input('Ingrese la cantidad de clusters (k): '))
    print('\n')
    
    k_means = Kmeans(k, data_set.data.values)
    k_means.cluster_data()

    # comparación de resultados contra la implementación de kmeans de la librería scikit-learn
    k_means.k_means_scikit_learn()

    # coeficiente silhouette
    k_means.silhouette_coefficient()

    if (k == 11):
        # ARI
        k_means.adjusted_rand_index(data_set.votes_per_party)
