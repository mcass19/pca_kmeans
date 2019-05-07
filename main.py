from data_set import DataSet
from pca import Pca
from k_means import Kmeans

print('*****************************************')
print('PCA Y K-MEANS')
print('*****************************************')

option_algorithm = int(input('Ingrese 1 si desea ejecutar el método PCA \no 2 si desea ejecutar el algoritmo K-MEANS: '))

print('\n')

data_set = DataSet()
data_prepared = data_set.prepare_data()
data_set.load_data_set(data_prepared)

if option_algorithm == 1:
    pca = Pca()
    pca.process_data(data_set.data.values, data_set.num_instances, data_set.cant_votes_per_candidate)
elif option_algorithm == 2:
    k_means = Kmeans()
    k_means.process_data()
