from data_set import DataSet
from pca import PCA

print('*****************************************')
print('PCA Y K-MEANS')
print('*****************************************')

option_algorithm = int(input('Ingrese 1 si desea ejecutar el método PCA \no 2 si desea ejecutar el algoritmo K-MEANS: '))

data_set = DataSet()
data_set.load_data_set()

if option_algorithm == 1:
    pca = PCA()
    pca.process_data(data_set.data.values, data_set.num_instances)
