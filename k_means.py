import numpy as np

from copy import deepcopy

from sklearn.cluster import KMeans
from sklearn import metrics

class Kmeans(object):
    
    def __init__(self, k, data_set):
        self.k = k
        self.data = data_set

        self.centers = np.zeros((self.k, len(data_set[0])))
        self.centers_library = np.zeros((self.k, len(data_set[0])))

        self.clusters = np.zeros(len(self.data))
        self.clusters_library = np.zeros(len(self.data))

    def cluster_data(self):
        # se eligen k instancias al azar para tomar como centros iniciales
        centers = np.zeros((self.k, len(self.data[0])))
        index_centers = np.random.randint(0, len(self.data), size=self.k)
        
        i = 0
        for j in index_centers:
            centers[i] = deepcopy(self.data[j])
            i += 1

        # estructura para guardar los centros viejos y luego calcular la 
        # distancia a los centros nuevos
        old_centers = np.zeros(centers.shape)

        # vector que indica en que número cluster se encuentra la instancia
        clusters = np.zeros(len(self.data))

        # error = distancia de centros nuevos a centros viejos
        error = self.distance(centers, old_centers)

        # loop hasta que la diferencia (error) entre los centros sea cero
        while error.all() != 0:
            # se asigna cada instancia a su centro más cercano
            for i in range(len(self.data)):
                distances = self.distance(self.data[i], centers)
                cluster = np.argmin(distances)
                clusters[i] = cluster

            # se guardan los centros viejos
            old_centers = deepcopy(centers)

            # se hallan los centros nuevos
            for i in range(self.k):
                points = []
                for j in range(len(self.data)):
                    if clusters[j] == i:
                        points.append(self.data[j])
                centers[i] = np.mean(points, axis=0)
            
            # se calcular distancia (error) a los centros nuevos
            error = self.distance(centers, old_centers)

        self.centers = centers
        self.clusters = clusters
        
        print('Data set: ', str(self.data), '\n')
        print('Error: ', str(error), '\n')
        print('Clusters: ', str(self.clusters), '\n')
        print('Centros: ', str(self.centers), '\n')

    # distancia euclidiana
    def distance(self, instance_a, instance_b):
        return np.linalg.norm(instance_a - instance_b, axis=1)

    # implementación librería scikit-learn
    def k_means_scikit_learn(self):
        kmeans = KMeans(n_clusters=self.k)
        kmeans = kmeans.fit(self.data)
        self.clusters_library = kmeans.labels_
        self.centers_library = kmeans.cluster_centers_
        print('Centros algoritmo librería: ', self.centers_library, '\n')

    def silhouette_coefficient(self):
        sc = metrics.silhouette_score(self.data, self.clusters, metric='euclidean')
        scl = metrics.silhouette_score(self.data, self.clusters_library, metric='euclidean')
        print('Coeficiente silhouette algoritmo propio: ', sc)
        print('Coeficiente silhouette algoritmo librería: ', scl, '\n')

    def adjusted_rand_index(self, votes_per_party):
        ari = metrics.adjusted_rand_score(votes_per_party, self.clusters) 
        aril = metrics.adjusted_rand_score(votes_per_party, self.clusters_library) 
        print('ARI algoritmo propio: ', ari)
        print('ARI algoritmo librería: ', aril, '\n')
