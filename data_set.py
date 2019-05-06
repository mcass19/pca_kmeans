import pandas as pd

class DataSet(object):
    
    def __init__(self):
        self.data = []
        self.num_instances = 0

    def load_data_set(self):    
        # self.data = pd.read_csv('data.csv', sep=',')
        self.data = pd.read_csv('dataPrueba.csv', sep=',')
        # ordenar por número de candidato -> ver si sirve para algo
        self.data.sort_values(by='candidatoId', inplace=True)

        # self.num_instances = 32499
        self.num_instances = 117
        
        del self.data['id']
        del self.data['candidatoId']
        del self.data['fecha']
