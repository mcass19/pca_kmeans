import pandas as pd

class DataSet(object):
    
    def __init__(self):
        self.data = []

    def load_data_set(self):    
        self.data = pd.read_csv('data.csv', sep=',')
        del self.data['id']
        del self.data['candidatoId']
        del self.data['fecha']
