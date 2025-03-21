import pandas as pd
import numpy as np
import scipy.io


class Load :

    ''' 
        Partie 2 transformation et calcule des Matrices sur les quelles on va travailler en 
        l'occurrence M et fea
    '''

    def __init__(self):
        pass

    #on déclare cette méthode comme étant privée vu que son utilité reste interne
    def _transform_data(self,filename) : 

        ''' 
        Changement des fichiers .mat afin de récuperér "fea, gnd et W" 
        @Mouhamadou Lamine GNING
        '''
        file = scipy.io.loadmat(filename)
        fea = file['fea']
        gnd = file['gnd']
        W = file['W']
        
        return fea, gnd , W

    #on déclare de même cette méthode comme étant privée pour les même raisons
    def _data_to_df(self,fea, gnd, W) :
        fea_df = pd.DataFrame(fea)
        gnd_df = pd.DataFrame(gnd)

        return fea_df, gnd_df, W

    def load_data(self,filename) : 

        ''' 
        Calcul des Matrices D et M sur les quelles on va travailler M
        les données sur les quelles porteront cette étude sont fea et M
        '''

        fea, gnd, W = self._transform_data(filename)
        fea, gnd, W = self._data_to_df(fea, gnd, W)

        D = np.zeros((fea.shape[0],fea.shape[0]))
        for i in range(fea.shape[0]) : 
            D[i][i] = W[i].sum()
        M = np.linalg.inv(D).dot(W.dot(fea))
            
        return fea, gnd, W, D, M