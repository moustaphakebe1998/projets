import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# from tranformation import Load

class Echantillonnage : 
    ''' 
    Cette classe fournit les méthodes d'échantionnage ( under, over et SMOTE)
    '''
    def __init__(self):
        pass

    
    def under_sampling(self, X, y) :

        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        self.plot_distribution(y, "Distribution originale")
        self.plot_distribution(y_res, "Distribution après équilibrage undersampling")

        return X_res, y_res

    def over_sampling(self, X, y) : 

        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)

        self.plot_distribution(y, "Distribution originale")
        self.plot_distribution(y_res, "Distribution après équilibrage oversampling")

        return X_res, y_res

    def smote_sampling(self, X, y) :

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        self.plot_distribution(y, "Distribution originale")
        self.plot_distribution(y_res, "Distribution après équilibrage SMOTE")

        return X_res, y_res
    
    def plot_distribution(self, data, title):

        # Distribution des classes
        plt.figure(figsize=(8, 6))
        plt.title(title)
        sns.countplot(x=data.iloc[:, -1])
        plt.title('Distribution des classes')
        plt.show()

class Processing : 

    ''' 
    Cette classe nous permet entre autre de faire le preprocessing 
    à savoir réduction de dimension, standardiser au besoin, spliter
    des données en ensemble d'entrainement et de test mais aussi
    d'effectuer une validation croisée kfold.   
    '''
    def __init__(self):
        pass
    
    #réduction de dimension
    def dimred_with_pca(self, X, y, n_components=2) : 

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)

        if n_components == 2 : 

            df_pca = pd.DataFrame(pca_result, columns=['x', 'y'])
            df_pca['gnd'] = y

            pca_chart = alt.Chart(df_pca.iloc[:5000]).mark_circle(size=100).encode(
                x='x:Q',
                y='y:Q',
                color='gnd:N',
            ).properties(
                width=300,
                height=300,
                title="Projection PCA des mots-clés"
            ).interactive()

            return df_pca, pca_chart

        if n_components == 3 : 

            df_pca = pd.DataFrame(pca_result, columns=['x', 'y','z'])
            df_pca['gnd'] = y

            import plotly.express as px

            # 3D PCA chart
            fig_pca = px.scatter_3d(df_pca, x='x', y='y', z='z', color='gnd', title="3D PCA Projection")
            fig_pca.show()

            return df_pca
        
    #spliterdes données
    def train_test(self, X, y) : 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42, stratify = y)

        return X_train, X_test, y_train, y_test
    
    #standardiser
    def scaler(self, data) : 

        scaler = StandardScaler()
        rescaledX = scaler.fit_transform(data)

        return rescaledX

    #validation croisée kfold. 
    def cross_val(self, models, num_folds, X_train, Y_train) :

        results = []
        names = []
        scoring = "accuracy"
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        return results


        