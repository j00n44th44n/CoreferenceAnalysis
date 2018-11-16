# -*- coding: utf8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from scipy.stats import mode
from os.path import exists 
from os import getcwd

import dataset

import nltk 

import matplotlib.pyplot as plt





class Collective_Classifier():
    def __init__(self, selection=None):
        self.cls = []
        self.clsBool = None
        self.dict = {
                        0: (DecisionTreeClassifier(criterion='gini'), 'tree_with_gini_1'),
                        1: (DecisionTreeClassifier(criterion='entropy'), 'tree_with_entropy_2'),
                        2: (BernoulliNB(), 'bernoulli'),
                        3: (MultinomialNB(), 'multinimial'),
                        4: (SVC(kernel='poly'), 'svc_with_poly')
                    }

        if not self.load():
            self.__select_classifier(selection)
            self.fit(dataset.features_train, dataset.corefers_train)
            self.save()
    
    def __select_classifier(self, selection):
        """
        Selección de los clasificadores. Se disponen:
            - árbol de decisión, con criterio de búsqueda de impurezas de Gini
            - árbol de decisión, con criterio de búsqueda de máxima entropía
            - modelo probabilístico, usando multinomial
            - modelo probabilístico, usando Bernoulli
            - clasificación por soportes C (SVC), con kernel polinomal
        param selection: Vector booleano donde en la posición i-ésima si indicará 
                         si el clasificador i-ésimo ayudará a decidir o no, en el 
                         momento de clasificar. Si no hay ninguno activado, entonces
                         se tomará por defecto el segundo clasificador. En caso de
                         ser None, entonces se tomarán todos los clasificadores.
        """
        countCls = len(self.dict)
        if selection is None or len(selection) != countCls:
            selection = [True for _ in range(countCls)]
            
        self.clsBool = selection.copy()
        
        for i in range(countCls):
            if selection[i]:
                self.cls.append(self.dict[i][0])                
        
        if len(self.cls) == 0:
            self.cls.append(self.dict[1][0])              
            self.clsBool =  [False for _ in range(countCls)]
            self.clsBool[1] = True
        else:
            self.clsBool = selection.copy()
    
    def fit(self, x, y):
        """
        Entrena los clasificadores
        param x: Vector de características. Vector de vectores
        param y: Vector de clasificación, por cada vector de características
        """
        for cls in self.cls:
            cls.fit(x[:100], y[:100])

    def predict(self, x):
        """
        Predice la clase a la que pertenece el elemento x
        param x: Vector de características a clasificar
        return: Clasificación
        """

        f=self.cls[0].predict(x)
        e=self.cls[1].predict(x)
        d=self.cls[2].predict(x)
        c=self.cls[3].predict(x)
        b=self.cls[4].predict(x) 
        a = mode([cls.predict(x) for cls in self.cls])[0][0]
        return mode([cls.predict(x) for cls in self.cls])[0][0]

    def score(self, x, y):
        """
        Computa la exactitud de clasificar
        param x: Vector de características. Vector de vectores
        param y: Vector de clasificación, por cada vector de características
        return: Valor numérico entre 0 y 1
        """
        ok = 0
        for i in range(len(x)):
            if self.predict(x[i]) == y[i]:
                ok+=1
        return float(ok) / len(y) 
    
    def save(self):
        """
        Salva las funciones de clasificación
        """
        for i in range(len(self.clsBool)):
            if self.clsBool[i]:
                joblib.dump(self.cls[i], self.dict[i][1] + '.joblib')
    
    def load(self):
        """
        Carga las funciones de clasificación
        """       
        for elem in self.dict.values():
            file = getcwd() + '\\' + elem[1] + '.joblib'
            if exists(file):
                self.clsBool = []
                self.cls = []
                self.clsBool.append(True)
                self.cls.append(joblib.load(file))
                return True
            else:
                return False



def plot_validation(cls, x, y, size=0.8):
    """
    Muestra resultados gráficos del clasificador 
    param cls: Instancia de un clasificador
    param x: Vector de características. Vector de vectores
    param y: Vector de clasificación, por cada vector de características
    """
    x_coord = []
    mTrain = []
    mTest = []
    for i in range(1, len(y), 10):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8)
        cls.fit(xtrain, ytrain)
        
        x_coord.append(i)
        mTrain.append(cls.score(xtrain, ytrain))
        mTest.append(cls.score(xtest, ytest))
    
    plt.plot(x_coord, mTrain, label="Training")
    plt.plot(x_coord, mTest, label="Validation")
    plt.ylim([0, 1.2])
    plt.legend()
    plt.show()




Collective_Classifier = Collective_Classifier()
