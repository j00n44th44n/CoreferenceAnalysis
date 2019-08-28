
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.svm import SVC

# from sklearn.cross_validation import train_test_split
# from sklearn.externals import joblib

# from scipy.stats import mode
# from os.path import exists 
# from os import getcwd

# import dataset

# import nltk 

# import matplotlib.pyplot as plt


# # coded by jonathan & carlos C511
# import sklearn
# from sklearn.naive_bayes import BernoulliNB
# import random
# import nltk
# import os
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import names
# from sklearn.externals import joblib
# from os.path import exists




from text_processing import get_markables
import classifiers

def myrun(text):
    a = cls.Collective_Classifier()
    data,pair_noun_phrases = get_markables(text)

    # for i in data:
    #     print(i, '\n')
    #     if len(i) != 12:
    #         print('error', len(i))            
    #         return
    
    output = ''
    for features,pair_corefers in zip(data,pair_noun_phrases):
        print(a.predict(features))
        if a.predict(features) == True:
            output += (pair_corefers + '\n')
    return output

myrun('Qt Designer allows you to design your GUIs visually and save them')