import pandas as pd

from sklearn import svm, grid_search, tree
from sklearn import preprocessing
import numpy as np
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from sklearn.externals import joblib

def read(file, sep):
    df = pd.read_csv(file,sep=sep, header=0, na_values=["nan", "NA", "NaN"], keep_default_na = False)
    print "data read: ",format(len(df)),",",len(df.columns)
    return df


def filter_data(data,remove_classes):
    #Filtrowanie kolumn
    filtered = data[~data["res_name"].isin(remove_classes)]

    #Unikalnosc pdb_code , res_name
    unique = filtered.drop_duplicates(subset=["pdb_code", "res_name"], keep='first')

    #Ograniczenie do 5 powtorzen
    counts = unique[['res_name']].stack().value_counts()
    values = counts[counts>=5].index
    min5Occurs = unique.loc[unique['res_name'].isin(values)]

    print "original: ",format(len(data)),",",len(data.columns)
    print "filtered: ",format(len(filtered)),",",len(filtered.columns)
    print "unique: ",format(len(unique)),",",len(unique.columns)
    print "5+ occurs: ",format(len(min5Occurs)),",",len(min5Occurs.columns)

    return min5Occurs


def remove_na_columns(df):
    no_na_columns = df.count()==len(df)
    no_na_columns = no_na_columns.values
    return df.iloc[:,no_na_columns]

##def getCommonColumns(data,test):
##    return data[data.columns.isin(test.columns)]


def learn(data,test_data):
    #zamiana reprezentacji tekstowej na binarna
    encoder = preprocessing.LabelEncoder()
    encoder.fit(data[['res_name']])
    data[['res_name']] = encoder.transform(data[['res_name']])

    #usuniecie kolumn ktore zawieraja NA
    data = remove_na_columns(data)
    test_data = remove_na_columns(test_data)

    ##zmienna tymczasowa, potrzebna pozniej przy klasyfikacji
    original = data

    ##selekcja kolumn ktore sa typu float (
    float_idx = data.dtypes == np.float64
    data = data.loc[:,float_idx]

    test_float_idx = test_data.dtypes == np.float64
    test_data = test_data.loc[:,test_float_idx]

    ##selekcja kolumn ktore sa wspolne dla test_data i data
    common_columns = [col for col in test_data if col in data]
    test_data = test_data[common_columns]
    data = data[common_columns]

    ##klasyfikacja
    df, classes = data.loc[:,float_idx], original[['res_name']]

    my_tree = tree.DecisionTreeClassifier(max_features = "auto",max_depth = None)
    #method = AdaBoostClassifier(base_estimator = my_tree)
    method = BaggingClassifier(base_estimator = my_tree)

    params = {
        "base_estimator__criterion" : ["gini", "entropy"],
        "base_estimator__splitter" :   ["best", "random"],
        "n_estimators": [1, 5]
     }

    classificator = grid_search.GridSearchCV(method, param_grid=params, scoring = 'recall')
    classificator.fit(df, np.asarray(classes).ravel())

    joblib.dump(classificator, "klasyfikator.pk")

    print 'grid_scores:', classificator.grid_scores_
    print 'best_estimator:', classificator.best_estimator_
    print 'best_score:', classificator.best_score_
    print 'best_params:', classificator.best_params_