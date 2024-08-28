import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score , make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
import warnings 
warnings.filterwarnings("ignore")

class ML_System_Regression():
    def __init__(self):
        pass

    def load_data(self): 
        path = "C:/Users/PC/Downloads/"
        dataset = pd.read_csv(path + "iris_dataset.csv",header = 0,sep=";",decimal=",") 
        prueba = pd.read_csv(path + "iris_prueba.csv",header = 0,sep=";",decimal=",")

        covariables = [x for x in dataset.columns if x not in ["y"] ]
        X= dataset.get(covariables)
        y = dataset["y"]

        X_nuevo = prueba.get(covariables)
        y_nuevo = prueba["y"]
    
        return X, y, X_nuevo, y_nuevo 

    def preprocessing_Z(self, X):
        Z= preprocessing.StandardScaler()
        Z.fit(X)
        X_z= Z.transform(X)
        return Z, X_z
    
    def training_model(self, X, y,X_nuevo,y_nuevo, grilla_completo):
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.5)
        #MODELO 1
        Z_1,X_train_Z= self.preprocessing_Z(X_train)
        X_test_Z= Z_1.transform(X_test)

        modelo1 = LogisticRegression(random_state=123)#C parametro de regularizacion el modelo y se optiene el beta con ese parametro, valor optimo de c para tener el mejor acuracy
        parametros ={'C': np.arange(0.1,5.1,0.1)}
        grilla1 = GridSearchCV(estimator=modelo1,param_grid=parametros,cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
        grilla1.fit(X_train_Z, y_train)
        y_hat_test= grilla1.predict(X_test_Z)
        
        #MODELO 2
        Z_2,X_test_Z= self.preprocessing_Z(X_test)
        X_train_Z= Z_2.transform(X_train)

        modelo2 = LogisticRegression(random_state=123)#C parametro de regularizacion el modelo y se optiene el beta con ese parametro, valor optimo de c para tener el mejor acuracy
        grilla2 = GridSearchCV(estimator=modelo2,param_grid=parametros,cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
        grilla2.fit(X_test_Z, y_test)
        y_hat_train= grilla2.predict(X_train_Z)
        
        u1 = accuracy_score(y_test, y_hat_test)
        u2 = accuracy_score(y_train, y_hat_train)
        
        if np.abs(u1-u2)< 10: #modelo no esta sobrentrenado
            modelo_completo= LogisticRegression(random_state=123)
            grilla_completo = GridSearchCV(estimator=modelo_completo,param_grid=parametros,cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
            Z= preprocessing.StandardScaler()
            Z.fit(X)
            X_z= Z.transform(X)
            grilla_completo.fit(X_z,y)
        else: #sino cumple, no optimice 
            grilla_completo = LogisticRegression(random_state=123)
            Z= preprocessing.StandardScaler()
            Z.fit(X)
            X_z= Z.transform(X)
            grilla_completo.fit(X_z,y)
        #Predicciones 
        X_nuevo_Z= Z.transform(X_nuevo)
        y_hat_nuevo = grilla_completo.predict(X_nuevo_Z)
        #Evaluación modelo
        return accuracy_score(y_nuevo,y_hat_nuevo)

    def ML_Flow_regression(self):
        try: #
            X, y, X_nuevo, y_nuevo  = self.load_data()
            metric = self.training_model(X,y, X_nuevo, y_nuevo)
            return {'success':True, 'accuracy':metric }
        except Exception as e:
            return {'success':False, 'message':str(e)}