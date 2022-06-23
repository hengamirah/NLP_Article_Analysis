# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:39:18 2022

@author: Amirah Heng
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Masking
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout, Bidirectional, Embedding

#%% CLASSES & FUNCTIONS


class Model_Creation():       
    def __init__(self):
        pass
    
    def NLP_model(self, nb_features, output_node, vocab_size, embedding_dim = 64, 
                  drop_rate=0.2, num_node=64):
        
        # Create NLP model
        model = Sequential()
        model.add(Input(shape=nb_features))
        model.add(Masking(mask_value=0)) #Masking Layer - Remove the 0 from padded data 
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=True))) 
        model.add(LSTM(num_node,return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node)) #Going to lower dense layer no need return_sequences
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node, activation ='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node, activation ='softmax'))
        model.summary()
        return model



class Model_Analysis():       
    def __init__(self):
        pass
    
    def plot_analysis(self,hist):
        '''
        Generate graphs to evaluate model loss and accuracy 

        Parameters
        ----------
        hist : TYPE
            model fitted with Training and Validation(test) dataset.

        Returns
        -------
        None.

        '''
        hist_keys = [i for i in hist.history.keys()]

        plt.figure()
        plt.plot(hist.history[hist_keys[0]],'r--') #loss
        plt.plot(hist.history[hist_keys[2]]) #val loss
        plt.legend(['training_loss','validation_loss'])
        plt.show()

        plt.figure()
        plt.plot(hist.history[hist_keys[1]],'r--')
        plt.plot(hist.history[hist_keys[3]])
        plt.legend(['training_acc','validation_acc'])
        

    def Model_Evaluation(self, model,X_test,y_test):
        '''
        Generates confusion matrix and classification report based
        on predictions made by model using test dataset.

        Parameters
        ----------
        model : model
            Prediction model.
        x_test : ndarray
            Columns of test features.
        y_test : ndarray
            Target column of test dataset.
        label : list
            Confusion matrix labels.

        Returns
        -------
        Returns numeric report of model.evaluate(), 
        classification report and confusion matrix.

        # '''
        # result = model.evaluate(X_test,y_test)
        
        # print(result) # loss, metrics
        y_pred=np.argmax(model.predict(X_test),axis=1)
        y_true=np.argmax(y_test,axis=1)
        
        #Confusion_matrix
        cm=confusion_matrix(y_true,y_pred)
        print('Confussion matrix: \n ',cm)
        
        #show Confusion Matrix plot graph
        label = ['tech','business','sport','entertainment','politics']
        disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        disp.plot(cmap=plt.cm.Reds)
        plt.show()
                
        #Classification report
        cr=classification_report(y_true, y_pred)
       
        print('Classification Report: \n ',cr)
       
        #Accuracy score
        print('\n Accuracy score: \n ', accuracy_score(y_true,y_pred))
        
       


