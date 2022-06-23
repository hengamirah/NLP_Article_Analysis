# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:21:39 2022

@author: Amirah Heng
"""

import os
import re
import pickle
import json
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from module_for_article_analysis import Model_Analysis, Model_Creation

#%%Staics

CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'model', 'tokenizer_article.json')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)

#EDA
#%% STEP 1 - Data Loading

df = pd.read_csv(CSV_URL) 

#%% STEP 2 - Data Inspection

#view the data content
df.head(10)
df.tail(10) #2225 dataset

sns.countplot(df.category) #The bar plot for category

#to get the unique target variables
df['category'].unique()  #There are 5 category in this :['tech', 'business', 'sport', 'entertainment', 'politics']

df['category'][5]# politics category
df['text'][5]#example text of politics category

df.isna().sum() # no missing values

df.duplicated().sum() #There are 99 duplicated datas 
df[df.duplicated()] #Extracting the duplicated data 

#%% STEP 3 - Data Cleaning
# 1) Remove all the duplicated data
df= df.drop_duplicates() #Remove 99duplicates, left 2126 datas
df.duplicated().sum() # no more duplicate values

category =  df['category'].values #Extract Features with 5 categories
text=df['text'].values #Target y

# 2) Remove numerics inside text file
for index, t in enumerate(text):
    text[index] = re.sub('.*?',' ',t)
    text[index] =re.sub('[^a-zA-Z]',' ',t).lower().split() # ^ means NOT alphabet
# substituting that is not a-z and A-Z  will be replaced with a space
# Hence, all numeric will be removedso now we have changed every word into 
# lower case and splitted them into a list of words

text[10] #All the words has been split into a list of words with lower case

#but now the text has more rows than category, because text is linked to df

#%% STEP 4 - Features Selection
# Nothing to select since this NLP data

#%% STEP 5 - Preprocessing
# 1) Tokenization

vocab_size = 400
OOV_token = 'OOV'

tokenizer= Tokenizer(num_words=vocab_size, oov_token=OOV_token)
tokenizer.fit_on_texts(text) #to learn all the words
word_index = tokenizer.word_index
# print(word_index) #each word has now an index

# so need to encode all this into numbers to fit the text
train_sequences = tokenizer.texts_to_sequences(text) #to convert into numbers

#Get the average number of text inside a row
#Test to see the number of words from one category of text
# len(train_sequences[0]) #744
# len(train_sequences[1]) #296
# len(train_sequences[500])#600
# len(train_sequences[1500]) #306

length_of_text=[len(i) for i in train_sequences]
np.median(length_of_text) #334 median value of number of text
np.mean(length_of_text) #387 mean value of number of text
np.max(length_of_text) #4469 max value of text

# Pick the reasonable padding value
# Median is chosen for this padding values
# Padding is to make each length to be similar

# 2) Padding and Truncating
max_len=380
padded_text = pad_sequences(train_sequences, 
                            maxlen=max_len,
                            padding='post',
                            truncating = 'post')  # so now all in equal length 


# 4) One Hot Encoding for the Target - category
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

# 5) Train-test-split because this is a classification problem
X_train,X_test,y_train,y_test = train_test_split(padded_text,
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)
X_train= np.expand_dims(X_train, axis=-1)
X_test= np.expand_dims(X_test, axis=-1)


#%% STEP 6 - Model Development
# USE LSTM layers, dropout, dense, input
# acheive > 90% F1 score

nb_features = 380
output_node=len(y_train[1]) 
model = Model_Creation().NLP_model(nb_features, output_node,vocab_size, 
                                   embedding_dim = 128, drop_rate=0.2, 
                                   num_node= 128 )

#Callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_FOLDER_PATH)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics ='acc') #since its a classifier problem, categorical crossentropy is chosen

#%% STEP 7) MODEL ANALYSIS

# Visualising Natural Language Processing model
plot_model(model,show_shapes=True, show_layer_names=(True))

# Model Fitting and Testing(
hist= model.fit(X_train, y_train, batch_size=20, epochs=80, 
                validation_data=(X_test, y_test),
                callbacks= tensorboard_callback)

#0.27% 1st train
#afterbidirectional still not improve #0.29

#Plot hist evaluation
Model_Analysis().plot_analysis(hist)   

#%% STEP 7 - Model Evaluation

Model_Analysis().Model_Evaluation(model,X_test,y_test)

#%% Step 8) Model Saving
#save NLP model
model.save(MODEL_SAVE_PATH)

#save tokenizer
token_json = tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file) # token_json is our dictionary now

#save One Hot Encoding model
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

#%% Discussion)

# The model achieved high  with average 83% F1-score and accuracy score of 83%
# Model evaluated with test data has 83% accuracy 
# when put earlystopping the model reduces accuracy to 29% only
# To further increase the performance of NLP model: 
    # 1) Increasing number of epochs, 
    # 2) Increase number of samples
    # 3) Change dropout rate value
    # 4) Add word2vec to remove stop words from dataset







