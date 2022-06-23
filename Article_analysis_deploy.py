# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:44:10 2022

@author: Amirah Heng
"""

import os
import json
import pickle
import re
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1. Trained Model --> load model
# 2. tokenizer --> loading from json
# 3. MMS/OHE --> loading from pickle


TOKENIZER_PATH = os.path.join(os.getcwd(), 'model','tokenizer_article.json')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe.pkl')

# to load trained model
model = load_model (os.path.join(os.getcwd(),'model','model.h5'))


model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    

# to load OHE
with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)


#%% !

    
input_text = input('type your article here')
input_text = re.sub('<.*?>',' ',input_text)
input_text = re.sub('[^a-zA-Z]',' ',input_text).lower().split()                   
    
    
tokenizer = tokenizer_from_json(loaded_tokenizer)
input_text_encoded = tokenizer.texts_to_sequences(input_text)

input_text_encoded = pad_sequences(np.array(input_text_encoded).T,
                                     maxlen=180,
                                     padding='post',
                                     truncating='pre')

outcome = model.predict(np.expand_dims(input_text_encoded,axis=-1))


print(loaded_ohe.inverse_transform(outcome)) 
                