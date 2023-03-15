#package imports
import re

import cufflinks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns


from nltk.corpus import stopwords
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import BertConfig, BertTokenizerFast, TFBertModel


#read in the data
data = pd.read_csv("NDA.csv")
NDA = data.copy()
NDA = NDA.dropna(how = 'all')

#draw the graph indicating the original classification method
NDA['Category'].value_counts().plot(kind = "bar")
plt.show()


'''a function which re-catogerize the NDA clauses in order to improve model accuracy and make the 
classification easier
Originally there are 17 categories, and now we are compressing it to 7'''

def reclassify(row):
    if row['Category'] == 1 or row['Category'] == 3:
        return 'DEF'
    elif row['Category'] == 2:
        return 'RIG'
    elif row['Category'] == 4 or row['Category'] == 15:
        return 'EXP'
    elif row['Category'] == 5 or row['Category'] == 6:
        return 'WAR'
    elif row['Category'] == 7 or row['Category'] == 10:
        return 'GOV'
    elif row['Category'] == 8:
        return 'REM'
    elif row['Category'] == 13 or row['Category'] == 16:
        return 'TER'
    else:
        return 'GEN'

#apply the new classification method
NDA['CAT'] = NDA.apply(lambda row: reclassify(row), axis = 1)

#the value counts
NDA['CAT'].value_counts()
#draw the graph indicating the new classification method
NDA['CAT'].value_counts().plot(kind = "bar")
plt.show()

#data cleaning process 
NDA = NDA.reset_index(drop=True)
REPLACE_BY_SPACE = re.compile('[/(){}\[\]\|@,;\n\r*]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(para):

    para = str(para)
    para = para.lower() #turn all the text into lowercase
    para = REPLACE_BY_SPACE.sub(' ', para) 
    para = BAD_SYMBOLS.sub('', para) #remove the meaningless symbols
    para = ' '.join(word for word in para.split() if word not in STOPWORDS) #remove stopwords
    para = para.lstrip('0123456789.- ') #remove the digits/space/dots at the start of the string

    return para


NDA['cleaned_txt'] = NDA['Text'].apply(clean_text) #the cleaned data
print(NDA.head())
    

