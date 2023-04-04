#package imports
import re
import tensorflow


# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.figure_factory as ff
# import seaborn as sns


from nltk.corpus import stopwords
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
# plt.show()


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
# plt.show()

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


#make the categories of the clauses into numerical category and split the train/test
NDA = NDA[['CAT', 'cleaned_txt']]
NDA = NDA[NDA['CAT'] != "GEN"] #remove the general category
NDA = NDA.reset_index(drop=True)

NDA['CAT_label'] = pd.Categorical(NDA['CAT'])
NDA['CAT'] = NDA['CAT_label'].cat.codes
NDA, NDA_test = train_test_split(NDA, test_size = 0.2, stratify = NDA[['CAT']])



############################set up bert model############################
# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 100

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)


############################Build the model############################

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
CAT = Dense(units=len(NDA.CAT_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='CAT')(pooled_output)
outputs = {'CAT': CAT}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()


############################Train the model############################

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'CAT': CategoricalCrossentropy(from_logits = True)}
metric = {'CAT': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
y_CAT = to_categorical(NDA['CAT'])


# Tokenize the input (takes some time)
x = tokenizer(
    text=NDA['cleaned_txt'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'CAT': y_CAT},
    validation_split=0.2,
    batch_size=64,
    epochs=1)
    

### ----- Evaluate the model ------ ###

# Ready test data
test_y_CAT = to_categorical(NDA_test['CAT'])

test_x = tokenizer(
    text=NDA_test['cleaned_txt'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)


#evaluation in more detail
def convert_prediction(matrix):
    row = matrix.shape[0]
    prediction = np.zeros(row)
    for i in range(row):
        prediction[i] = np.argmax(matrix[i])
    return prediction



target_names = ['DEF', 'EXP', 'GOV', 'REM', 'RIG', 'TER', 'WAR']
result = model.predict(test_x['input_ids'])
print(classification_report(NDA_test['CAT'], convert_prediction(result['CAT']), target_names= target_names))


model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tensorflow.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()


