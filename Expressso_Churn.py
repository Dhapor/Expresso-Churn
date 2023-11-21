import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('Expresso_churn_dataset.csv')
print(data.head())

df = data.copy()
# df.drop('user_id', axis = 1, inplace = True)
# for i in df:
#     if df[i].isnull().sum() / len(df) * 100 > 50:
#         df.drop(i, axis = 1, inplace = True)
# df.isnull().sum()

# sampling0 = df[df.CHURN == 0]
# sampling0 = sampling0.dropna()
# sampling0.reset_index(drop = True, inplace = True)
# sampling0.shape

# sampling1 = df[df.CHURN == 1]
# sampling1.drop(['REGION', 'TOP_PACK', 'FREQ_TOP_PACK'], axis = 1, inplace = True)
# sampling1.dropna(inplace = True)
# sampling1.reset_index(drop = True, inplace =True)
# sampling1.shape

# sampling0 = sampling0.sample(35000)
# cols = sampling1.columns
# df = pd.concat([sampling1, sampling0[cols]], axis = 0)
# df.isnull().sum()

# df = df.copy()
# df.dropna()

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

def outlierRemoval(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtypes != 'O': # --------------------------------------- If the data type is not an object type
            Q1 = dataframe[i].describe()[4] # ---------------------------------- Identify lower Quartile
            Q3 = dataframe[i].describe()[6] # ---------------------------------- Identify the upper quartile
            IQR = Q3 - Q1 # ---------------------------------------------------- Get Inter Quartile Range
            minThreshold = Q1 - 1.5 * IQR # ------------------------------------ Get Minimum Threshold
            maxThreshold = Q3 + 1.5 * IQR # ------------------------------------ Get Maximum Threshold

            dataframe = dataframe.loc[(dataframe[i] >= minThreshold) & (dataframe[i] <= maxThreshold)]
    return dataframe


df = outlierRemoval(df)

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in df.drop('CHURN', axis = 1).columns: # ...................................................... If the selected column is found in the general dataframe
        df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in df.drop('CHURN', axis = 1).columns: # ...................................................... If the selected columns are found in the general dataframe
        df[i] = encoder.fit_transform(df[i])# .................................. encode it

df.head()

sel_cols = ['REGULARITY', 'DATA_VOLUME','REVENUE',  'ORANGE', 'ON_NET', 'MONTANT','FREQUENCE']
x = df[sel_cols]

x = x
y = df.CHURN
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 75, stratify = y)


# # -----------------------MODELLING--------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest)

# save model
model = pickle.dump(model, open('Expresso_churn.pkl', 'wb'))

print(f'\nModel is Saved\n')

#-------Streamlit development------
model = pickle.load(open('Expresso_churn.pkl', "rb"))

st.markdown("<h1 style = 'color: #860A35; text-align: center; font-family:  montserrat'>EXPRESSO CHURN</h1>", unsafe_allow_html=True)
st.markdown("<h6 style = 'margin: -15px; color: #860A35; text-align: center; font-family:montserrat'>Churn Prediction for Expresso Clients</p></h6>", unsafe_allow_html=True)
st.markdown('<br1>', unsafe_allow_html= True)
st.image('bg.jpeg',width = 400) #---- to give it image
st.markdown("<h5 style='color: #ffffff; background-color: #860A35; text-align: center; padding: 5px; font-family: Arial, sans-serif;'>BACKGROUND OF STUDY</h5>", unsafe_allow_html=True)

st.markdown('<br1>', unsafe_allow_html= True)

st.markdown("<h6>Expresso, a well-known telecommunications provider in Africa under the Sudatel Group, connects various nations with vital services like internet and mobile phone service. As one of the major telecom companies in Africa, Expresso is essential to improving connectivity, fostering social interaction, and advancing economic development. This work aims to forecast the likelihood of churn among Expresso customers by utilizing a dataset comprising over 2.5 million users and over 15 behavior elements. To effectively plan customer retention campaigns, handle obstacles, and capitalize on innovation opportunities in this fast-paced sector, telecommunications companies must comprehend and anticipate customer attrition.</h6>", unsafe_allow_html=True)

st.sidebar.image('pngwing.com (8).png')

dx = data[['REGULARITY', 'DATA_VOLUME','REVENUE',  'ORANGE', 'ON_NET', 'MONTANT','FREQUENCE']]
st.write(data.head())

input_type = st.sidebar.radio("Select Your Prefered Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    REGULARITY = st.sidebar.slider("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    DATA_VOLUME = st.sidebar.slider("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    REVENUE = st.sidebar.slider("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ORANGE = st.sidebar.slider("ORANGE", data['ORANGE'].min(), data['ORANGE'].max())
    ON_NET = st.sidebar.slider("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    MONTANT = st.sidebar.slider("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE = st.sidebar.slider("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())
else:
    st.sidebar.header('Input Your Information')
    REGULARITY = st.sidebar.number_input("REGULARITY", data['REGULARITY'].min(), data['REGULARITY'].max())
    DATA_VOLUME = st.sidebar.number_input("DATA_VOLUME", data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
    REVENUE = st.sidebar.number_input("REVENUE", data['REVENUE'].min(), data['REVENUE'].max())
    ORANGE = st.sidebar.slider("ORANGE", data['ORANGE'].min(), data['ORANGE'].max())
    ON_NET = st.sidebar.slider("ON_NET", data['ON_NET'].min(), data['ON_NET'].max())
    MONTANT = st.sidebar.slider("MONTANT", data['MONTANT'].min(), data['MONTANT'].max())
    FREQUENCE = st.sidebar.slider("FREQUENCE", data['FREQUENCE'].min(), data['FREQUENCE'].max())

st.header('Input Values')

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'REGULARITY':REGULARITY, 'DATA_VOLUME': DATA_VOLUME, 'REVENUE': REVENUE, 'ORANGE':ORANGE, 'ON_NET':ON_NET, 'MONTANT': MONTANT, 'FREQUENCE':FREQUENCE}])

st.write(input_variable)

# Standard Scale the Input Variable.
for i in input_variable.columns:
    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h4 style = 'color: #860A35; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)



if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('CHURNERS Predicted')
    st.image('check icon.png', width = 200)
    st.success(f'predicted CHURN with provided information is {predicted}')

st.markdown('<br><br>', unsafe_allow_html= True)


st.markdown("<h8 style = 'color: #860A35; text-align: left; font-family:montserrat'>Expresso Churn built by Datapsalm</h8>",unsafe_allow_html=True)


