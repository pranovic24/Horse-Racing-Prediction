# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:38:04 2019

@author: psood
"""

import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.metrics import mean_absolute_error
from keras.utils import plot_model
import keras


with open("C:/Users/psood/Desktop/Deep Learning Project/preprocessed_dataset/raw_data.list", 'r') as f:
    rowsList = ast.literal_eval(f.read())

race_lengths = pd.read_csv("C:/Users/psood/Desktop/Deep Learning Project/race_data_raw_11_6_2019.csv")["Race Length"]
rowsList[0].append("Race Length")
for i,r in enumerate(rowsList[1:]):
    r.append(race_lengths[i])
    
raw_data = pd.DataFrame(rowsList)

raw_data.to_csv('C:/Users/psood/Desktop/Deep Learning Project/raw_data.csv', index=False)
 
features = [5,0,1,2,6,8,9,11,12,13,14,15,16,17,18,19,20,21,22,36,25]

rowsList = [row for row in rowsList if row[25] != '']

print('-' * 50)    
print("Used features:")
header = [rowsList[0][i] for i in features]
print(header) 
print('-' * 50)  

rowsList = rowsList[1:]

# Creating the new CSV File
horse_array = np.zeros((len(rowsList),len(header)))
#plc = np.zeros( len(rowsList), )
#win_odds = np.zeros( len(rowsList), )

for i,row in enumerate(rowsList):
    
    new_data = [rowsList[i][j] for j in features]
    
    new_data[5] = 0 if (new_data[5] == '') else int(new_data[5])
    
    new_data = [0 if (type(x) is str) else x for x in new_data]
    
    horse_array[i,:] = np.array(new_data)

df = pd.DataFrame(horse_array)

df.columns = header

df.to_csv('C:/Users/psood/Desktop/Deep Learning Project/new_data.csv', index = False)

# Remove horses that have only done one race
min_race = 1
df = df[df.groupby(['Horse'])['Horse'].transform('size') > min_race]
df.to_csv('C:/Users/psood/Desktop/Deep Learning Project/new_data1.csv', index = False)

# Preprocessing of train and test data
test_size = 3014  #3087
test_data= df.iloc[:test_size, :]
train_data = df.iloc[test_size:, :]

# Sorting the horse number and date of the race
train_data = train_data.sort_values(['Horse','Date'], ascending=[True,True])
test_data = test_data.sort_values(['Horse','Date'], ascending=[True,True])

# Convert everything to float
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')


train_data.to_csv('C:/Users/psood/Desktop/Deep Learning Project/train_data.csv', index = False)
test_data.to_csv('C:/Users/psood/Desktop/Deep Learning Project/test_data.csv', index = False)


# =============================================================================
# Convert series to supervised learning
def supervised_series(data, n_in = 1, n_out = 1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        
    # combining everything together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg    

# =============================================================================

# load dataset
train_dataset = pd.read_csv('C:/Users/psood/Desktop/Deep Learning Project/train_data.csv')
test_dataset = pd.read_csv('C:/Users/psood/Desktop/Deep Learning Project/test_data.csv')

# frame as supervised learning
train_reframed = supervised_series(train_dataset, 1, 1)
test_reframed = supervised_series(test_dataset, 1, 1)


# removal of horse numbers that do not match 
train_reframed = train_reframed[train_reframed['var1(t-1)'] == train_reframed['var1(t)']]
test_reframed = test_reframed[test_reframed['var1(t-1)'] == test_reframed['var1(t)']]

train_reframed.to_csv('C:/Users/psood/Desktop/Deep Learning Project/train_reframed.csv', index=False)
test_reframed.to_csv('C:/Users/psood/Desktop/Deep Learning Project/test_reframed.csv', index=False)


train_horseDate = train_reframed[['var1(t)','var2(t)','var4(t)']]
#print(train_horseDate)
test_horseDate = test_reframed[['var1(t)','var2(t)', 'var4(t)']]

new_test = pd.DataFrame(test_horseDate.values)

# Dropping columns  
train_reframed.drop(train_reframed.columns[[0,1,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
                               35,36,37,38,39,40]],axis=1, inplace=True)

test_reframed.drop(test_reframed.columns[[0,1,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
                               35,36,37,38,39,40]],axis=1, inplace=True)

train_values = train_reframed.values
test_values = test_reframed.values
   
# normalize features 
#scaler = StandardScaler()

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.fit_transform(test_values)


# split into input and outputs
train_X, train_y = train_scaled[:, :-1], train_scaled[:,-1]

test_X, test_y = test_scaled[:, :-1], test_scaled[:,-1]

# reshape input to be 3D [samples, timesteps, features]
timestep = 1

train_X = train_X.reshape((train_X.shape[0], timestep, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], timestep, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



# design network
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999,amsgrad=False)
model.compile(loss='mae', optimizer='adam', metrics = ['mae'])

# checkpoint
filepath="C:/Users/psood/Desktop/Deep Learning Project/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, 
                             save_weights_only=False, mode='auto', period=10)


# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_split=0.1, verbose=1, shuffle=False,
                    callbacks=[checkpoint])

# model summary
model.summary()


model.save('LSTM_64_nodes_MinMax(50).h5')

# plot history
#plot_model(model, to_file='model.png')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()
pyplot.show()

#model = keras.models.load_model('LSTM_128_nodes.h5')

# Testing and Evaluation

model = keras.models.load_model('LSTM_64_nodes_MinMax(50).h5')

# Make a prediction
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, :], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

#print(inv_yhat)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))


inv_y = np.concatenate((test_X[:, :], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]


# calculate MAE
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)

header2 = ['Horse','Date', 'Race No.', 'Finish Time']

inv_y = pd.DataFrame(inv_y)
inv_yhat = pd.DataFrame(inv_yhat)

actual_test = pd.concat([new_test,inv_y], axis=1)
actual_test.columns = header2

predicted_test = pd.concat([new_test,inv_yhat], axis=1)
predicted_test.columns = header2

actual_test = actual_test.sort_values(['Date','Race No.','Finish Time'], ascending=[False,True,True])
predicted_test = predicted_test.sort_values(['Date','Race No.','Finish Time'], ascending=[False,True,True])

actual_test.to_csv('C:/Users/psood/Desktop/Deep Learning Project/actual_test.csv', index=False)
predicted_test.to_csv('C:/Users/psood/Desktop/Deep Learning Project/predicted_test.csv', index=False)

# Winner Accuracy Evaluation

actual_win = actual_test.drop_duplicates(subset=['Date', 'Race No.'], keep='first')

predicted_win = predicted_test.drop_duplicates(subset=['Date', 'Race No.'], keep='first')

actual_winList = list(actual_win['Horse'])
predicted_winList = list(predicted_win['Horse'])

num_win_poss = len(actual_winList)
num_win_corr = 0

for i in range(num_win_poss):
    if actual_winList[i] == predicted_winList[i]:
        num_win_corr += 1
      
winner_corr = (num_win_corr/num_win_poss)*100
print("Win Accuracy = " , winner_corr)

# Place Accuracy Evaluation

df1 = pd.read_csv('C:/Users/psood/Desktop/Deep Learning Project/actual_test.csv')
df2 = pd.read_csv('C:/Users/psood/Desktop/Deep Learning Project/predicted_test.csv')

def extract_info(df0):
    dataframes=[]
    llist = list(set(df0['Date']))
    for i in range(len(llist)):
        dff = df0[df0['Date']==llist[i]]
        lllist = list(set(dff['Race No.']))
        for race in lllist:
    #         print(race)
            dfff = dff[dff['Race No.']==race]
            dataframes.append(dfff.head(3))
    return dataframes

a = extract_info(df1)[8]['Horse']

b = extract_info(df2)[8]['Horse']

num_corr_place = 0

for i in range(len(extract_info(df1))):
    
    actual = extract_info(df1)[i]['Horse']
    predicted = extract_info(df2)[i]['Horse'] 
    count = 0
    for j in actual: 
        for k in predicted:  
            if j == k:
                count += 1
    
    num_corr_place += count

num_pos_pla = 3*num_win_poss
place_corr = (num_corr_place/num_pos_pla)*100
print("Place Accuracy = " , winner_corr)
