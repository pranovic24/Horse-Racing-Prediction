import ast
import os
import numpy as np
import pandas as pd
from functools import cmp_to_key

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.decomposition import PCA


database = r'..\..\data'
features = [0, 1, 2, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 36]

# =============================================================================
# helpers
def convertMillis(millis):
    seconds= (millis/1000)%60 
    minutes=int( (millis/(1000*60))%60 )
    return str(minutes) + ":" + str(seconds)[:6]

# =============================================================================
# data
with open(os.path.join(database, r'race_data_processed.list'), 'r') as f:
    rowsList = ast.literal_eval(f.read())

# Adding the race lengths to the data
race_lengths = pd.read_csv( os.path.join(database, r'race_data_raw.csv') )["Race Length"]
rowsList[0].append("Race Length")
for i,r in enumerate(rowsList[1:]):
    r.append(race_lengths[i])

# Deleting Rows where we don't have a finish time
for i, row in enumerate(rowsList):
    if row[25] == '':
        rowsList.pop(i)
  
print('-' * 50)    
print("Used features:")
header = [rowsList[0][i] for i in features]
print(header)    
print('-' * 50)

X = np.zeros( (len(rowsList) - 1, len(header)) ) # 1 line = 1 horse
y = np.zeros( (len(rowsList) - 1, ) ) # 1 line = finish time
plc = np.zeros( len(rowsList) - 1, )
win_odds = np.zeros( len(rowsList) - 1, )

for j, list in enumerate(rowsList):
    if j == 0:
        continue
    
    # read out the horse rankings
    try:
        plc[j-1] = rowsList[j][23]
        win_odds[j-1] = rowsList[j][26]
    except:
        plc[j-1] = 1000
        win_odds[j-1] = -1
    
    ## create X
    # take desired features
    filtered = [rowsList[j][i] for i in features]
    # convert overweight feature to int
    filtered[4] = 0 if (filtered[4] == '') else int(filtered[4])
    # make sure no str feature left (replace all strings by 0)
    filtered = [0 if (type(x) is str) else x for x in filtered ]
    
    X[j-1, :] = np.array(filtered)
    
    ## Create y
    y[j-1] = rowsList[j][25]

    
# standardize
X = preprocessing.scale(X, axis = 0)

# split into train & test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
TEST_SIZE = 3087
X_test = X[:TEST_SIZE, :]
X_train = X[TEST_SIZE:, :]
y_test = y[:TEST_SIZE]
y_train = y[TEST_SIZE:]
plc_test = plc[:TEST_SIZE]
plc_train = plc[TEST_SIZE:]

# 1 element = 1 race list, in which 1 element = 1 tuple: (input_vec, plc)
race_list_test = []
i = 0
while i <= TEST_SIZE:
    race = []
    race.append( (X[i,:], plc[i], win_odds[i]) )
    d = X[i,0]
    v = X[i,1]
    r = X[i,2]
    while ( d == X[i+1,0] and v == X[i+1,1] and r == X[i+1,2] ):
        i += 1
        race.append( (X[i,:], plc[i], win_odds[i]) )
    race_list_test.append(race)
    i+=1
    
# =============================================================================
    
#pca = PCA(n_components=2, svd_solver='full')
#pca.fit(X_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)
#
#import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [10, 5]
#x = X_train[:, 0]
#y = X_train[:, 1] 
#plt.scatter(x, y, marker='x', color='red')
#plt.show()
    
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)
i = 3
print("Example Prediction of sample ", i, " of the test set:")
print("Prediction: ", convertMillis(y_pred[i]), " Actual Finish Time: ", convertMillis(y_test[i]))

# sorting based on finish times
def cmp_FT(tup1, tup2):
    if (tup1[3] < tup2[3]):
        return -1
    elif (tup1[3] > tup2[3]):
        return 1
    else:
        return 0

# sorting based on win-odds
def cmp_WO(tup1, tup2):
    if (tup1[2] < tup2[2]):
        return -1
    elif (tup1[2] > tup2[2]):
        return 1
    else:
        return 0

# for each horracese, predict the winner & the 3 placers 
for idx, race in enumerate(race_list_test):
    if len(race) < 7: # failed to download whole race
        race_list_test.pop(idx)
num_win_poss = len(race_list_test)
num_place_poss = 3 * len(race_list_test)
num_win_corr = 0
num_place_corr = 0
earned_by_win = - len(race_list_test) * 10   # how much money we won, if betting 10 dollars on every predicted WIN (not including place)
for race in race_list_test:
    for i, tup in enumerate(race):
        x = tup[0]
        predicted_time = clf.predict( x.reshape(1,x.shape[0]) )[0]
#        x = pca.transform(x.reshape(1,x.shape[0])) 
#        predicted_time = clf.predict( x  )[0]
        race[i] = (tup[0], tup[1], tup[2], predicted_time) # (feature_vec, gt_plac, win_odds, predicted finish time)
    race = sorted(race, key=cmp_to_key(cmp_FT))
    # extracting the ground truth place of who we predicted to go 1st, 2nd, 3rd
    first = race[0][1]
    second = race[1][1]
    third = race[2][1]
    if (first == 1):
        num_win_corr += 1
        earned_by_win += 10 * race[0][2]
#        print(race[0][2])
    if (first == 1 or first == 2 or first == 3):
        num_place_corr += 1
    if (second == 1 or second == 2 or second == 3):
        num_place_corr += 1
    if (third == 1 or third == 2 or third == 3):
        num_place_corr += 1
        
print("Win Accuracy = " , num_win_corr / num_win_poss * 100 , "%")
print("Money Earned by betting on 10 HKD on every WIN = " , earned_by_win, "HKD")
print("Place Accuracy = " , num_place_corr / num_place_poss * 100 , "%")       
    
    
