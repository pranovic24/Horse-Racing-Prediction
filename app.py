

# ***************************
date = '20191023'
venue = 'Happy Valley'
race_nr = '6'
# ***************************


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import datetime
import csv
import ast
import os
import re
import numpy as np
import pandas as pd
from functools import cmp_to_key
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import datetime, time
from datetime import datetime
import json
from sklearn.externals import joblib

# =============================================================================
# helpers 
def checkXpath(driver, xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except:
        return False
    return True 

def getTextByXpath(driver, xpath):
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return driver.find_element_by_xpath(xpath).text.split(' ')[1];
    except:
        return ''
    
def findMatch(horseNo, resultsList):
    for h in resultsList:
        if (horseNo == h[1]):
            h.pop(1) 
            return h
    
def prettyDate(date):
    return date[6:8] + '/' + date[4:6] + '/' + date[0:4]

def resultDate(date):
    return date[0:4] + '/' + date[4:6] + '/' + date[6:8]

def getDates(numdays):
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
    date_list = [str(d.year) + "{:02d}".format(d.month) + "{:02d}".format(d.day) for d in date_list]
    return date_list

def validDate(date, driver):
    
    print("Checking " + date)
    
    driver.get('https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx?RaceDate=%s&Racecourse=%s&RaceNo=%s' % (resultDate(date), 'HV', 1))
    if len(driver.find_elements_by_id('errorContainer')) > 0:
        driver.get('https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx?RaceDate=%s&Racecourse=%s&RaceNo=%s' % (resultDate(date), 'ST', 1))
        if len(driver.find_elements_by_id('errorContainer')) > 0:
            return False
        
    try:    
        driver.get('https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/%s/HV/%s' % (date, 1))
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, xpath)))
    except:
        try:
            driver.get('https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/%s/ST/%s' % (date, 1))
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, xpath)))
        except:
            return False
    
    return True
 
def closePopupWindow(driver):
    window_before = driver.window_handles[0]
    window_after = driver.window_handles[1]
    driver.switch_to.window(window_after)
    driver.close()
    driver.switch_to.window(window_before)
    
def convert_datestr_daysno(dateStr):
	try:
		t = datetime.strptime(dateStr, '%d/%m/%Y')
	except:
		print("OOOPSSS: data format is wrong for date: <%s>" % dateStr)
		return ""

	dayNo = int(round(time.mktime(t.timetuple())/(24*3600)))
	return dayNo 
 
# sorting based on finish times
def cmp_FT(tup1, tup2):
    if (tup1[1] < tup2[1]):
        return -1
    elif (tup1[1] > tup2[1]):
        return 1
    else:
        return 0
# =============================================================================

# =============================================================================
# collect race card
driverRaceCard = webdriver.Firefox()
xpath = '/html/body/div[2]/div[2]/div[2]/div[8]/table'
xpathLength = '/html/body/div[2]/div[2]/div[2]/div[4]/div[1]/table'
races = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
features = [1, 5, 7, 8, 10, 11, 12, 13, 15, 18, 19, 20]

venueAbbr = 'HV' if (venue == 'Happy Valley') else 'ST'
r = race_nr
allData = []
horseList = []


# Loade Race Cards Page
try:
    if (venue == 'Sha Tin'):
        raise Exception
    # HV: Happy Valley
    driverRaceCard.get('https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/%s/HV/%s' % (date, r))
    WebDriverWait(driverRaceCard, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
    venue = 'Happy Valley'
except:
    try:
        # ST: Sha Tin 
        driverRaceCard.get('https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/%s/ST/%s' % (date, r))
        WebDriverWait(driverRaceCard, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
        venue = 'Sha Tin'
    except:
        print('No Race Card available for: ' + prettyDate(date) + ', race ' + r)

# make all hidden columns visible
driverRaceCard.execute_script("for(var i = 0; i < document.getElementsByTagName('td').length; i++) { document.getElementsByTagName('td')[i].style.display = 'block';}")

print('Collecting Race information for: ' + prettyDate(date) + ', race ' + r + ', (' + venue + ')')

# get race length
regex = re.compile(r'\d\d\d\dM')
try:
    table = driverRaceCard.find_elements_by_xpath(xpathLength)
    for idx, row in enumerate(table[0].find_elements_by_xpath(".//tr")):
        length = int(regex.search(row.text).group()[:4])
except Exception as Fail:
    print(Fail)
    length = -1
    
# Crawl RaceCard table
try:
    table = driverRaceCard.find_elements_by_xpath(xpath)
    for idx, row in enumerate(table[0].find_elements_by_xpath(".//tr")):
        if idx < 2:
            continue
        elif idx == 2:
            header = [td.text for td in row.find_elements_by_xpath(".//td")]
            header = ['Date', 'Venue', 'Race'] + [header[q] for q in features] + ['win% Jockey', 'place% Jockey', 'win% Trainer', 'place% Trainer'] + ['RaceLength']
        else:
            raceData = [td.text for td in row.find_elements_by_xpath(".//td")]
            horseList.append(raceData[3])
            raceData = [raceData[q] for q in features]
            
            if ('Withdrawn' in raceData[2]):
                continue
            
            xpathJockey = '/html/body/div[2]/div[2]/div[2]/div[8]/table/tbody/tr[2]/td/table/tbody/tr[%d]/td[7]/a' % (idx - 2)
            xpathTrainer = '/html/body/div[2]/div[2]/div[2]/div[8]/table/tbody/tr[2]/td/table/tbody/tr[%d]/td[10]/a' % (idx - 2)
            xpathWinRateJockey = '/html/body/div/div[3]/table/tbody/tr[3]/td[2]'
            xpathPlaceRateJockey = '/html/body/div/div[3]/table/tbody/tr[2]/td[4]'
            xpathWinRateTrainer = '/html/body/div/div[3]/table/tbody/tr[2]/td[4]'
            xpathPlaceRateTrainer = '/html/body/div/div[3]/table/tbody/tr[3]/td[4]'
            
            driverRaceCard.find_element_by_xpath(xpathJockey).click()
            window_before = driverRaceCard.window_handles[0]
            window_after = driverRaceCard.window_handles[1]
            driverRaceCard.switch_to.window(window_after)
            winRateJockey = getTextByXpath(driverRaceCard, xpathWinRateJockey)
            placeRateJockey = getTextByXpath(driverRaceCard, xpathPlaceRateJockey)
            driverRaceCard.close()
            driverRaceCard.switch_to.window(window_before)
            
            driverRaceCard.find_element_by_xpath(xpathTrainer).click()
            window_before = driverRaceCard.window_handles[0]
            window_after = driverRaceCard.window_handles[1]
            driverRaceCard.switch_to.window(window_after)
            winRateTrainer = getTextByXpath(driverRaceCard, xpathWinRateTrainer)
            placeRateTrainer = getTextByXpath(driverRaceCard, xpathPlaceRateTrainer)
            driverRaceCard.close()
            driverRaceCard.switch_to.window(window_before)
            
            allData.append([prettyDate(date)] + [venue] + [r] + raceData + [winRateJockey, placeRateJockey, winRateTrainer, placeRateTrainer] + [length])

except Exception as Fail:
    print( Fail )
driverRaceCard.close()

# =============================================================================
# preprocessing
X = np.zeros((len(allData), 25))

# read uniquify dicts
data_base = r'C:\Users\janik\Dropbox\1 - ETHZ\HS 19\3 - Deep Learning\1 Project\datasets\preprocessed_dataset'
with open(os.path.join(data_base, r'Venue_enumeration.dict')) as json_file:
    venueDict = json.load(json_file)
with open(os.path.join(data_base, r'Priority_enumeration.dict')) as json_file:
    prioDict = json.load(json_file)
with open(os.path.join(data_base, r'Gear_enumeration.dict')) as json_file:
    gearDict = json.load(json_file)
    
for i,row in enumerate(allData):
    for j,field in enumerate(row):
        try:
            if (j == 0):                    # date
                X[i,j] = convert_datestr_daysno(field)
            if (j == 1):                    # venue
                X[i,j] = venueDict[field]
            if (j == 2):
                X[i,j] = int(field)         # race id
            if(j == 3):                     #l6r1-6 "3/4/2/5/2/3"
                l6rList = np.array(field.split('/')[::-1])
                X[i, -7:(-7+l6rList.shape[0])] = l6rList
            if (j in [4,5,6,7,8,9,10,11,12]):    # all ints
                try:
                    X[i,j-1] = int(field)
                except:
                    X[i,j-1] = 0
            if (j == 13):                   # priority
                X[i,j-1] = prioDict[field]
            if (j == 14):                   # gear
                X[i,j-1] = gearDict[field]
            if(j in [15, 16, 17, 18]):      # win/place %
                X[i,j-1] = float(field.strip('%'))
            if (j == 19):                   # race length
                X[i, -1] = field
        except Exception as Fail:
            print( Fail )
            
# standardize
scaler = joblib.load(r'C:\Users\janik\Dropbox\1 - ETHZ\HS 19\3 - Deep Learning\1 Project\code\simple_regression\X_scaler.save') 
X = scaler.transform(X)

# =============================================================================
# load model & make predictions
model = keras.models.load_model(r'C:\Users\janik\Dropbox\1 - ETHZ\HS 19\3 - Deep Learning\1 Project\code\simple_regression\5_layers_128_nodes.h5')
finish_times = model.predict(X)
ranking = [(horseList[i], finish_times[i][0]) for i in range(len(horseList))]
ranking = sorted(ranking, key=cmp_to_key(cmp_FT))
print("Predicted Ranking: ")
print('1.: ' , ranking[0])
print('2.: ' , ranking[1])
print('3.: ' , ranking[2])
# =============================================================================
        
        
        
        
        
        
        
        
        
