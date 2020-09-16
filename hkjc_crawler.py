"""
Race Card:
    https://racing.hkjc.com/racing/info/meeting/Racecard/English/Local
Results:
    https://racing.hkjc.com/racing/information/english/racing/LocalResults.aspx
Odds:
    https://bet.hkjc.com/default.aspx?url=/racing/pages/odds_wp.aspx&lang=en&dv=local
"""

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import datetime
import csv

RECOMPUTE_DATES = False # set to True to recompute valid racing dates

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
    
# =============================================================================
# variables
        
races = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
features = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 19, 20]
featuresResults = [0, 1, 9, 10, 11]
# keeps track of how many races have been crawled
raceID = 0
allData = []
driverRaceCard = webdriver.Firefox()
driverResults = webdriver.Firefox()
#driverOdds = webdriver.Firefox()


# xpath of the race-data table
xpath = '/html/body/div[2]/div[2]/div[2]/div[8]/table'
xpathResults = '/html/body/div/div[5]/table'

# =============================================================================
# collect race dates (from HKJC Results page)
if RECOMPUTE_DATES:
    candidates = getDates(1757)
    dates = [d for d in candidates if validDate(d, driverResults)]
    
    # write the valid dates to a file
    with open('valid_dates.txt', 'w') as f:
        for item in dates:
            f.write("%s\n" % item)
else:
    with open('valid_dates.txt') as f:
        dates = f.readlines()
    dates = [d.rstrip() for d in dates]
    
    dates = dates[364:] # REMOVE ALREADY PROCESSED AFTER CRASH
# =============================================================================
# collect data
for j, date in enumerate(dates):
    
    if ( (j != 0) and (j % 5 == 0)):
        driverRaceCard.close()
        driverResults.close()
        driverRaceCard = webdriver.Firefox()
        driverResults = webdriver.Firefox()
    
    # holds data for all races of the day
    venue = ''
    for r in races:
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
                continue
        
        print('Collecting Race information for: ' + prettyDate(date) + ', race ' + r + ', (' + venue + ')')
        
        # Load Results Page
        try:
            venueAbbr = 'HV' if (venue == 'Happy Valley') else 'ST'
            driverResults.get('https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx?RaceDate=%s&Racecourse=%s&RaceNo=%s' % (resultDate(date), venueAbbr, r))
            WebDriverWait(driverResults, 10).until(EC.presence_of_element_located((By.XPATH, xpathResults)))
        except:
            print('No Results available for: ' + prettyDate(date) + ', race ' + r)
            continue
            
        
        # make all hidden columns visible
        driverRaceCard.execute_script("for(var i = 0; i < document.getElementsByTagName('td').length; i++) { document.getElementsByTagName('td')[i].style.display = 'block';}")
        
        raceID += 1
        
        
        # Crawl Results table
        resultsData = []
        try:
            table = driverResults.find_elements_by_xpath(xpathResults)
            for idx, row in enumerate(table[0].find_elements_by_xpath(".//tr")):
                if idx == 0:
                    headerResults = [td.text for td in row.find_elements_by_xpath(".//td")]
                    headerResults = [headerResults[q] for q in featuresResults]
#                    print(headerResults)
#                    print('===================================================== \n')
                else:
                    horseData = [td.text for td in row.find_elements_by_xpath(".//td")]
                    horseData = [horseData[q] for q in featuresResults]
                    resultsData.append(horseData)
#                    print(resultsData)
#                    print('----------------------------------------------------- \n')
                
        except Exception as Fail:
            print( Fail )
        
        # Crawl RaceCard table
        try:
            table = driverRaceCard.find_elements_by_xpath(xpath)
            for idx, row in enumerate(table[0].find_elements_by_xpath(".//tr")):
                if idx < 2:
                    continue
                elif idx == 2:
                    header = [td.text for td in row.find_elements_by_xpath(".//td")]
                    header = ['Date', 'Venue', 'Race'] + [header[q] for q in features] + ['win% Jockey', 'place% Jockey', 'win% Trainer', 'place% Trainer'] + ['Plc.', 'Running Position', 'Finish Time', 'Win Odds']
#                   print(header)
#                   print('===================================================== \n')
                else:
                    raceData = [td.text for td in row.find_elements_by_xpath(".//td")]
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
                    
                    allData.append([prettyDate(date)] + [venue] + [r] + raceData + [winRateJockey, placeRateJockey, winRateTrainer, placeRateTrainer] + findMatch(raceData[0], resultsData))
                    
#                   print(raceData)
#                   print('----------------------------------------------------- \n')
        except Exception as Fail:
            print( Fail )
                
    
    with open('tmp/race_data_%d.csv' % j, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([header] + allData)
            
driverRaceCard.close()
driverResults.close()
        
allData = [d for d in allData if len(d) > 0]   


with open('race_data.csv', 'w', newline='') as f:
     writer = csv.writer(f)
     writer.writerows([header] + allData)
