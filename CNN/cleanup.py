import csv
import codecs
import numpy as np 
import json
import datetime, time
from datetime import datetime

# function to get unique values 
def unique(list1): 
	x = np.array(list1) 
	return np.unique(x)

def convert_datestr_daysno(dateStr):
	t = datetime.strptime(dateStr, '%d/%m/%Y')
	dayNo = int(round(time.mktime(t.timetuple())/(24*3600)))
	return dayNo 

def convert_timestr_msec(timeStr):
	try:
		return datetime.strptime(timeStr, '%M:%S.%f').minute*60000+datetime.strptime(timeStr, '%M:%S.%f').second*1000+datetime.strptime(timeStr, '%M:%S.%f').microsecond/1000
	except:
		return ""

def writeCsvFile(fname, data, *args, **kwargs):
    """
    @param fname: string, name of file to write
    @param data: list of list of items

    Write data to file
    """
    mycsv = csv.writer(open(fname, 'wb'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)

def do_initial_cleanup(fileName):
	fileHandle=open(fileName, "r")
	reader = csv.reader(codecs.EncodedFile(fileHandle, 'utf8', 'utf_8_sig'), delimiter=",")
	fields_lists = []
	unique_field_list = []
	for idx,row in enumerate(reader):
		if(idx == 0):
			header = row
			header+= ['L6r-1', 'L6r-2', 'L6r-3', 'L6r-4', 'L6r-5', 'L6r-6' , 'RP-1', 'RP-2', 'RP-3', 'Win']
			fields_lists = [[] for i in range(len(header))] #l6r-1 to l6r-6 and rp-1 to rp-3 and Win
		else:
			for fidx,field in enumerate(row):
				converted_field = field

				# remove the data row when race length (#27) is -1!
				if(int(row[27]) == -1):
					break

				# remove the data row when finishing time is invalid!
				if(convert_timestr_msec(row[25]) == ""):
					break

				# remove the data row when there is no historical information available.
				l6rList = list(row[4].split('/'))
				for l6r in l6rList:
					try:
						int(l6r)
					except:
						l6rList.remove(l6r)

				flag = False
				for i in [23]: #integers
					try:
						int(float(row[i]))
					except:
						flag = True
						break
				if(flag == True):
					break

				if(len(l6rList) == 0): # remove the row if there is no historical info available
					break	

				if(fidx in [2, 3, 23]): #strict integers
					converted_field = int(float(field))

				if(fidx in [6, 8, 9, 11, 12, 13, 14, 15, 16, 27]): #integers
					try:
						converted_field = int(float(field))
					except:
						converted_field = 0
						print "OOOPSSS: data format is wrong for int: <%s:%s>" % (header[fidx], field)

				if(fidx in [0]): #date
					converted_field  = convert_datestr_daysno(field)
				if(fidx in [19, 20, 21, 22]): #float+%
					try:
						converted_field = float(field.strip('%'))
					except:
						converted_field = 0
						print "OOOPSSS: data format is wrong for float-percent: <%s:%s>" % (header[fidx], field)
				if(fidx in [25]): #time
					converted_field = convert_timestr_msec(field)
				if(fidx in [26]): #float
					try:
						converted_field = float(field)
					except:
						converted_field = 0
				if(fidx in [4]): #l6r1-6 e.g. "3/4/2/5/2/3"
					l6rList = list(field.split('/'))
					for l6r in l6rList:
						try:
							int(l6r)
						except:
							l6rList.remove(l6r)

					for l6ridx,l6r in enumerate(l6rList):
							try:
								fields_lists[28+l6ridx] += [int(l6r)]
							except:
								fields_lists[28+l6ridx] += [""]
								print "OOOPSSS: data format is wrong for int: <%s:%s>" % (header[fidx], l6r)
					for i in range(len(l6rList),6):
						fields_lists[28+i] += [""]

				if(fidx in [24]): #rp1-3 "3 4 2 5"
					rpList = field.split(' ')
					rpList = rpList[:-1] # remove the last item of the list, it is the same as place!
					for rpidx,rp in enumerate(rpList):
						if(rpidx > 2):
							print "OOOPSSS more item in rpList than expectation: %"
							continue
						try:
							fields_lists[34+rpidx] += [int(rp)]
						except:
							fields_lists[34+rpidx] += [""]
							print "OOOPSSS: data format is wrong for int: <%s:%s>" % (header[fidx], rp)
					for i in range(len(rpList),3):
						fields_lists[34+i] += [""]

				if(fields_lists[27] != -1):
					fields_lists[fidx] += [converted_field]

				if(fidx in [23]): #place
					if(field == "1"):
						fields_lists[37] += [1]
					else:
						fields_lists[37] += [0]

	cleanedup_rows_list = [header]

	rows_no = len(fields_lists[0])
	for i in range(rows_no):
		cleanedup_row = []
		for j in range(len(fields_lists)):
			cleanedup_row.append(fields_lists[j][i])
		cleanedup_rows_list += [cleanedup_row]

	with open("./preprocessed/"+"cleanedup_dataset.list", 'w') as file:
			 file.write(json.dumps(cleanedup_rows_list))
	writeCsvFile("./preprocessed/"+"cleanedup_dataset.csv", cleanedup_rows_list)

path = './'
fileName=path +"dataset.csv"
do_initial_cleanup(fileName)
