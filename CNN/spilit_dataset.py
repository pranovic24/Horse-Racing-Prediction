import csv
import codecs
import numpy as np 
import json
import ast

# function to get unique values 
def unique(list1): 
	x = np.array(list1) 
	return np.unique(x)

def split(fileName, split_column):
	with open(fileName, 'r') as f:
	    rowsList = ast.literal_eval(f.read())

	fields_lists = []
	unique_field_list = []
	idxs_to_dump = [i for i in range(28+6+3)]
	for idx,row in enumerate(rowsList):
		if(idx == 0):
			header = row
			fields_lists = [[] for i in range(len(header))]
		else:
			for fidx,field in enumerate(row):
				fields_lists[fidx] += [field]

	unique_field_list = unique(fields_lists[split_column])
	unique_field_no = len (unique_field_list)

	for unique_field in unique_field_list: 
		splited_rows_list = []
		new_header = []
		for j in idxs_to_dump:
			new_header.append(header[j])
		splited_rows_list += [new_header]

		rows_no = len(fields_lists[0])
		for i in range(rows_no):
			new_converted_row = []
			new_row = []
			if(fields_lists[split_column][i] != unique_field):
				continue
			for j in idxs_to_dump:
				new_converted_row.append(fields_lists[j][i])
			splited_rows_list += [new_converted_row]

		file_name = "./preprocessed/"+"split_dataset_" + header[split_column] + "_" +str(unique_field) + ".list"
		with open(file_name, 'w') as file:
			 file.write(json.dumps(splited_rows_list))

path = './preprocessed/'
fileName=path +"split_dataset_Venue_Sha Tin.list"
split(fileName, 27)
