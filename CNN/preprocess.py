import csv
import codecs
import numpy as np 
import json
import ast

# function to get unique values 
def unique(list1): 
	x = np.array(list1) 
	return np.unique(x)

def convert_dataset_to_neumeric(fileName):
	with open(fileName, 'r') as f:
	    rowsList = ast.literal_eval(f.read())

	fields_lists = []
	unique_field_list = []
	idxs_to_uniquify = [1, 5, 7, 10, 17, 18] #Venue, Horse, Jockey, Trainer, Priority, Gear
	#idxs_to_uniquify = [1] + [i for i in range (5, 23)] + [26] #Venue, Horse, Jockey, Trainer, Priority, Gear
	idxs_to_dump = [i for i in range(28+6+3+1)]
	for idx,row in enumerate(rowsList):
		if(idx == 0):
			header = row
			fields_lists = [[] for i in range(len(header))]
		else:
			for fidx,field in enumerate(row):
				fields_lists[fidx] += [field]

	dicts_of_uniq_dicts = {}
	for flidx in idxs_to_uniquify:
		unique_field_list = unique(fields_lists[flidx])
		res_dict = dict(zip(unique_field_list, [i+1 for i in range(len(unique_field_list))]))
		with open("./preprocessed/"+header[flidx]+"_enumeration.dict", 'w') as file:
			 file.write(json.dumps(res_dict))
		dicts_of_uniq_dicts[flidx] = res_dict

	converted_rows_list = []
	rows_list = []
	new_header = []
	for j in idxs_to_dump:
		new_header.append(header[j])
	converted_rows_list += [new_header]
	rows_list += [new_header]

	rows_no = len(fields_lists[0])
	for i in range(rows_no):
		new_converted_row = []
		new_row = []
		for j in idxs_to_dump:
			if(j in idxs_to_uniquify):
				numeric_field = dicts_of_uniq_dicts[j][fields_lists[j][i]]
				new_converted_row.append(numeric_field)
			else:
				new_converted_row.append(fields_lists[j][i])
			new_row.append(fields_lists[j][i])
		rows_list += [new_row]
		converted_rows_list += [new_converted_row]

	with open("./preprocessed/"+"subset_cleanedup_uniquified_enumerated_dataset.list", 'w') as file:
		 file.write(json.dumps(converted_rows_list))
	with open("./preprocessed/"+"subset_cleanedup_dataset.list", 'w') as file:
			 file.write(json.dumps(rows_list))

path = './preprocessed/'
fileName=path +"cleanedup_dataset.list"
convert_dataset_to_neumeric(fileName)
