import csv
import codecs
import numpy as np 
import json
import ast

rowsList = []
fieldsList = []
def fetch_dataset(fileName):
	global rowsList
	global fieldsList
	with open(fileName, 'r') as f:
	    rowsList = ast.literal_eval(f.read())	
	for idx,row in enumerate(rowsList):
		if(idx == 0):
			header = row
			fieldsList = [[] for i in range(len(header))]
		else:
			for fidx,field in enumerate(row):
				fieldsList[fidx] += [field]

def print_header():
	for i,s in enumerate(rowsList[0]):
		print(i,s)

def find_histogram_l6run():
	counter = {}
	for i in range(7):
		counter[i] = 0
	
	for row in rowsList:
		flag = False
		for i in range(33, 27, -1):
			if (row[i] != ""):
				counter[i-27] = counter[i-27] + 1 
				flag = True
				break
		if(flag == False):
			counter[0] = counter[0] + 1
	print (counter)

def replicate_prev_runs():
	for row in rowsList:
		flag = False
		for i in range(29, 34):
			if (row[i] == ''):
				row[i] = row[i-1]

def find_histogram_race_length():
	histogram = {}
	
	for row in rowsList[1:]:
		race_length = row[27]
		if(race_length in histogram):
			histogram[race_length] = histogram[race_length] + 1
		else:
 			histogram[race_length] = 1

	print (histogram)

def find_histogram_venue():
	histogram = {}
	
	for row in rowsList[1:]:
		race_length = row[1]
		if(race_length in histogram):
			histogram[race_length] = histogram[race_length] + 1
		else:
 			histogram[race_length] = 1

	print (histogram)

def find_histogram_horse_no():
	histogram = {}
	
	for row in rowsList[1:]:
		horse_no = row[3]
		if(horse_no in histogram):
			histogram[horse_no] = histogram[horse_no] + 1
		else:
 			histogram[horse_no] = 1

	print (histogram)

def add_normalized_finishing_time():
	rowsList[0].append("normalized_finishing_time")
	for i in range(1, len(rowsList)):
		row = rowsList[i]
		race_length = row[27]
		finishing_time = row[25]
		normalized_finishing_time = finishing_time * float(2400)/float(race_length)
		rowsList[i].append(normalized_finishing_time)
		print (finishing_time, normalized_finishing_time)

clusters = {}
def do_clustering():
	global clusters
	for i in range(1, len(rowsList)):
		row = rowsList[i]		
		date = row[0]
		venue = row[1]
		race = row[2]
		cluster_id = ("%d_%d_%d"%(date, venue, race))
		if (cluster_id in clusters):
			clusters[cluster_id].append(row)
		else:
			clusters[cluster_id] = [row]

	#for cluster_id in clusters:
	#	print (cluster_id, len(clusters[cluster_id]))

	#for cluster_id in clusters:
	#	if (len(clusters[cluster_id]) == 3):
	#		print (clusters[cluster_id])

def print_clusters_histogram():
	histogram = {}
	for cluster_id in clusters:
		key = len(clusters[cluster_id])
		if(key in histogram):
			histogram[key] = histogram[key] + 1
		else:
 			histogram[key] = 1

	for h in histogram:
		print h,histogram[h], h*histogram[h]

clusters_removed_missing_horses = {}	
def remove_clusters_with_missing_horses():
	global clusters_removed_missing_horses
	for cluster_id in clusters:
		rows_list = clusters[cluster_id]
		horses_set = set()
		for row in rows_list:
			horses_set.add(row[3])
		flag = True
		for i in range(1, len(horses_set)+1):
			if(i not in horses_set):
				flag = False
				break
		if(flag == True):
			clusters_removed_missing_horses[cluster_id] = clusters[cluster_id]

		histogram = {}
	for cluster_id in clusters_removed_missing_horses:
		key = len(clusters_removed_missing_horses[cluster_id])
		if(key in histogram):
			histogram[key] = histogram[key] + 1
		else:
 			histogram[key] = 1

	for h in histogram:
		print h,histogram[h], h*histogram[h]			

clusters_removed_small_clusters = {}	
def remove_clusters_with_small_horses(noHorses):
	global clusters_removed_small_clusters
	for cluster_id in clusters:
		rows_list = clusters[cluster_id]
		if(len(rows_list) > noHorses):
			clusters_removed_small_clusters[cluster_id] = clusters[cluster_id]

def check_clusters_order():
	for cluster_id in clusters:
		rows_list = clusters[cluster_id]
		horses_set = set()
		prev_horse_no = 0
		for row in rows_list:
			if(row[3] < prev_horse_no):
				print "OOOPSSS"
				exit()
			else:
				prev_horse_no = row[3]
				#print "good"

def add_missing_horses_to_clusters(numHorses):
	for cluster_id in clusters:
		rows_list = clusters[cluster_id]
		last_horse_no = rows_list[-1][3]
		new_row_to_add = rows_list[len(rows_list)-1]
		index_to_add = len(rows_list)-1

		if (new_row_to_add[23] == 1): # if place of horse is first choose another row
			new_row_to_add = rows_list[len(rows_list)-2]
			last_horse_no = last_horse_no + 1
			index_to_add = len(rows_list)-2

		row_num_horse = len(rows_list)

		for i in range(len(new_row_to_add)):
			new_row_to_add[i] = 0

		for hNo in range(row_num_horse+1, numHorses+1):
			last_horse_no = last_horse_no+1
			new_row_to_add[3] = last_horse_no
			clusters[cluster_id].insert(index_to_add, new_row_to_add)

import random

def make_final_lists():
	interesting_list = [x for x in range(5,23)]+[26]+[28,29,30]
	label_idx = 37

	final_train_list = []
	final_test_list = []

	final_train_label_list = []
	final_test_label_list = []

	keys =  list(clusters.keys())
	random.shuffle(keys)
	train_keys = [x for x in keys[0:int(len(keys)*0.90)]]
	test_keys = [x for x in keys if x not in train_keys]
	max_list = {}

	for i in interesting_list:
		max_list[i] = -1000

	for cluster_id in keys:
		row_list = clusters[cluster_id]
		for i in interesting_list:
			for row in row_list:
				max_list[i] = max(max_list[i], row[i])


	print "ine", max_list


	for cluster_id in train_keys:
		sub_list = []
		row_list = clusters[cluster_id]
		for i in interesting_list:
			for row in row_list:
				#sub_list.append(float(row[i])/float(max_list[i]))
				sub_list.append(float(row[i]))
		final_train_list.append(sub_list)
		for row in row_list:
			final_train_label_list.append(row[label_idx])		

	for cluster_id in test_keys:
		sub_list = []
		row_list = clusters[cluster_id]
		sorted_row_list = []
		# sort row_list according to draw column
		for i in range(1, 15):
			for row in row_list:
				if(row[9] == i): # draw column
					sorted_row_list.append(row)
		for row in row_list:
			if(row[9] == 0): # draw column
				sorted_row_list.append(row)

		for i in interesting_list:
			for row in sorted_row_list:
				#sub_list.append(float(row[i])/float(max_list[i]))
				sub_list.append(float(row[i]))
		final_test_list.append(sub_list)
		for row in sorted_row_list:
			final_test_label_list.append(row[label_idx])	

	print len(train_keys), len(test_keys), len(final_train_list), len(final_test_list)
	print len(final_train_label_list), len(final_test_label_list)

	with open("./preprocessed/"+"final_train.list", 'w') as file:
			 file.write(json.dumps(final_train_list))

	with open("./preprocessed/"+"final_test.list", 'w') as file:
			 file.write(json.dumps(final_test_list))

	with open("./preprocessed/"+"final_train_label.list", 'w') as file:
			 file.write(json.dumps(final_train_label_list))

	with open("./preprocessed/"+"final_test_label.list", 'w') as file:
			 file.write(json.dumps(final_test_label_list))

def dummy_print():
	for cluster_id in clusters:
		rows_list = clusters[cluster_id]
		try:
			last_horse_rank = rows_list[-1][23] * rows_list[-2][23]
		except:			
			return

		if (last_horse_rank == 1):
			print("OOOPS")

def normalize_dataset(fileName):
	
	idxs_to_dump = [i for i in range(28+6+3)]

	dicts_of_uniq_dicts = {}
	for flidx in idxs_to_uniquify:
		unique_field_list = unique(fieldsList[flidx])
		res_dict = dict(zip(unique_field_list, [i for i in range(len(unique_field_list))]))
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

	rows_no = len(fieldsList[0])
	for i in range(rows_no):
		new_converted_row = []
		new_row = []
		for j in idxs_to_dump:
			if(j in idxs_to_uniquify):
				numeric_field = dicts_of_uniq_dicts[j][fieldsList[j][i]]
				new_converted_row.append(numeric_field)
			else:
				new_converted_row.append(fieldsList[j][i])
			new_row.append(fieldsList[j][i])
		rows_list += [new_row]
		converted_rows_list += [new_converted_row]

	with open("./preprocessed/"+"subset_cleanedup_uniquified_enumerated_dataset.list", 'w') as file:
		 file.write(json.dumps(converted_rows_list))
	with open("./preprocessed/"+"subset_cleanedup_dataset.list", 'w') as file:
			 file.write(json.dumps(rows_list))

path = './preprocessed/'
fileName=path +"subset_cleanedup_uniquified_enumerated_dataset.list"
fetch_dataset(fileName)
print_header()
find_histogram_l6run()
replicate_prev_runs()
find_histogram_l6run()
find_histogram_race_length()
find_histogram_venue()
find_histogram_horse_no()

do_clustering()
check_clusters_order()
print_clusters_histogram()

print("*"*80)
remove_clusters_with_small_horses(9)
clusters = clusters_removed_small_clusters
print_clusters_histogram()

print("*"*80)
add_missing_horses_to_clusters(14)
print_clusters_histogram()

print("*"*80)
make_final_lists()

#dummy_print()
#remove_clusters_with_missing_horses()
#add_unique_raceID()
#add_normalized_finishing_time()
print_header()
#normalize_dataset(fileName)
