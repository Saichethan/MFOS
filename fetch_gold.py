import numpy as np
import csv
import os


data_dir = "/home/reddy/MFO/trials-final.csv"
gold_dir = "/home/reddy/MFO/gold/"



fname = open(data_dir,'rt')
data = csv.reader(fname)
data_list = list(data)

for i in range(1,len(data_list)):

	ID = data_list[i][0]
	gold = data_list[i][4]
	gold = gold.replace("?","?.")
	gold_list = gold.split(". ") 

	gold_ = gold_dir + ID + ".csv"
	f1 = open(gold_,"w+")
	writer_ = csv.writer(f1,delimiter=",")

	for x in range(len(gold_list)):
		writer_.writerow([gold_list[x]])

	f1.close()





