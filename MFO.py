import os
import numpy as np
import pandas as pd
import csv
from pythonrouge.pythonrouge import Pythonrouge


score_dir = "/home/reddy/MFO/scores/"
gold_dir = "/home/reddy/MFO/gold/"
best_dir = "/home/reddy/MFO/best_MFO/"

final_scores = "/home/reddy/MFO/results/MFO.csv"

fname = open(final_scores,"w+")
writer = csv.writer(fname,delimiter=",")

a = 1
b = 1
c = 1
#d
def maximize(gen_list,limit):
	d = {}
	d1 = {}
	d2 = {}
	d3 = {}

	sd = {}
	summary1 = ""
	summary2 = ""
	summary3 = ""

	#print("shape", gen_list.shape)
	F1 = gen_list[:,0] #position
	F2 = gen_list[:,1] #tfidf
	F3 = gen_list[:,2] #wmd
	F4 = gen_list[:,3] #length
	F5 = gen_list[:,4]
	

	sentence_list = list(gen_list[:,4])
	#print(sentence_list)
	
	for i in range(len(sentence_list)):
		d1[i] = a*F1[i]
		d2[i] = b*F2[i]
		d3[i] = c*F3[i]
		
	
	#sd = {k: v for k, v in sorted(d.items(), key=lambda item: item[1],reverse=True)}#False)}
	sd1 = {k: v for k, v in sorted(d1.items(), key=lambda item: item[1],reverse=True)}
	sd2 = {k: v for k, v in sorted(d2.items(), key=lambda item: item[1],reverse=True)}
	sd3 = {k: v for k, v in sorted(d3.items(), key=lambda item: item[1],reverse=True)}

	#keys = list(sd.keys())
	keys1 = list(sd1.keys())
	keys2 = list(sd2.keys())
	keys3 = list(sd3.keys())	
	
	#count = 0
	count1 = 0
	count2 = 0
	count3 = 0

	for i in range(len(keys1)):
		if count1 < limit:
			try:
				summary1 = summary1 + str(sentence_list[keys1[i]])
			except:
				print(sentence_list,keys1,i)

	for i in range(len(keys2)):
		if count2 < limit:
			try:
				summary2 = summary2 + str(sentence_list[keys2[i]])
			except:
				print(sentence_list,keys2,i)


	for i in range(len(keys3)):
		if count3 < limit:
			try:
				summary3 = summary3 + str(sentence_list[keys3[i]])
			except:
				print(sentence_list,keys1,i)


	return summary1, summary2, summary3
			




for item in os.listdir(score_dir):

	
	#print("item? : ", item)
	fgold = open(gold_dir+item,'rt')
	data_gold = csv.reader(fgold)
	gold_list = list(data_gold)
	gold_sum = ""

	for i in range(len(gold_list)):
		gold_sum = gold_sum + ". " + gold_list[i][0]
	#print("Gold",gold_sum)


	gen_list = np.array(pd.read_csv(score_dir+item, header=None, encoding='utf-8'))
	#gen_sum = ""

	len_gold = len(gold_sum)
	limit = len_gold*2
	#print(len_gold,gen_limit)
	#count = 0

	gen_sum1, gen_sum2, gen_sum3 = maximize(gen_list,limit)
	#print("\nGen_sum",gen_sum)

	gold = []
	gold.append([[gold_sum]])

	gen1 = []
	gen1.append([gen_sum1])


	gen2 = []
	gen2.append([gen_sum2])


	gen3 = []
	gen3.append([gen_sum3])
	

	#print(gen,gold)	

	r1 = Pythonrouge(summary_file_exist=False,
                summary=gen1, reference=gold,
                n_gram=3, ROUGE_SU4=True, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=250,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=False, samples=1, favor=True, p=0.5)

	r2 = Pythonrouge(summary_file_exist=False,
                summary=gen2, reference=gold,
                n_gram=3, ROUGE_SU4=True, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=250,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=False, samples=1, favor=True, p=0.5)

	r3 = Pythonrouge(summary_file_exist=False,
                summary=gen3, reference=gold,
                n_gram=3, ROUGE_SU4=True, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=250,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=False, samples=1, favor=True, p=0.5)

	ID = item.replace(".csv","")
	score1 = r1.calc_score()
	score2 = r2.calc_score()
	score3 = r3.calc_score()
	#print(score)

	L = []
	L.append(-1)
	L.append(score1["ROUGE-L-F"])
	L.append(score1["ROUGE-L-F"])
	L.append(score1["ROUGE-L-F"])

	max_L = L.index(max(L))
	
	if max_L == 1:
		score = score1
		gen_sum = gen_sum1
	elif max_L == 2:
		score = score2
		gen_sum = gen_sum2		
	elif max_L == 3:
		score = score3
		gen_sum = gen_sum3
	else:
		print("error",L)

	
	if score["ROUGE-L-F"] > 0.75:
		#print(score["ROUGE-2-F"])
		ftemp = open(best_dir+ID+".txt","w+")
		ftemp.write(gen_sum)
		ftemp.close() 
		print(score)
	#xyz = input("continue?")
	writer.writerow([ID,score["ROUGE-1-P"], score["ROUGE-1-R"], score["ROUGE-1-F"], score["ROUGE-2-P"], score["ROUGE-2-R"], score["ROUGE-2-F"], score["ROUGE-L-P"], score["ROUGE-L-R"], score["ROUGE-L-F"]])

fname.close()
