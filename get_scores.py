import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import os
import math

data_dir = "/home/reddy/MFO/trials-final.csv"
#tfidf_dir = "/home/reddy/MFO/tfidf/"
#wmd_dir = "/home/reddy/MFO/wmd/"
score_dir = "/home/reddy/MFO/scores/"
result_dir = "/home/reddy/MFO/results/"
#gold_dir = "/home/reddy/CDR/gold/"

	
# Load Google's pre-trained Word2Vec model.
print("Model loading ....")
model = gensim.models.KeyedVectors.load_word2vec_format('/home/reddy/clss/wiki-news-300d-1M.vec')
print("Loaded")



fname = open(data_dir,'rt')
data = csv.reader(fname)
data_list = list(data)

for i in range(1,len(data_list)):
	#print(i)
	ID = data_list[i][0]
	sentence = data_list[i][2]

	desc = data_list[i][5]
	desc = desc.replace("?", "?.")
	desc_list = desc.split(". ")#re.split(". | ? | ;",desc)
	
	score = score_dir + ID + ".csv"
	f = open(score,"w+")
	writer = csv.writer(f,delimiter=",")

	d1 = {}
	vect_list = []

	desc_list.append(sentence)

	for i in range(len(desc_list)):
		vectorizer = TfidfVectorizer()
		vectorizer.fit_transform(desc_list)
		vector = vectorizer.transform([desc_list[i]])		
		temp = vector.toarray()
		temp = temp.tolist()
		#print(i,type(temp),len(temp[0]))
		vect_list.append(temp)
	
	max_score = 0
	for i in range(len(desc_list)-1):
		temp = cosine_similarity(vect_list[i],vect_list[-1])
		score = temp[0][0]
		d1[i] = score
		if score > max_score:
			max_score = score

	sd1 = {k: v for k, v in sorted(d1.items(), key=lambda item: item[1],reverse=True)}

	#key_list = list(sd1.keys())


	d2 = {}

	s1 = str(sentence).lower().split()
	for i in range(len(desc_list)-1):
		s2 = str(desc_list[i]).lower().split()
		s1 = [w for w in s1]
		s2 = [w for w in s2]

		distance = model.wmdistance(s1,s2)
		d2[i] = distance

	sd2 = {k: v for k, v in sorted(d2.items(), key=lambda item: item[1],reverse=False)}

	key_list = list(sd2.keys())

	for x in range(len(key_list)):
		position = math.sqrt(float(1)/(float(x)+1))
		tfidf = d1[x]/max_score
		wmd = math.sqrt(float(1)/(float(d2[x])+1))
		length = len(desc_list[x].split())
		writer.writerow([position,tfidf,wmd,length,desc_list[x]])


	f.close()





	


