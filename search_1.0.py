###Title : Toy Search program. Perform below functionalities - 
###1. Find simirality between query and document
###2. Find tf-idf weight of a document tokens and query tokens
###3. Find idf
###4. Create postings list arrange in desc order of document for each token
###5. Calculate upper bound if the document is not present in postings list for a given tokens in query

#*******************************************************************************************************
###Author : Mukund Prabhune
###Date   : 17-Oct-2016
###Program Name : search_1.0.py
###Subject : Data Mining
###Assignment : Program assignment 1
###Professor : Dr. Chengkay Li
###Machine : intel i7 6th generation
###Interpreter : python3.6
#*******************************************************************************************************

###Program execution time### 
#### with calculations of vectors and file write disable : 2.91 secs approx.
#### with calculations of vectors and file write enable : 5.22 secs approx.
#### read from files : 0.278 secs approx.
#*******************************************************************************************************

###References###
###ranger dr. chengkay li
###Stakoverflow.com for python concepts
###ReadMe instructions are at bottom of the program

#*******************************************************************************************************
###I/O functions: 
###Input   : List of 30 text files
###Ouput1  : Only console if appropriate flags are ON
###Output2 : Console and tf.json, idf.json, tfidf.json, postings_list.json files
###if below flags are set to - 
###calculate_tf = 1;
###calculate_idf = 1;
###calculate_tf_idf = 1;
###postings_list_flag = 1;
###write_enable_flag = 1;
#*******************************************************************************************************

import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem
from nltk.stem.porter import *
from collections import Counter
from math import log10, sqrt
import json
import time

###global variables
corpusroot = 'E:\DM\presidential_debates'
N = 30;
tf_dict = {};
idf = {};
tf_idf = {};
length_tf_dict = {};
postings_list = {};

###gobal flags 
calculate_tf = 0;				#1 - calculate tf 0 - read from file
calculate_idf = 0;				#1 - calculate idf 0 - read from file
calculate_tf_idf = 0;			#1 - calculate tfidf 0 - read from file
postings_list_flag = 0; 		#1 - calculate postings_list 0 - read from file
write_enable_flag = 0;			#1 - write to files 0- write disbale for all files


#*******************************************************************************************************
###normalize input corpus. This includes tokenization, 
###stemmer and lower case conversion
#*******************************************************************************************************
def	file_normalization(filename):
	stopword_dict = {};
	file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
	doc = file.read()
	file.close()
	doc = doc.lower()
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	tokens = tokenizer.tokenize(doc)
	stemmer = PorterStemmer()
	
	for word in stopwords.words('english'):
		stopword_dict[word] = 1
	stopset = set(stopwords.words('english'))
	d1 = [stemmer.stem(w) for w in tokens if not w in stopword_dict]
	return d1

#*******************************************************************************************************
###calculate tf, find term count in a file
###input is normalized file in d1
#*******************************************************************************************************
def calc_tf(d1):
	cnt = Counter()
	for word in d1:
		cnt[word] += 1
	return cnt

#*******************************************************************************************************	
###find length of term count. THis is used as a denominator 
###for normalize term count in a file
###input is term count(frequency)
#*******************************************************************************************************
def length_tfvec(tf):
	i = 0.0   
	for word in tf:
		i += tf[word]*tf[word]
	return (sqrt(i))

#*******************************************************************************************************	
###calculate idf. idf is frequency / number of documents for a given term
###input is list of all files and normalized terms count
#*******************************************************************************************************
def calc_idf(filename_list, tf_dict):
	global idf
	idf = {};
	if calculate_idf == 1:
		dft = Counter()
		for filename in filename_list:
			tf = tf_dict[filename]
			for word in tf:
				dft[word] += 1
		for word in dft:
			idf[word] = log10(N/dft[word])
		if write_enable_flag == 1:
			f = open("idf.json", "w")
			json.dump(idf, f)
			f.close()
	else:
		f = open("idf.json", "r")
		idf = json.load(f)
		f.close()

#*******************************************************************************************************
###calculate tf_idf for all terms from list of all files
###input is file lists and term frequency vector		
#*******************************************************************************************************
def calc_tf_idf(filename_list, tf_dict):
	global tf_idf;
	tf_idf = {};
	global length_tf_dict;
	length_tf_dict = {};
	
	if (calculate_tf_idf == 1):					#flag to check whether to fetch from json file or calculate
		for filename in filename_list:
			mydict = {}						 
			tf = tf_dict[filename]
			for word in idf:
				if word in tf:
					temp = tf[word]
					mydict[word] = (1 + log10(temp))* idf[word]
				else:
					mydict[word] = 0
			length_tf_dict[filename] = length_tfvec(mydict)
			for word in mydict:
				mydict[word] = mydict[word]/length_tf_dict[filename]
			tf_idf[filename] = mydict		
		if write_enable_flag == 1:
			file1 = open("tfidf.json", "w")
			json.dump(tf_idf, file1)
			file1.close()
	else:
		file1 = open("tfidf.json", "r")
		tf_idf= json.load(file1)
		file1.close()
#*******************************************************************************************************
###retrive idf for a given token. Token is not normalized
#*******************************************************************************************************
def getidf(token):
	if token in idf:
		return idf[token]
	else:
		return -1.0

#*******************************************************************************************************
###retrive tf_idf for a given token. Token is not normalized
#*******************************************************************************************************
def getweight(filename, token):
	if filename in tf_idf: 
		if token in tf_idf[filename]:
			return tf_idf[filename][token]
		else:
			return 0.0
	else:
		return 0.0

#*******************************************************************************************************
###Generate postings list for efficiency. This will create a top 10 documents 
###decending ordered with respect to weights for a given token		
###input tf_idf vector and idf vector. Create postings_list Json file if flag is 0
#*******************************************************************************************************
def	gen_postings_list(tf_idf, idf):
	global postings_list
	postings_list = {};
	
	if (postings_list_flag == 1):
		for word in idf:  
			word_dict = {};
			for filename in tf_idf:		
				word_dict[filename] = tf_idf[filename][word]
			mylist = [];
			for key, value in reversed(sorted(word_dict.items(), key=lambda kv: (kv[1], kv[0]))):
				mylist.append([key, value])
			i = 0;
			mylist2 = [];
			for i in range(0,10):
				mylist2.append(mylist[i])
			postings_list[word] = mylist2
		if write_enable_flag == 1:
			f = open("postings_list.json", "w")
			json.dump(postings_list, f)
			f.close()
	else:
		f = open("postings_list.json", "r")
		postings_list = json.load(f)
		f.close()
	return postings_list

#*******************************************************************************************************	
###calculate normalized weight for an given query
#*******************************************************************************************************
def calc_query_weight(query_text):
	query_text = query_text.lower()
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	tokens = tokenizer.tokenize(query_text)
	stopset = set(stopwords.words('english'))
	d1 = [w for w in tokens if not w in stopwords.words('english')]
	stemmer = PorterStemmer()
	d1 = [stemmer.stem(d) for d in d1]
	#print(d1)
	tf = calc_tf(d1)
	mydict = {}
	for word in tf:
		temp = tf[word]
		mydict[word] = 1 + log10(temp)

	length_tf = length_tfvec(mydict)
	for word in mydict:
		mydict[word] = mydict[word]/length_tf
	return mydict
	
#*******************************************************************************************************
###calculate the upper-bound score for using the query tokens' actual 
###and upper-bound weights with respect to 's vector
###If a document's actual score is better than or equal to the actual scores and 
###upper-bound scores of all other documents, it is returned as the query answer	
#*******************************************************************************************************
def query(query_text):
	query_weight_vec = calc_query_weight(query_text)
	upperbound_dict = {};
	doc_scores = {};
	doc_is_actual_score = {};
	
	for token in query_weight_vec:
		if token in postings_list:
			for i in range(0,10):
				if postings_list[token][i][1] > 0.0: 	 
					upperbound_dict[token] = postings_list[token][i][1]
					if postings_list[token][i][0] in doc_scores:
						doc_scores[postings_list[token][i][0]] += query_weight_vec[token] * postings_list[token][i][1]
					else:
						doc_scores[postings_list[token][i][0]] = query_weight_vec[token] * postings_list[token][i][1]				
					doc_is_actual_score[postings_list[token][i][0]] = 1
					
	for token in query_weight_vec:
		for doc in doc_scores:
			found_flag = 0;
			if token in postings_list:
				for i in range(0,10):
					if postings_list[token][i][1] > 0.0:
						if postings_list[token][i][0] == doc:
							found_flag = 1;			
				if found_flag == 0:			
					if token in upperbound_dict:
						doc_scores[doc] += query_weight_vec[token] * upperbound_dict[token]
						doc_is_actual_score[doc] = 0

	result_doc_name = "None";
	result_doc_score = 0.0;
	
	for doc in doc_scores:
		if doc_scores[doc] > result_doc_score:
			result_doc_score = doc_scores[doc]
			result_doc_name = doc
	if result_doc_name in doc_is_actual_score:
		if doc_is_actual_score[result_doc_name] == 0:
			return ("fetch more", 0.0)
			#print(str(doc_scores))
	return (result_doc_name, result_doc_score)		

#*******************************************************************************************************	
#Main section
#*******************************************************************************************************
if __name__ == "__main__":
	start_time = time.time()	
	tf_dict = {};
	idf = {};
	tf_idf = {};
	length_tf_dict = {};
	
	filename_list = os.listdir(corpusroot);
	
	if (calculate_tf == 1):  
		for filename in filename_list:
			d1 = file_normalization(filename)
			tf = calc_tf(d1)	
			tf_dict [filename] = tf
		if write_enable_flag == 1:
			f = open("tf.json", "w")
			json.dump(tf_dict, f)
			f.close()
	else:
		f = open("tf.json", "r")
		tf_dict = json.load(f)
		f.close()

##Below three functions to create idf, tfidf and postings list 
##for given input corpus		
	calc_idf(filename_list, tf_dict)
	calc_tf_idf(filename_list, tf_dict)
	postings_list = gen_postings_list(tf_idf, idf)	

	
	print("(%s, %.12f)" % query("health insurance wall street"))
	print("(%s, %.12f)" % query("security conference ambassador"))
	print("(%s, %.12f)" % query("particular constitutional amendment"))
	print("(%s, %.12f)" % query("terror attack"))
	print("(%s, %.12f)" % query("vector entropy"))
	print("weights")
	print("%.12f" % getweight("2012-10-03.txt","health"))
	print("%.12f" % getweight("1960-10-21.txt","reason"))
	print("%.12f" % getweight("1976-10-22.txt","agenda"))
	print("%.12f" % getweight("2012-10-16.txt","hispan"))
	print("%.12f" % getweight("2012-10-16.txt","hispanic"))
	print("idf")
	print("%.12f" % getidf("health"))
	print("%.12f" % getidf("agenda"))
	print("%.12f" % getidf("vector"))
	print("%.12f" % getidf("reason"))
	print("%.12f" % getidf("hispan"))
	print("%.12f" % getidf("hispanic"))
	
	end_time = (time.time()-start_time)
	print("program execution time is :")
	print(end_time)

#*******************************************************************************************************		
###Program running instructions:
#Execute program without file write :
####calculate_tf = 1;
####calculate_idf = 1;
####calculate_tf_idf = 1;
####postings_list_flag = 1;
####write_enable_flag = 0;

#Execute program to create files write :
####calculate_tf = 1;
####calculate_idf = 1;
####calculate_tf_idf = 1;
####postings_list_flag = 1;
####write_enable_flag = 1;

#Execute program to read from files without calcualting tf, idf, tfidf, postings_list:
####calculate_tf = 0;
####calculate_idf = 0;
####calculate_tf_idf = 0;
####postings_list_flag = 0;
####write_enable_flag = 0;
	
	
###################################End of Program#####################################
	