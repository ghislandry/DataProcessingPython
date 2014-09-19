#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import sys
import re
import math

## Remove the squence to_strip form string
def strip_string(string, to_strip):
	if to_strip:
        	while string.startswith(to_strip):
            		string = string[len(to_strip):]
        	while string.endswith(to_strip):
            		string = string[:-len(to_strip)]
		for ch in ['/', '(', ',']:
			string = string.replace(ch,"")
    		return string

## take a keyword and extract its context!
def extract_context(x):
	if(x):
		if x == '-':
			return ("-","-")
		else:
			t = re.split(" ", x) # the last entry of t contains our context
			context = t.pop().translate(None, "()")
			return (' '.join(t).replace("+", ""), context)
	else:
		return ("-", "-") 

def parse_contentadGroup(s):
	if (s):
		s = s.translate(None, "{ID}")
		s = s.rstrip(" ").replace("  ", " ")
		t = s.split(":")
		t = [t[i].rstrip(" ").lstrip(" ") for i in range(len(t))]
		if(len(t) > 1):
			tmp = str(t[1]).split("(")
			tmp.insert(0,t[0])
			t = tmp
			t = [t[i].lstrip(" ").rstrip(" ").strip("\[\]:\(\)") for i in range(len(t))]
			if len(t) < 3:
				t.append("-")
			return (t[0], t[1], ' '.join(t[2:len(t)]))
		else:
			return (t[0], "-", "-")
	else:
		return ("-", "-", "-")



## Pre-process the data
def data_cleaning(filename):
	# We first specify that we need all the columns to be read as caracter strings
	dico = {}
	for i in range(16):
		dico[str(i)] = "str"	  
	df = pd.io.parsers.read_csv(filename, sep = "\t", engine ="python", converters=dico, thousands=",")
	# Make column names more easy to use
	colnames = [df.columns[i].replace(" ", "_").lower().strip("\%\(\)") for i in range(len(df.columns))]
	colnames = [strip_string(colnames[i].rstrip("_"), "op:_") for i in range(len(colnames))]
	# remove all "_" from columns' names
	colnames = [strip_string(colnames[i], "_attribution_multiple_external").translate(None, "_") for i in range(len(colnames))]
	df.columns = colnames
	## Now let us convert all numeric columns to numeric
	numeric_columns = [v for i, v in enumerate(colnames) if i not in set([0, 1, 2])]
	for elt in numeric_columns:
		df[elt] = df[elt].convert_objects(convert_numeric=True)
	#
	## 	Data modeling decision! 
	# Each entry of the semkeyword feature has an additional piece of information,
	# The context of appearance of the keyword: Exact, Phrase, Broad ..
	# We will extract that information to create a new feature! 
	# This make sense, because if a user uses the exact keyword, the odd that it is what he/she
	# is searching for is higher than when the keyword apears in a "Phrase"
	#
	##  
	## Convert the semkeyword to string to replace 
	df.semkeyword = df.semkeyword.apply(str)
	## split df.semkeyword into keywork, and context
	ckey = list( df.semkeyword.apply(extract_context))
	#
	## Create a new data frame with semkeyword and semcontext
	frame = pd.DataFrame(ckey, columns=('semkeyword','semcontext'))
	
	## replace semkeyword with the new keyword and add semcontext
	df.semkeyword = frame.semkeyword
	df['semcontext'] = frame.semcontext
	#
	## The contentadgroup seems to be a completely different table, so as part of the preprocessing,
	## we will simply split the table into the following collumns: Contentadgroup, adgroupId, and adgroupDesc
	#
	tmp = list (df.contentadgroup.apply(parse_contentadGroup))
	#
	#Create a new data frame with the parsed data!
	frame = pd.DataFrame(tmp, columns=('adgroup','adgroupkey', 'adgroupdesc'))
	#
	# Add our new features to the data frame
	df.contentadgroup = frame.adgroup
	df['adgroupkey'] = frame.adgroupkey
	df['adgroupdesc'] = frame.adgroupdesc
	#
	#
	df = df.replace(["-", "", "None"], np.nan)
	return df

	

def analysis(df):
	#
	## First we will remove rows with all NAs
	df = df.dropna(how='all')

	###################    Important ###############################################
	# Next, we will create a new variable that we call "y"or outcome, outcome is 1 #
	# if the observation resulted in an order, otherwise it is zero.               #
	# 									       #
	# Our model will basically consists of predicting the probability that         #
	# Outcome is 1, that is the probability that a search with a given keyword     #
	# will result in an order.						       #
	################################################################################
	#
	#
	
	## Set the seed to prevent data partitions from being changed from one execution to another
        ## This helps make the analysis reproducible
	np.random.seed(1345)

	# 
	df["outcome"] = pd.Categorical(df.ordervalue.apply(lambda x: 0 if math.isnan(x) else 1))
	## We will use the 3/4 th of the data for training our model and the rest for testing/ cross validation
	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
	#
	## List of features that we will use for our prediction model
	features = [v for i, v in enumerate(df.columns) if i not in set([0, 1, 2, 4, 14, 14, 15, 16, 17,18])]
	#
	# Create our test and training data sets
	train, test = df[df['is_train']==True], df[df['is_train']==False]
	#
	#
	## Before fitting our model, we will replace all NA, value with 0
	#
	train = train.fillna(0) 
	#
	ft = train[features].values
	labels = y = train["outcome"].values
	#
	et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, bootstrap=True)
	et_score = cross_val_score(et, ft, labels, n_jobs=-1).mean()
	#
	# Fit our model!
	#
	et.fit(ft, y)
	#
	## Now we can test out model!
	test = test.fillna(0)
	pred = et.predict(test[features].values)
	## Let's display the confusion matrix 
	#
	pd.crosstab(test['outcome'], pred)
	#
	## Now, let us compute the out of sample and cross validation error!
	#
	## Out of Sample error, i.e., the error on the cross validation set #########
	#
	out_sample_error = (1 - (sum([pred[i] == np.asarray(test.outcome)[i] for i in range(len(pred))])/len(test.outcome)))*100 
	# The out of sample error is relatively small, so we can leave the model as it is!
	print("Out of sample error,  i.e., the error on the cross validation set: {0}%".format("%.2f" % out_sample_error))
	
	## Now create a validation test! It is a data set that only include keywords that have seen 
	## less that 100 visits over the period!
	in_validation = map(lambda x: True if x <= 100 else False, df.visitors)
	df["in_validation"] = in_validation
	#
	validation = df[df["in_validation"] == True]
	#
	pred = et.predict(validation[features].values)
	#
	predictions = map(lambda x: True if x == 1 else False, pred)
	validation["predictions"] = predictions
	## Data set containing keywords that have seen less that 100 visits
	## over the period and are likely to generate orders
	good = validation[validation["predictions"] == True]
	#
	## Select 5% of the keywords that are likely to generate orders
	#
	output = np.random.uniform(0, 1, len(pd.unique(good.semkeyword))) <= .05
	good_results = pd.unique(good.semkeyword)[output == True]
	type(good_results)
 	#
	## Data set containing keywords that have seen least that 100 visits 
	## over the period and are least likely to generate generate orders! 
	bad = validation[validation["predictions"] == False]
	#
	output = np.random.uniform(0, 1, len(pd.unique(bad.semkeyword))) <= .05
	bad_results = pd.unique(bad.semkeyword)[output == True]
	
	dfg = pd.DataFrame(good_results, columns=["semkeyword"])	
	dfb = pd.DataFrame(bad_results, columns=["semkeyword"])	
	return [dfg, dfb]

