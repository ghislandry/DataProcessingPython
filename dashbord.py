#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pandas import DataFrame
import os, os.path
import urllib
from settings import APP_STATIC
import math

def dasboard_code():
	datafile = "ecommerce.csv"
	url = 'http://www.semtrack.de/e?i=f654793ba2b71c63e9288fa3c02be7662c5d91c1'
	
	## Download the file from the web if we haven't done it yet!
	filepath = os.path.join(APP_STATIC,datafile)
	 
	if(os.path.isfile(filepath) == False):
		download = urllib.URLopener()
		download.retrieve(url, filepath)

        df = pd.read_csv(str(filepath), sep=";", parse_dates=True, engine="python", encoding=None)
        
	## Format the columns names for better reading
        df.columns = map(lambda x: str(x).translate(None, "_"), df.columns)
        
	## Aggregated number of products per category_1
        ## We will only keep the number of products in stok!
        dfbycat1 = df.groupby(['category1', 'brand']).sum()['availabilityinstock'].reset_index()
        
	## Get the top most expensive products after
        ## To get that number, we first need to remove duplicates from products names to ensure that
        ## A product that appears twice does not get two observations in our final data set.
        grouped = df.groupby('productname')
        index = [gp_keys[0] for gp_keys in grouped.groups.values()]
        udf = df.reindex(index)[['productname', 'description', 'price', 'discountedprice']]
        
	## Sort the dataframe in-place!
        udf.sort('discountedprice', ascending=False,  inplace=True)
        mostexpensivep = udf[0:5][['price', 'productname', 'description']]
        mostexpensivep = mostexpensivep.reset_index()[['price', 'productname', 'description']]
        
	## We assume that cat when a product is not discounted the corresponding discount in not
        ## defined that is NAN
        df["discountoffered"] = map(lambda x: False if math.isnan(x) else True, df.discount)
        
	## data frame containing products not offering discount!
        no_discount_df = df[df["discountoffered"] == False]
        grouped = no_discount_df.groupby('catm')
        index = [gp_keys[0] for gp_keys in grouped.groups.values()]
        no_discount_df = no_discount_df.reindex(index)[['catm']]
	
	#Fix the indexing for better display in html
        z = DataFrame(no_discount_df.reset_index()['catm'], columns=['catm'])
	
	return [dfbycat1, mostexpensivep, z]	
	
