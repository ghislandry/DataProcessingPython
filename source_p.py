#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pandas import DataFrame
import re
import math
import os
from settings import APP_STATIC

## 
def to_print(x):
	return '%s' % ' '.join(x['countryCode'])


## Create a (size x 1) numpy array containing the string as value
def make_column(string, size):
	new_col = [string for i in range(size )]
	new_col = np.array(new_col)
	new_col = new_col.reshape(size, 1)
	return new_col

## A generic fuction for reading data frames!
def read_datafile(datafile, header_skip, footer_skip, delim, string):
	path = os.path.join(APP_STATIC,datafile)
	data = np.recfromcsv(path, skip_header=header_skip, skip_footer=footer_skip, autostrip=True, delimiter=delim)
	data = np.array(map(list, data)) # convert the data into a normal numpy array
	extra_col = make_column(string, data.shape[0])
	data = np.hstack((data, extra_col))
	return data

## Display the array passed as argument in 3 lines or less
def Q1(array): 
	v = [array[i][j] for (i,j) in [(i,j) for i in range(len(array)) for j in range(len(array[i]))]]
	return '%s' % ' \t'.join(v)

## f1, f2, and f3 are names of the files that we want to load
def Q2(f1, f2, f3):
	d1 = read_datafile(os.path.join(APP_STATIC,f1), 8, 2, "\t", "VN")
	d2 = read_datafile(os.path.join(APP_STATIC,f2), 0, 0, "\t", "ID")
	d3 = read_datafile(os.path.join(APP_STATIC,f3), 0, 0, "\t", "PH")

	all_data = np.vstack((d1, d2, d3))
	df = DataFrame(all_data, columns=["days", "pageImpressions", "visits" , "bounces", "countryCode"])
	
	## Convert numeric columns to numeric
	for k, v in df.iloc[:,1:4].convert_objects(convert_numeric=True).iteritems():
		df[k] = v

	## Summary of the day data frame to get the daily regional number for each columns accross countries

	dfsum = df.groupby(['days']).sum()
        ## reset the index to get the days as a column instead of index
        dfsum = dfsum.reset_index()
	
        ## Let us subset the data frame again to only keep days and country code
        days_country = df[['days', 'countryCode']]

        ## Convert the countryCode to string so that we can apply our join function
        days_country['countryCode'] = days_country['countryCode'].astype(str)
        ## group Country codes by the dates on which data where collected!

        days_country = days_country.groupby(['days']).apply(to_print)

        # to simplyfy, we will assume that multiple columns form a region
        days_country = DataFrame(days_country, columns=["regions"])

        ## As for the previous case, reset the index to get the days as a column instead of index
        days_country = days_country.reset_index()

        ## Let us merge the two data sets to get what we expect!
        dfsum = DataFrame.merge(dfsum, days_country, on = ['days'])

	return [df, dfsum]
		
