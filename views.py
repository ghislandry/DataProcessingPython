from __future__ import division
from flask import render_template, request, url_for
from flask import jsonify
from app import app
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pandas import DataFrame
import os
import re
import math, random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from settings import APP_STATIC, ML_DATA_FILE
from ml_python import data_cleaning, analysis, parse_contentadGroup, extract_context, strip_string
from source_python import to_print, make_column, read_datafile, Q1, Q2
from tenis_problem import permutations, format_team, schedule
from ecommerce_dashbord import dasboard_code


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title = 'Home')


@app.route('/python')
def python():
	## Code source available in source_python.py
	## file names
	f1 = "webtrekk_report_2012-12-15_Sandra VN.csv"
	f2 = "webtrekk_report_2012-12-15_Sandra ID.csv"
	f3 = "webtrekk_report_2012-12-15_Sandra PH.csv"

    	array=[['a','b','c'],['d','e','f']]
	y = Q1(array)
		
	df, dfsum = Q2(f1, f2, f2) 

	return render_template('python.html',title = "Python", x=df, y = y, z = dfsum)


@app.route('/ecommerce')
def ecommerce():
	## Code in ecommerce_dashbord.py
		
	dfbycat1, mostexpensivep, no_discount_catm = dasboard_code()
	dfbycat1.category1 = map(lambda x: str(x).decode('utf8', 'ignore'), dfbycat1.category1)
	dfbycat1.brand = map(lambda x: str(x).decode('utf8', 'ignore'), dfbycat1.brand)
	dfbycat1.availabilityinstock = map(lambda x: str(x).decode('utf8', 'ignore'),dfbycat1.availabilityinstock)
	return render_template("ecommerce.html", title = "E-commerce", x=dfbycat1, y=mostexpensivep, z=no_discount_catm)


@app.route('/machinelearning')
def machinelearning():
	## The code is in ml_python.py
		
	datafile = ML_DATA_FILE
        filename = os.path.join(APP_STATIC,datafile)
	
	tidydata = data_cleaning(filename)
	
	good, bad = analysis(tidydata)
	
	return render_template("machinelearning.html", title = "Machine Learning", x=good, y=bad)

@app.route('/tennisproblem')
def form():
	return render_template('form_submit.html')


@app.route('/tennisproblem', methods=['POST'])
def tennisproblem():
	## code in tenis_problem.py
	n = int (request.form['numberplayers'])
	valid_number = 0
	df = None
	if(n > 1):
		valid_number = 1
		df = schedule(n)

	return render_template("tennisproblem.html", title = "The tennis problem", x=df, y=valid_number, z=n)
@app.route('/stats')
def stats():
	return render_template("stats.html", title = "Stats")


@app.route('/bashscripting')
def bashscripting():
	return render_template("bashscripting.html", title = "Bash Scripting")
