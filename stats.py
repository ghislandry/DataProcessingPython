#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import sys
import re
import math
import datetime
from datetime import datetime
import statsmodels.api as sm
import scipy as sp



def clean_stats_file(filename):
	#filename = "static/webtrekk_report_2012-11-01_Visit IDs.csv"
	#
	df = pd.io.parsers.read_csv(filename, sep = "\t", engine ="c", nrows=345121, dtype=object)
	#
	df.columns = map(lambda x: x.lower().translate(None, ".").replace(" ",""), df.columns)
	#
	df.time = pd.to_datetime(df.time)
	#
	#
	## For this assigment, we will first split our data set in order to count the number of 
	## visits per hour every day. To accomplish this, let us add two more columns to our data 
	## set the date and the hour of the visit on the corresponding day
	#
	df['day'] = map(lambda x: x.date(), df.time)
	# similarly, let us add the corresponding hour!
	df['hourofday'] = map(lambda x: x.hour, df.time)
	#
	groups = df.groupby(['day', 'hourofday']) 
	#
	dfx = groups['sessionids'].aggregate([len]).reset_index()
	#
	dfx['time'] = map(lambda x, y: datetime.strptime(str(x) + " " + str(y), "%Y-%m-%d %H"), dfx.day, dfx.hourofday)
	dfx = dfx[['time', 'len', 'day', 'hourofday']]
	#
	dfx.columns = ['time', 'visits', 'day', 'hourofday']
	#
	index = pd.Index(dfx.time)
	tsdf = pd.DataFrame(index = index)
	#
	## Now we have our time series of hourly counts of the number of visitors of the web site
	## for the month of november 2012!
	tsdf["visits"] = list(dfx.visits)
	tsdf["days"] = list(dfx.day)
	tsdf["hourofday"] = list(dfx.hourofday)
	tsdf.days = pd.to_datetime(tsdf.days, format="%Y-%m-%d")
	#
	## Let us switch interactive mode off so that plot directly displays the graphic!
	#
	plt.ion()
	#
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
	tsdf[['visits']].plot(ax=ax, fontsize=16)
	plt.savefig("timeplot_visits.png")
	plt.close()
	# We can observe a cyclycal pattern corresponding to weekends and weekdays!
	#
	## let us display basic statistics regarding the patterns of visits of the website
	tsdf.describe()
	# the average number of visits is per hour 478, and half of the time, the website recieves between 227 and 680 visits per hour.
	#
	# To capture the daily variation of the number of visits, we are going to plot a box plot of the number of visits per day
	tsdf[['visits','days']].boxplot(by='days', rot=90)
	z = map(lambda x: x.strftime('%Y-%m-%d'), pd.unique(dfx.day))
	plt.xticks(range(1, 32), z)
	plt.savefig("Boxplotdaily.png")
	plt.close()
	# from the box plot, we can observe that the overall range of visits shows lower values at the begining of the month
	# It then globaby increases progressively and drops down again by the end of the month. Also, we can observe a relatively
	# low numbre of visits from friday do sunday.
	#
	# Let us look at the hourly variation. The hourly visit pattern is a lot easier to interpret. the total number of visits 
	# decreases from mitnight to 5am. Which make sense because most people are sleeping during that period of time. It then start in
	# creasing again from 5am to 11am a goes a little bit dowm during lunch time (from 11am to 1pm). The number of visits also decreases
	# beween 5pm to 7pm where it starts increasing again until 11pm.
	#
	tsdf[['visits','hourofday']].boxplot(by='hourofday')
	z = map(lambda x: x.strftime('%y-%m-%d'), pd.unique(dfx.day))
	plt.xticks(range(1, 31), z)
	plt.savefig("Boxplothourly.png")
	plt.close()
	# 
	## Let split our time series into a trend and a cyclical component.
	v_cycle, v_trend = sm.tsa.filters.hpfilter(tsdf.visits)
	tsdf["visit_cycle"] = v_cycle
	tsdf["visit_trend"] = v_trend
	#
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111)
	tsdf[["visits","visit_trend", "visit_cycle"]].plot(ax=ax, fontsize=16)
	legend = ax.get_legend()
	legend.prop.set_size(20);
	plt.savefig("cycleplot.png")
        plt.close()
	#
	# The overall trend confirm our observation that the website also have a 7 day (weekly) cycle, and that the number of visits 
	# increases: from the first week to the last. 
	#
	## The autocorrelation plot indicates that the series is non-stationary with correlation structure.
	#
	pd.tools.plotting.autocorrelation_plot(tsdf[['visits']])
	plt.savefig("autocorrelation.png")
	plt.close()
	#
	#
	dfdiff = tsdf[["visits"]].diff()
	pd.tools.plotting.autocorrelation_plot(dfdiff[['visits']][1:dfdiff.shape[0]])
	plt.savefig("autocorreldiff.png")
	plt.close()


