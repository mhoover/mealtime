# merge in rain data to order data
# written by: matt hoover (matthew.a.hoover@gmail.com)
# written for: insight project
# last edited: 28 oct 2015

# import libraries
import pickle
import os
import pymysql as mdb
import numpy as np
import pandas as pd
from datetime import datetime
from __future__ import division

# functions
# identify unique values
def uniques(x):
	return(len(np.unique(x)) - len(x[x.isnull()]))

# directories
dir = os.getcwd()
if not os.path.exists(dir + '/data'):
	os.makedirs(dir + '/data')

# load order data
d = pickle.load(open(dir + '/data/clean_data.pkl', 'rb'))

# open database connection
db = mdb.connect(user = 'root', host = 'localhost', password = '', 
	db = 'yum', charset = 'utf8')
cursor = db.cursor()

# prep order data with merge data for rain
d.rename(columns = {'day': 'week_of_month'}, inplace = True)

d['month'] = d.order_submitted_date.apply(lambda x: pd.Timestamp(x).month)
d['day'] = d.order_submitted_date.apply(lambda x: pd.Timestamp(x).day)
d['hour'] = d.order_submitted_date.apply(lambda x: pd.Timestamp(x).hour)
d['rain'] = np.nan
d.sort(['order_submitted_date'], inplace = True)

# loop over order data and merge in rainfall value
for i in d.index:
	ans = cursor.execute('SELECT rainfall FROM rain WHERE year = %s AND month = %s AND day = %s AND hour = %s', (int(d.year[i]), int(d.month[i]), int(d.day[i]), int(d.hour[i])))
	if(ans == 0):
		for j in range(int(d.hour[i] - 1), -1, -1):
			ans = cursor.execute('SELECT rainfall FROM rain WHERE year = %s AND month = %s AND day = %s AND hour = %s', (int(d.year[i]), int(d.month[i]), int(d.day[i]), int(j)))
			if(ans > 0):
				d.rain[i] = cursor.fetchone()[0]
				continue
			if(j == 0):
				d.rain[i] = 0
	else:
		d.rain[i] = cursor.fetchone()[0]

# add in new features
d['day_of_week'] = d.order_submitted_date.apply(lambda x: 
	pd.Timestamp(x).weekday())
driver_numbers = d.groupby('order_day').agg({'driver_name': uniques})

d = pd.merge(d, driver_numbers, left_on = 'order_day', right_index = True)

d.rename(columns = {'driver_name_x': 'driver_name', 
	'driver_name_y': 'drivers_working'}, inplace = True)

# save data for modeling
pickle.dump(d, open(dir + '/analysis_data.pkl', 'wb'))
