# weather collection script via api
# written by: matt hoover (matthew.a.hoover@gmail.com)
# written for: insight project
# last updated: 28 oct 2015

# import libraries
import requests
import os
import pymysql as mdb
import time as time
import calendar

# define parameters
token = os.getenv('WUNDERGROUND')
sql_call = 'INSERT INTO rain (year, month, day, hour, rainfall) VALUES (%s, %s, %s, %s, %s)'

# loop over years, months, and days
for yr in range(2015, 2016):
	for mth in range(10, 11):
		for day in range(1, 32):
			# error-checking methods
			elif(mth == 2 and day > 28):
				break
			elif(mth in [4, 6, 9, 11] and day > 30):
				break
			elif(yr == 2015 and mth >= 6 and day > 6):
				break 
			
			# open up database connection
			db = mdb.connect(user = 'root', host = 'localhost', password = '', 
				db = 'rain', charset = 'utf8')
			
			# grab current time
			tm = time.time()
			
			# adjust the day and month variables to create address
			str_day = str(day)
			if day <= 9:
				str_day = '0' + str_day

			str_mth = str(mth)
			if mth <= 9:
				str_mth = '0' + str_mth
			
			date = 'history_' + str(yr) + str_mth + str_day
			
			print('Getting data for: ' + str(yr) + '-' + str_mth + '-' + 
				str_day)

			# keep requests to under 10 per minute
			while(time.time() < tm + 6):
				continue
			
			# make request via api
			r = requests.get('http://api.wunderground.com/api/' + token + 
				'/' + date + '/q/Kenya/Nairobi.json').json()

			# keep in loop until information has been recorded
			while(r['response']['features'] == {}):
				r = requests.get('http://api.wunderground.com/api/' + token + 
					'/' + date + '/q/Kenya/Nairobi.json').json()
			
			# extract relevant information from json
			for item in r['history']:
				if type(r['history'][item]) == type({}):
					continue
				for hr in r['history'][item]:

					# write data to database
					db.cursor().execute(sql_call, (hr['date']['year'], 
						hr['date']['mon'], hr['date']['mday'], 
						hr['date']['hour'], hr['rain']))
					db.cursor().close()

			# save and close database
			db.commit()
			db.close()

# note:
#  1. to add in the future, make script robust to pull in last entry from 
#	   database and only pull data from that point forward