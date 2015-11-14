""" Collect weather data (rain) for Nairobi on a daily basis 
using the Weather Underground API. Written by Matthew Hoover 
(matthew.a.hoover at gmail.com). """

import requests
import os
import pymysql as mdb
import time
import calendar

token = os.getenv('WUNDERGROUND')
sql_call = 'INSERT INTO rain (year, month, day, hour, rainfall) VALUES (%s, %s, %s, %s, %s)'

# loop over years, months, and days
for yr in range(2012, 2016):
    for mth in range(1, 13):
        for day in range(1, 32):
            # error-checking methods
            if mth==2 and day>28:
                break
            if mth in [4, 6, 9, 11] and day > 30:
                break
            if (yr==int(time.strftime('%Y')) and 
                mth>=int(time.strftime('%m')) and 
                day>int(time.strftime('%d'))):
                break 

            # open up database connection
            db = mdb.connect(user='root', host='localhost', password='', 
                             db='rain', charset='utf8')

            # adjust the day and month variables to create address
            if day<=9:
                str_day= '0{}'.format(day)

            if mth<=9:
                str_mth='0{}'.format(mth)

            date = 'history_{}{}{}'.format(yr, str_mth, str_day)

            print('Getting data for: {}-{}-{}'.format(yr, str_mth, str_day)

            # keep requests to under 10 per minute
            time.sleep(7)

            # make request via api
            r = requests.get('http://api.wunderground.com/api/' + token + 
                             '/' + date + '/q/Kenya/Nairobi.json').json()

            # keep in loop until information has been recorded
            while r['response']['features']=={}:
                r = requests.get('http://api.wunderground.com/api/' + token + 
                                 '/' + date + '/q/Kenya/Nairobi.json').json()

            # extract relevant information from json
            for item in r['history']:
                if type(r['history'][item])==type({}):
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
