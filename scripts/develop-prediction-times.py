# prediction time model
# written by: matt hoover (matthew.a.hoover@gmail.com)
# written for: insight project
# last edited: 28 oct 2015

# import libraries
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from __future__ import division
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error

# directories
dir = os.getcwd()
if not os.path.exists(dir + '/output'):
	os.makedirs(dir + '/output')

# functions
# create analysis datasets
def make_data(d, cat, con):
	d['rand'] = np.random.uniform(-1, 1, len(d))
	d.sort('rand', inplace = True)

	enc = preprocessing.OneHotEncoder()
	enc.fit(d[cat])
	
	X = np.concatenate((enc.transform(d[cat]).toarray(), d[con]), axis = 1)
	y = d.total_time.values / np.timedelta64(1, 'm')
	
	X_tr = X[:int(round(len(X) * .6))]
	X_cv = X[int(round(len(X) * .6)):int(round(len(X) * .8))]
	X_ts = X[int(round(len(X) * .8)):]
	y_tr = y[:int(round(len(y) * .6))]
	y_cv = y[int(round(len(y) * .6)):int(round(len(y) * .8))]
	y_ts = y[int(round(len(y) * .8)):]
	
	return(X_tr, X_cv, X_ts, y_tr, y_cv, y_ts)

# run regression and calculate mean squared error
def reg_run(X_tr, y_tr, X_cv, y_cv, X_ts, y_ts):
	m = linear_model.LinearRegression(normalize = True)
	m.fit(X_tr, y_tr)
	
	m_ridgecv = linear_model.RidgeCV(alphas = [.001, .003, .01, .03, .1, .3, 
		1.0, 3.0, 10.0], normalize = True)
	m_ridgecv.fit(X_cv, y_cv)
	
	m_ridge = linear_model.Ridge(alpha = m_ridgecv.alpha_, normalize = True)
	m_ridge.fit(X_cv, y_cv)
	
	m_mse = mean_squared_error(y_ts, m.predict(X_ts))
	m_ridge_mse = mean_squared_error(y_ts, m_ridge.predict(X_ts))
	
	return(m, m_mse, m_ridge, m_ridge_mse)

# load order data
d = pickle.load(open(dir + '/analysis_data.pkl', 'rb'))

# outcome and feature selection
# outcome: total_time
# features: payment_type (1 (cash) = [CASH], 2 (mobile) = [COOPMOBILE, MPESA, 
#  ZAP], 3 (credit) = [CREDITCARD, CREDITCARDMC, CREDITCARD_CS_KE, PESAPAL], 4 
#  (promo) = [COUPON_FREE, None], delivery_hood, rest_name, week_of_month 
#  (factor), online, open_orders, rain, time of day
# features to add later: ...
# subset: rest_name = [ArtCaffe The Junction Mall, Osteria Lenana Road, Cedars, 
#  Sushi Soo, La Salumeria Italian Restaurant], delivery_hood = [Kilimani, 
#  Westlands, Lavington], exclude = 0

# condense payment types and make numeric
d.payment_type.loc[d.payment_type == 'CASH'] = 1
d.payment_type.loc[(d.payment_type == 'COOPMOBILE') | 
	(d.payment_type == 'MPESA') | (d.payment_type == 'ZAP')] = 2
d.payment_type.loc[(d.payment_type == 'CREDITCARD') | 
	(d.payment_type == 'CREDITCARDMC') | 
	(d.payment_type == 'CREDITCARD_CS_KE') | 
	(d.payment_type == 'PESAPAL')] = 3
d.payment_type.loc[(d.payment_type == 'COUPON_FREE') | 
	(d.payment_type == 'None')] = 4
d.payment_type = d.payment_type.convert_objects(convert_numeric = True)

# take subset of restaurants and convert to numeric
d.rest_name.fillna(d.restaurant, inplace = True)
d_sub = d[d.rest_name.isin(['ArtCaffe The Junction Mall', 'Osteria Lenana Road', 
	'Cedars', 'Sushi Soo', 'La Salumeria Italian Restaurant'])]
d_sub.rest_name.loc[d_sub.rest_name == 'ArtCaffe The Junction Mall'] = 1
d_sub.rest_name.loc[d_sub.rest_name == 'Osteria Lenana Road'] = 2
d_sub.rest_name.loc[d_sub.rest_name == 'Cedars'] = 3
d_sub.rest_name.loc[d_sub.rest_name == 'Sushi Soo'] = 4
d_sub.rest_name.loc[d_sub.rest_name == 'La Salumeria Italian Restaurant'] = 5
d_sub.rest_name = d_sub.rest_name.convert_objects(convert_numeric = True)

# take subset of neighborhoods and convert to numeric
d_sub = d_sub[d_sub.delivery_hood.isin(['Kilimani', 'Lavington', 'Westlands'])]
d_sub.delivery_hood.loc[d_sub.delivery_hood == 'Kilimani'] = 1
d_sub.delivery_hood.loc[d_sub.delivery_hood == 'Lavington'] = 2
d_sub.delivery_hood.loc[d_sub.delivery_hood == 'Westlands'] = 3
d_sub.delivery_hood = d_sub.delivery_hood.convert_objects(convert_numeric = 
	True)

# extract output vector and feature matrix
d_sub = d_sub[d_sub.exclude.isin([0])]

d_sub = d_sub[['payment_type', 'rest_name', 'delivery_hood', 'day_of_week', 
	'online', 'open_orders', 'rain', 'drivers_working', 'hour', 'total_time']]
d_sub = d_sub.dropna()

d_sub['open2'] = d_sub.open_orders ** 2
d_sub['open3'] = d_sub.open_orders ** 3
d_sub['drivers2'] = d_sub.drivers_working ** 2
d_sub['drivers3'] = d_sub.drivers_working ** 3
d_sub['time_day2'] = d_sub.hour ** 2
d_sub['time_day3'] = d_sub.hour ** 3

d_sub['pay_online'] = d_sub.payment_type * d_sub.online
d_sub['rain_rest'] = d_sub.rain * d_sub.rest_name
d_sub['rain_hood'] = d_sub.rain * d_sub.delivery_hood
d_sub['rain_day'] = d_sub.rain * d_sub.day_of_week
d_sub['rain_open'] = d_sub.rain * d_sub.open_orders
d_sub['rain_drivers'] = d_sub.rain * d_sub.drivers_working
d_sub['rain_hour'] = d_sub.rain * d_sub.hour
d_sub['rain_open_hour'] = d_sub.rain * d_sub.open_orders * d_sub.hour
d_sub['hour_hood'] = d_sub.hour * d_sub.delivery_hood
d_sub['hour_day'] = d_sub.hour * d_sub.day_of_week
d_sub['hour_open'] = d_sub.hour * d_sub.open_orders
d_sub['hour_drivers'] = d_sub.hour * d_sub.drivers_working

# create exploratory visuals
sns.set_style('white')
c1, c2, c3, c4, c5, c6, c7 = sns.color_palette('Set2', 7)

# by-restaurant delivery time densities
sns.kdeplot(d_sub.total_time[d_sub.rest_name.isin([1])] / np.timedelta64(1, 
	'm'), color = c1, label = 'ArtCaffe')
sns.kdeplot(d_sub.total_time[d_sub.rest_name.isin([2])] / np.timedelta64(1, 
	'm'), color = c2, label = 'Osteria')
sns.kdeplot(d_sub.total_time[d_sub.rest_name.isin([3])] / np.timedelta64(1, 
	'm'), color = c3, label = 'Cedars')
sns.kdeplot(d_sub.total_time[d_sub.rest_name.isin([4])] / np.timedelta64(1, 
	'm'), color = c4, label = 'Sushi Soo')
sns.kdeplot(d_sub.total_time[d_sub.rest_name.isin([5])] / np.timedelta64(1, 
	'm'), color = c5, label = 'La Salumeria')

plt.legend(title = 'Restaurant name')
plt.xlabel('Delivery time (minutes)')
plt.title('Delivery time distribution by restaurant')

plt.savefig(dir + '/output/restaurant_density.pdf')
plt.close()

# delivery time densities by rain status
sns.kdeplot(d_sub.total_time[d_sub.rain.isin([0])] / np.timedelta64(1, 'm'), 
	color = c1, label = 'No rain')
sns.kdeplot(d_sub.total_time[d_sub.rain.isin([1])] / np.timedelta64(1, 'm'), 
	color = c2, label = 'Rain')

plt.legend(title = 'Weather conditions')
plt.xlabel('Delivery time (minutes)')
plt.title('Delivery time distribution by weather')

plt.savefig(dir + '/output/rain_density.pdf')
plt.close()

# correlation of continuous features and outcome
sns.set(style = 'darkgrid')

corrd = d_sub[['total_time', 'online', 'open_orders', 'rain', 'drivers_working', 
	'hour', 'day_of_week']]
corrd.total_time = corrd.total_time / np.timedelta64(1, 'm')

corrd.rename(columns = {'total_time': 'Delivery time', 'online': 'Online order', 
	'open_orders': '# orders', 'rain': 'Rain', 'drivers_working': '# drivers', 
	'hour': 'Time of order', 'day_of_week': 'Day of week'}, inplace = True)

f, ax = plt.subplots(figsize = (8, 7))
sns.corrplot(corrd, annot = False, sig_stars = False, diag_names = False, 
	cmap = sns.diverging_palette(250, 0, as_cmap = True), ax = ax)
f.tight_layout()

plt.title('Feature/outcome correlation')
plt.savefig(dir + '/output/correlation_plot.pdf')
plt.close()

# run iterations of model to find best model fit
mse1, mse2, mse3, mse4 = ([], [], [], [])
mse_ridge1, mse_ridge2, mse_ridge3, mse_ridge4 = ([], [], [], [])

for i in range(5000):
	# create datasets for analysis
	X1_tr, X1_cv, X1_ts, y1_tr, y1_cv, y1_ts = make_data(d_sub, 
		['payment_type', 'rest_name', 'delivery_hood', 'day_of_week'], 
		['online', 'open_orders', 'rain', 'drivers_working', 'hour'])
	
	X2_tr, X2_cv, X2_ts, y2_tr, y2_cv, y2_ts = make_data(d_sub, 
		['rest_name', 'delivery_hood'], ['online', 'open_orders', 'rain', 
		'drivers_working', 'hour'])
	
	X3_tr, X3_cv, X3_ts, y3_tr, y3_cv, y3_ts = make_data(d_sub, 
		['payment_type', 'rest_name', 'delivery_hood', 'day_of_week', 
		'pay_online', 'rain_rest', 'rain_hood', 'rain_day', 'hour_hood', 
		'hour_day'], ['online', 'open_orders', 'rain', 'drivers_working', 
		'hour', 'open2', 'open3', 'drivers2', 'drivers3', 'time_day2', 
		'time_day3', 'rain_open', 'rain_drivers', 'rain_hour', 'rain_open_hour', 
		'hour_open', 'hour_drivers'])
	
	X4_tr, X4_cv, X4_ts, y4_tr, y4_cv, y4_ts = make_data(d_sub, 
		['payment_type', 'rest_name', 'delivery_hood', 'day_of_week', 
		'rain_rest', 'rain_day'], ['online', 'open_orders', 'rain', 
		'drivers_working', 'hour', 'rain_open', 'rain_drivers', 'rain_hour', 
		'rain_open_hour', 'hour_open', 'hour_drivers'])
	
	# train linear model
	m1, m1_mse, m1_ridge, m1_ridge_mse = reg_run(X1_tr, y1_tr, X1_cv, y1_cv, 
		X1_ts, y1_ts)
	
	m2, m2_mse, m2_ridge, m2_ridge_mse = reg_run(X2_tr, y2_tr, X2_cv, y2_cv, 
		X2_ts, y2_ts)
	
	m3, m3_mse, m3_ridge, m3_ridge_mse = reg_run(X3_tr, y3_tr, X3_cv, y3_cv, 
		X3_ts, y3_ts)
	
	m4, m4_mse, m4_ridge, m4_ridge_mse = reg_run(X4_tr, y4_tr, X4_cv, y4_cv, 
		X4_ts, y4_ts)
	
	# save mean square errors
	(mse1.append(m1_mse), mse2.append(m2_mse), mse3.append(m3_mse), 
		mse4.append(m4_mse))
	(mse_ridge1.append(m1_ridge_mse), mse_ridge2.append(m2_ridge_mse), 
		mse_ridge3.append(m3_ridge_mse), mse_ridge4.append(m4_ridge_mse))

# print average mean square errors (and number of values used to calculate)
for x in [mse1, mse2, mse3, mse4]:
	print('mean = ' + str(round(np.mean([y for y in x if y < 1000]), 2)) + 
		' (n = ' + str(len([y for y in x if y < 1000])) + ')')
for x in [mse_ridge1, mse_ridge2, mse_ridge3, mse_ridge4]:
	print('mean (ridge) = ' + str(round(np.mean([y for y in x if y < 1000]), 
		2)) + ' (n = ' + str(len([y for y in x if y < 1000])) + ')')

# model 2 performs the best, without parameter regularization

# save data
pickle.dump(d_sub, open(dir + '/model_data.pkl', 'wb'))
