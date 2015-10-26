# render flask template for user interface
# written by: matt hoover (matthew.a.hoover@gmail.com)
# written for: insight project
# last edited: 28 jun 2015

# import libraries
from flask import render_template, request
from app import app
import numpy as np
import scipy as sp
import model_funcs as mf
import pickle
import pandas as pd

# directories
dd = '/home/ubuntu/webapp/data/'

# load prediction parameters
params, distros = pickle.load(open(dd + 'model_params.pkl', 'rb'))

# define app routines
@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html")

@app.route('/about')
def mealtime_about():
	return render_template('about.html')

@app.route('/analytics')
def mealtime_analytics():
	return render_template('analytics.html')

@app.route('/slides')
def mealtime_slides():
	return render_template('slides.html')

@app.route('/input')
def mealtime_input():
	return render_template('input.html')

@app.route('/output')
def mealtime_ouput():
	restaurant = int(request.args.getlist('restaurant')[0])
	neighborhood = request.args.getlist('place')[0]
	online = request.args.get('online')
	orders = request.args.get('orders')
	rain = request.args.get('rain')
	riders = request.args.get('riders')
	hour = request.args.get('hour')
	
	if((orders == '') or (riders == '') or (hour == '')):
		return render_template('output_error.html')
	elif mf.valid_params(orders, riders, hour) == False:
		return render_template('output_error.html')
	
	restaurants = ['ArtCaffe - Junction', 'Osteria', 'Cedars', 'Sushi Soo', 
		'La Salumeria']
	
	rest_range = range(5)
	rest_range.pop(int(restaurant))
	
	select = np.zeros(5)
	select[restaurant] = 1
	select = select.tolist()
	
	distro = distros[restaurant].tolist()
	smin = np.min(distro)
	smax = np.max(distro)
	
	h_numeric = mf.hood_vec(neighborhood)
	o_numeric = mf.binary_vec_opp(online)
	rn_numeric = mf.binary_vec(rain)

	deliv_time = mf.predict(params, h_numeric, o_numeric, orders, 
		rn_numeric, riders, hour)
	
	bootstrap = []
	bootstrap = mf.bs_se(deliv_time, bootstrap)
	
	for i in deliv_time:
		i = int(round(i))
	
	deliv_time = deliv_time.astype(int)
	
	bootstrap = np.ceil(deliv_time[restaurant] + sp.stats.norm.ppf(.99) * 
		np.std(bootstrap)).astype(int)
	
	times = deliv_time.tolist()
	
	return render_template('output.html', restaurants = restaurants, 
		deliv_time = deliv_time, bootstrap = bootstrap, 
		restaurant = restaurant, other_rest = rest_range, select = select, 
		times = times, distro = distro, smin = smin, smax = smax)
