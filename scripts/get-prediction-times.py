""" Develop best model for predicting delivery time of meals. 
Written by Matthew Hoover (matthew.a.hoover at gmail.com). """

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from __future__ import division
from scipy.stats import boxcox
from datetime import datetime
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

dir = os.getcwd()
if not os.path.exists('{}/output'.format(dir)):
    os.makedirs('{}/output'.format(dir)

def make_data(d, cat, con, tf = False):
    """ Create analysis datasets """
    d['rand'] = np.random.uniform(-1, 1, len(d))
    d.sort('rand', inplace=True)

    enc = preprocessing.OneHotEncoder()
    enc.fit(d[cat])

    X = np.concatenate((enc.transform(d[cat]).toarray(), d[con]), axis=1)
    X_tr = X[:int(round(len(X) * .6))]
    X_cv = X[int(round(len(X) * .6)):int(round(len(X) * .8))]
    X_ts = X[int(round(len(X) * .8)):]

    if not tf:
        y = d.total_time.values / np.timedelta64(1, 'm')
        y_tr = y[:int(round(len(y) * .6))]
        y_cv = y[int(round(len(y) * .6)):int(round(len(y) * .8))]
        y_ts = y[int(round(len(y) * .8)):]

        return(X_tr, X_cv, X_ts, y_tr, y_cv, y_ts)
    else:
        y, l = boxcox(d.total_time.values / np.timedelta64(1, 'm'))
        y_tr = y[:int(round(len(y) * .6))]
        y_cv = y[int(round(len(y) * .6)):int(round(len(y) * .8))]
        y_ts = y[int(round(len(y) * .8)):]

        return(X_tr, X_cv, X_ts, y_tr, y_cv, y_ts, l)


def reg_run(X_tr, y_tr, X_cv, y_cv, X_ts, y_ts):
    """ Run regression and calculate mean squared error """
    m = linear_model.LinearRegression(normalize=True)
    m.fit(X_tr, y_tr)

    m_ridgecv = linear_model.RidgeCV(alphas = [.001, .003, .01, .03, .1, .3, 
                                     1.0, 3.0, 10.0], normalize=True)
    m_ridgecv.fit(X_cv, y_cv)

    m_ridge = linear_model.Ridge(alpha = m_ridgecv.alpha_, normalize=True)
    m_ridge.fit(X_cv, y_cv)

    return(m, m_ridge)


def trans(x, l):
    """ Implement a power transform """
    return(np.power((x * l) + 1, 1 / l))

d = pickle.load(open('{}/model_data.pkl'.format(dir), 'rb'))
d.drop('rand', axis=1, inplace=True)

times = [d.total_time[d.rest_name.isin([1])].values / np.timedelta64(1, 'm'), 
         d.total_time[d.rest_name.isin([2])].values / np.timedelta64(1, 'm'), 
         d.total_time[d.rest_name.isin([3])].values / np.timedelta64(1, 'm'), 
         d.total_time[d.rest_name.isin([4])].values / np.timedelta64(1, 'm'), 
         d.total_time[d.rest_name.isin([5])].values / np.timedelta64(1, 'm')]

# set up training, cross-validation, and test data
X_tr, X_cv, X_ts, y_tr, y_cv, y_ts = make_data(d, ['rest_name', 
                                               'delivery_hood'], 
                                               ['online', 'open_orders', 'rain', 
                                               'drivers_working', 
                                               'hour'], tf=False)
Xb_tr, Xb_cv, Xb_ts, yb_tr, yb_cv, yb_ts, l = make_data(d, ['rest_name', 
                                                        'delivery_hood'], 
                                                        ['online', 
                                                        'open_orders', 'rain', 
                                                        'drivers_working', 
                                                        'hour'], tf=True)

# train linear model
m, m_ridge = reg_run(X_tr, y_tr, X_cv, y_cv, X_ts, y_ts)
mb, mb_ridge = reg_run(Xb_tr, yb_tr, Xb_cv, yb_cv, Xb_ts, yb_ts)

# train alternate models
rf = RandomForestRegressor(n_estimators=100)
rf.fit(np.concatenate((X_tr, X_cv), axis=0), np.concatenate((y_tr, y_cv)))

rfb = RandomForestRegressor(n_estimators=1000)
rfb.fit(np.concatenate((Xb_tr, Xb_cv), axis=0), np.concatenate((yb_tr, 
        yb_cv)))

gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=.01)
gb.fit(np.concatenate((X_tr, X_cv), axis=0), np.concatenate((y_tr, y_cv)))

gbb = GradientBoostingRegressor(n_estimators=1000, learning_rate=.01)
gbb.fit(np.concatenate((Xb_tr, Xb_cv), axis=0), np.concatenate((yb_tr, 
        yb_cv)))

by = linear_model.BayesianRidge(n_iter=1000, normalize=True)
by.fit(np.concatenate((X_tr, X_cv), axis=0), np.concatenate((y_tr, y_cv)))

byb = linear_model.BayesianRidge(n_iter=1000, normalize=True)
byb.fit(np.concatenate((Xb_tr, Xb_cv), axis=0), np.concatenate((yb_tr, 
        yb_cv)))

# check mean square errors
for val in [m, m_ridge, rf, gb, by]:
    print('mse: {}'.format(round(mean_squared_error(y_ts, val.predict(X_ts)), 
          2)))

for val in [mb, mb_ridge, rfb, gbb, byb]:
    print('mse: {}'.format(round(mean_squared_error(trans(yb_ts, l), 
          trans(val.predict(X_ts), l)), 2)))

# density plots
c1, c2, c3, c4, c5, c6, c7 = sns.color_palette('Set2', 7)
sns.set_style('white')

sns.kdeplot(y_ts, color=c1, label='Actual')
sns.kdeplot(m.predict(X_ts), color=c2, label='Linear')
sns.kdeplot(m_ridge.predict(X_ts), color=c3, label='Linear (reg)')
sns.kdeplot(rf.predict(X_ts), color=c4, label='Random forest')
sns.kdeplot(gb.predict(X_ts), color=c5, label='Boosting')
sns.kdeplot(by.predict(X_ts), color=c6, label='Bayes')

plt.legend(title='Delivery times (test set)')
plt.xlabel('Delivery time (minutes)')
plt.title('Delivery time distribution by method')
plt.savefig('{}/output/deliv_times.pdf'.format(dir))
plt.close()

sns.kdeplot(trans(yb_ts, l), color=c1, label='Actual')
sns.kdeplot(trans(mb.predict(Xb_ts), l), color=c2, label='Linear')
sns.kdeplot(trans(mb_ridge.predict(Xb_ts), l), color=c3, 
            label = 'Linear (reg)')
sns.kdeplot(trans(rfb.predict(Xb_ts), l), color=c4, label='Random forest')
sns.kdeplot(trans(gbb.predict(Xb_ts), l), color=c5, label='Boosting')
sns.kdeplot(trans(byb.predict(Xb_ts), l), color=c6, label='Bayes')

plt.legend(title='Delivery times (test set)')
plt.xlabel('Delivery time (minutes)')
plt.title('Delivery time distribution by method')
plt.savefig('{}/output/deliv_times_trans.pdf'.format(dir))
plt.close()

# random forests way more robust to outliers on both ends -- others are very 
#  clumped in the middle of the distribution

# calculate percent predictions within 10 percent of actual delivery time
for val in [m, m_ridge, rf, gb, by]:
    pct = val.predict(X_ts)<=y_ts + y_ts * .1
    print('within 10%: {}'.format(round(np.mean(pct), 2)))

for val in [mb, mb_ridge, rfb, gbb, byb]:
    pct = trans(val.predict(X_ts), l)<=y_ts + y_ts * .1
    print('within 10%: {}'.format(round(np.mean(pct), 2)))

# plot the residuals versus fitted values to check iid assumption
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharex=True, sharey=True)
ax1.scatter(y_ts - m.predict(X_ts), m.predict(X_ts))
ax1.set_title('Linear regression')
ax1.set_ylabel('Fitted values')
ax2.scatter(y_ts - rf.predict(X_ts), rf.predict(X_ts))
ax2.set_title('Random forest')
ax3.scatter(y_ts - gb.predict(X_ts), gb.predict(X_ts))
ax3.set_title('Gradient boosting')
ax3.set_ylabel('Fitted values')
ax3.set_xlabel('Residual values')
ax4.scatter(y_ts - by.predict(X_ts), by.predict(X_ts))
ax4.set_title('Bayes regression')
ax4.set_xlabel('Residual values')

plt.suptitle('Residual versus predicted values')
plt.savefig('{}/output/residual_plots.pdf'.format(dir))
plt.close()

# plot feature importance for gradient boosting
fimp = gb.feature_importances_

idx = np.argsort(fimp)
ps = np.arange(idx.shape[0]) + .5
fname = np.array(['ArtCaffe', 'Osteria', 'Cedars', 'Sushi Soo', 'La Salumeria', 
                 'Kilimani', 'Lavington', 'Westlands', 'Online', '# orders', 
                 'Rain', '# riders', 'Time of order'])

plt.figure()
plt.subplot(1, 1, 1)

plt.barh(ps, fimp[idx], align='center')
plt.yticks(ps, fname[idx])
plt.xlabel('Importance')
plt.title('Feature importance (gradient boosting)')
plt.savefig('{}/output/feature_importance.pdf'.format(dir))
plt.close()

# decision:
# 1. there is probably not a need for the transform -- performance across 
#	  metrics and visually is not very different or better
# 2. random forests -- on metrics -- performs the worst of all methods; however, 
#	  visually, it makes a very compelling case as it better estimates the 
#	  high/low end of the actual delivery time distribution
# 3. going to go with gradient boosting for prediction coefficients as i'm 
#	  concerned with good prediction and the ability to easily interpret random 
#	  forests is difficult. intuitively, gradient boosting makes more sense, 
#	  even if the distribution is not as 'similar' to actual values -- it 
#	  performs well on metrics

# save coefficients
pickle.dump([gb, times], open('{}/model_params.pkl'.format(dir), 'wb'))
