#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Sun May  5 21:09:46 2019'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''


"""

Name:           filename.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Rachel Green
ORGANIZATION:   University of California, Santa Barbara
Contact:        rachel.green@geog.ucsb.edu
Copyright:      (c) Rachel Green 2019


"""


#%% IMPORTS
import os
import numpy as np
import pandas as pd
import math as m

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from scipy.fftpack import fft, rfft, fftfreq
from scipy import signal
import pywt
import waipy

from stargazer.stargazer import Stargazer
from IPython.core.display import HTML



%matplotlib qt5 

#%% FUNCTIONS



#%% MAIN
def main():
#%%    
    
#Load data
    os.chdir('/Users/rgreen/Documents/Github/NDVI_Projection/')
    
    oromia = pd.ExcelFile('oromia.xlsx')
    arsi = pd.read_excel(oromia, 'arsi')
    bale = pd.read_excel(oromia, 'bale')
    borena = pd.read_excel(oromia, 'borena')
    guji = pd.read_excel(oromia, 'guji')
    westarsi = pd.read_excel(oromia, 'westarsi')
    
    
    arsi.insert(0, 'Time', np.linspace(1,585,585))
    bale.insert(0, 'Time', np.linspace(1,585,585))
    borena.insert(0, 'Time', np.linspace(1,585,585))
    guji.insert(0, 'Time', np.linspace(1,585,585))
    westarsi.insert(0, 'Time', np.linspace(1,585,585))
    
#%%   
    #dekadal data (P, LST, ET)    

    ddf = pd.DataFrame()
    ddf['D_NDVI'] = arsi.NDVI.diff()[1:]
    ddf = ddf.reset_index(drop=True)
    ddf['N_P'] = ((max(arsi.NDVI) - (arsi.NDVI[:-1]))* (arsi.P[:-1]))
    ddf['N_LST'] = (((arsi.NDVI[:-1]) - min(arsi.NDVI))* (arsi.LST[:-1])).shift(4) #need to shift back lags, use shift not index
    ddf['N_ET'] = (((arsi.NDVI[:-1]) - min(arsi.NDVI))* (arsi.ET[:-1])).shift(4)

    
    ddf = pd.DataFrame()
    ddf['D_NDVI'] = arsi.NDVI.diff().shift(-1)
    ddf = ddf.reset_index(drop=True)
    #L_NDVI = arsi.NDVI.shift(-1) #lag
    ddf['N_P'] = ((max(arsi.NDVI) - (arsi.NDVI))* (arsi.P))
    ddf['N_LST'] = (((arsi.NDVI) - min(arsi.NDVI))* (arsi.LST)).shift(4) #need to shift back lags, use shift not index
    ddf['N_ET'] = (((arsi.NDVI) - min(arsi.NDVI))* (arsi.ET)).shift(4)
    
    
    
    #shift is (n-1) dekads, ex. shift(4) is 5 dekads lag
        
    #mask
    mask_lst = ~np.isnan(ddf.N_LST) & ~np.isnan(ddf.D_NDVI)
    mask_et = ~np.isnan(ddf.N_ET) & ~np.isnan(ddf.D_NDVI)
    
    #stats.lingress (X,Y)
    slope1, intercept1, r1, p1, std1 = stats.linregress(ddf.N_P, ddf.D_NDVI)
    line1 = slope1*ddf.N_P+intercept1
    print("r-squared: %f" % r1**2)
    slope2, intercept2, r2, p2, std2 = stats.linregress(ddf.N_LST[mask_lst], ddf.D_NDVI[mask_lst])
    line2 = slope2*ddf.N_LST+intercept2
    print("r-squared: %f" % r2**2)
    slope3, intercept3, r3, p3, std3 = stats.linregress(ddf.N_ET[mask_et], ddf.D_NDVI[mask_et])
    line3 = slope3*ddf.N_ET+intercept3
    print("r-squared: %f" % r3**2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharey =True)
    ax1.scatter(ddf.N_P, ddf.D_NDVI, color = 'darkcyan')
    ax1.plot(ddf.N_P, line1, color = 'k')
    ax1.set(xlabel = r'$(NDVI_{max} - NDVI_{t-1})*P_{t-1}$', ylabel = '')
    ax1.text(16, -0.04, r'y = 0.005x + 0.02', fontsize=8)
    ax1.text(16, -0.05, r'$r^2$ = 0.528', fontsize=8)
    ax2.scatter(ddf.N_LST, ddf.D_NDVI, color = 'forestgreen')
    ax2.set(xlabel = r'$(NDVI_{t-5} - NDVI_{min})*LST_{t-5}$', ylabel = '')
    ax2.plot(ddf.N_LST, line2, color = 'k')
    ax2.text(9, 0.07, r'y = -0.005x + 0.038', fontsize=8)
    ax2.text(9, 0.06, r'$r^2$ = 0.386', fontsize=8)
    ax3.scatter(ddf.N_ET, ddf.D_NDVI, color = 'cornflowerblue')
    ax3.set(xlabel = r'$(NDVI_{max} - NDVI_{t-5})*ET_{t-5}$', ylabel = '')
    ax3.plot(ddf.N_ET, line3, color = 'k')
    ax3.text(15, 0.07, r'y = -0.003x + 0.031', fontsize=8)
    ax3.text(15, 0.06, r'$r^2$ = 0.414', fontsize=8)
    fig.text(0.06, 0.5, r'$\Delta NDVI$', ha='center', va='center', rotation='vertical') #common ylabel
# =============================================================================
#     
#     sns.set(style="ticks", color_codes=True)
#     
#     fig = plt.figure()
#     sns.regplot(x=ddf.N_P, y=ddf.D_NDVI)
#     
#     g = sns.PairGrid(ddf, y_vars=["D_NDVI"], x_vars=["N_P", "N_LST", "N_ET"], height=4)
#     g.map(sns.regplot, color=".3")
#     
#     replacements = {'D_NDVI': r'$\Delta NDVI$', 'N_P': '(maxNDVI - NDVIt-1)*Pt-1',
#                 'N_LST': '(NDVIt-5 - minNDVI)*LSTt-5', 'N_ET': '(NDVIt-5 - minNDVI)*N_ETt-5'}
# 
#     for i in range(4):
#         for j in range(4):
#             xlabel = g.axes[i][j].get_xlabel()
#             ylabel = g.axes[i][j].get_ylabel()
#             if xlabel in replacements.keys():
#                 g.axes[i][j].set_xlabel(replacements[xlabel])
#             if ylabel in replacements.keys():
#                 g.axes[i][j].set_ylabel(replacements[ylabel])
#         
# =============================================================================
    
# =============================================================================
#    
#     r, p = stats.pearsonr(ddf.D_NDVI, ddf.N_P)
#     r, p = stats.pearsonr(ddf.D_NDVI, ddf.N_LST)
#     r, p = stats.pearsonr(ddf.D_NDVI, ddf.N_LST)
#     
# =============================================================================
    
 #%%   

    X = ddf[['N_P']]
    Y = ddf['D_NDVI']
    X = sm.add_constant(X)
    est = sm.OLS(Y,X, missing = 'drop').fit()
    est.summary()
    Y_pred = est.predict()
    
    original = Y + arsi.NDVI
    predicted = Y_pred + arsi.NDVI[:-2]
    result = pd.concat([original, predicted], axis=1)
    result.columns= ['Original','Predicted']
    rms1 = sqrt(mean_squared_error(result.Original[:-2], result.Predicted[:-2])) #0.0186
    
    X = ddf[['N_P', 'N_LST', 'N_ET']]
    Y = ddf['D_NDVI']
    X = sm.add_constant(X)
    est = sm.OLS(Y,X, missing = 'drop').fit()
    est.summary()
    Y_pred = est.predict()
    
    original = Y + arsi.NDVI
    predicted = Y_pred + arsi.NDVI[:-6]
    result = pd.concat([original, predicted], axis=1)
    result.columns= ['Original','Predicted']
    rms1 = sqrt(mean_squared_error(result.Original[:-6], result.Predicted[:-6])) #0.0186

    fig = plt.figure()
    plt.plot(arsi.DT, result.Original, 'darkblue', label = 'Original')
    plt.plot(arsi.DT, result.Predicted, 'dodgerblue', label = 'Projected')
    plt.legend(loc='upper right')
    plt.title('Arsi NDVI')
    


    
    X = ((max(arsi.NDVI) - arsi.NDVI)* arsi.P)  #for multivariate regression ddf[['N_P', 'N_LST']]
    Y = arsi.NDVI.diff().shift(-1)
    X = sm.add_constant(X) #only use when doing first run OLS then remove when fitting prediction
    est = sm.OLS(Y,X, missing = 'drop').fit()
    est.summary()
    Y_pred = est.predict()
    Y_pred2 = 0.0047*X - 0.0201
    
    
    #take delta predictions and convert to forecast
    original = Y + arsi.NDVI
    original = np.append(result.original, np.nan)
    predicted = Y_pred2 + arsi.NDVI
    result = pd.concat([original, predicted], axis=1)
    result.columns= ['Original','Predicted']
    
    fig = plt.figure()
    plt.plot(arsi.DT, result.Original, 'darkblue', label = 'Original')
    plt.plot(arsi.DT, result.Predicted, 'dodgerblue', label = 'Projected')
    plt.legend(loc='upper right')
    plt.title('Arsi NDVI')
    
    rms1 = sqrt(mean_squared_error(result.Original[:-2], result.Predicted[:-2])) #0.0186

    
    
    #create stargazer model table
    stargazer = Stargazer([est])
    HTML(stargazer.render_html())
    
    
    
    model = sm.OLS(ddf.D_NDVI, sm.add_constant(ddf.N_P)).fit()
    #predict values of Y
    Y_pred = model.predict()
    #summary table
    model.summary()
    
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.528
Model:                            OLS   Adj. R-squared:                  0.528
Method:                 Least Squares   F-statistic:                     652.1
Date:                Wed, 08 May 2019   Prob (F-statistic):           4.64e-97
Time:                        09:56:00   Log-Likelihood:                -1442.4
No. Observations:                 584   AIC:                             2889.
Df Residuals:                     582   BIC:                             2897.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.2334      0.119     35.708      0.000       4.001       4.466
NDVI         111.7800      4.377     25.536      0.000     103.183     120.377
==============================================================================
Omnibus:                      120.942   Durbin-Watson:                   1.502
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              256.260
Skew:                           1.122   Prob(JB):                     2.26e-56
Kurtosis:                       5.344   Cond. No.                         36.9
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 D_NDVI   R-squared:                       0.630
Model:                            OLS   Adj. R-squared:                  0.629
Method:                 Least Squares   F-statistic:                     491.4
Date:                Thu, 30 May 2019   Prob (F-statistic):          2.48e-125
Time:                        14:07:34   Log-Likelihood:                 1557.5
No. Observations:                 580   AIC:                            -3109.
Df Residuals:                     577   BIC:                            -3096.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0068      0.002      2.910      0.004       0.002       0.011
N_P            0.0036      0.000     19.506      0.000       0.003       0.004
N_LST         -0.0035      0.000    -12.619      0.000      -0.004      -0.003
==============================================================================
Omnibus:                       12.972   Durbin-Watson:                   0.787
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               13.462
Skew:                           0.361   Prob(JB):                      0.00119
Kurtosis:                       2.813   Cond. No.                         27.9
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

#%%
#monthly data (soil moisture)
    mdf = pd.DataFrame()
    mdf['D_NDVIm_arsi'] = arsi['NDVI.1'].diff()[1:]
    mdf['D_NDVIm_bale'] = bale['NDVI.1'].diff()[1:]
    mdf['D_NDVIm_borena'] = borena['NDVI.1'].diff()[1:]
    mdf['D_NDVIm_guji'] = guji['NDVI.1'].diff()[1:]
    mdf['D_NDVIm_westarsi'] = westarsi['NDVI.1'].diff()[1:]
    mdf = mdf.reset_index(drop=True)
    mdf['N_SM_arsi'] = (((max(arsi['NDVI.1']) - (arsi['NDVI.1']))* arsi.SM[:-1])).shift(1)
    mdf['N_SM_bale'] = (((max(bale['NDVI.1']) - (bale['NDVI.1']))* bale.SM[:-1])).shift(1)
    mdf['N_SM_borena'] = (((max(borena['NDVI.1']) - (borena['NDVI.1']))* borena.SM[:-1])).shift(1)
    mdf['N_SM_guji'] = (((max(guji['NDVI.1']) - (guji['NDVI.1']))* guji.SM[:-1])).shift(1)
    mdf['N_SM_westarsi'] = (((max(westarsi['NDVI.1']) - (westarsi['NDVI.1']))* westarsi.SM[:-1])).shift(1)
    
    
    #mask
    mask_sm_arsi = ~np.isnan(mdf.N_SM_arsi) & ~np.isnan(mdf.D_NDVIm_arsi)
    mask_sm_bale = ~np.isnan(mdf.N_SM_bale) & ~np.isnan(mdf.D_NDVIm_bale)
    mask_sm_borena = ~np.isnan(mdf.N_SM_borena) & ~np.isnan(mdf.D_NDVIm_borena)
    mask_sm_guji = ~np.isnan(mdf.N_SM_guji) & ~np.isnan(mdf.D_NDVIm_guji)
    mask_sm_westarsi = ~np.isnan(mdf.N_SM_westarsi) & ~np.isnan(mdf.D_NDVIm_westarsi)
    
    #stats.lingress (X,Y)
    slope1, intercept1, r1, p1, std1 = stats.linregress(mdf.N_SM_arsi[mask_sm_arsi], mdf.D_NDVIm_arsi[mask_sm_arsi])
    line1 = slope1*mdf.N_SM_arsi+intercept1
    print("r-squared: %f" % r1**2)
    slope2, intercept2, r2, p2, std2 = stats.linregress(mdf.N_SM_bale[mask_sm_bale], mdf.D_NDVIm_bale[mask_sm_bale])
    line2 = slope2*mdf.N_SM_bale+intercept2
    print("r-squared: %f" % r2**2)
    slope3, intercept3, r3, p3, std3 = stats.linregress(mdf.N_SM_borena[mask_sm_borena], mdf.D_NDVIm_borena[mask_sm_borena])
    line3 = slope3*mdf.N_SM_borena+intercept3
    print("r-squared: %f" % r3**2)
    slope4, intercept4, r4, p4, std4 = stats.linregress(mdf.N_SM_guji[mask_sm_guji], mdf.D_NDVIm_guji[mask_sm_guji])
    line4 = slope4*mdf.N_SM_guji+intercept4
    print("r-squared: %f" % r4**2)
    slope5, intercept5, r5, p5, std5 = stats.linregress(mdf.N_SM_westarsi[mask_sm_westarsi], mdf.D_NDVIm_westarsi[mask_sm_westarsi])
    line5 = slope5*mdf.N_SM_westarsi+intercept5
    print("r-squared: %f" % r2**2)
    
    
    #plt.scatter (Y,X)
    fig = plt.figure()
    plt.scatter(mdf.N_SM, mdf.D_NDVIm)
    

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharey =True)
    ax1.scatter(mdf.N_SM_arsi, mdf.D_NDVIm_arsi, color = 'firebrick')
    ax1.plot(mdf.N_SM_arsi, line1, color = 'k')
    ax1.text(6, -0.1, r'y = 0.022x - 0.094', fontsize=8)
    ax1.text(6, -0.15, r'$r^2$ = 0.414', fontsize=8)
    ax1.title.set_text('Arsi')
    ax2.scatter(mdf.N_SM_bale, mdf.D_NDVIm_bale, color = 'maroon')
    ax2.plot(mdf.N_SM_bale, line2, color = 'k')
    ax2.text(6, -0.1, r'y = 0.034x - 0.157', fontsize=8)
    ax2.text(6, -0.15, r'$r^2$ = 0.761', fontsize=8)
    ax2.title.set_text('Bale')
    ax3.scatter(mdf.N_SM_borena, mdf.D_NDVIm_borena, color = 'salmon')
    ax3.plot(mdf.N_SM_borena, line3, color = 'k')
    ax3.text(8, -0.1, r'y = 0.032x - 0.201', fontsize=8)
    ax3.text(8, -0.15, r'$r^2$ = 0.687', fontsize=8)
    ax3.title.set_text('Borena')
    ax4.scatter(mdf.N_SM_guji, mdf.D_NDVIm_guji, color = 'tomato')
    ax4.plot(mdf.N_SM_guji, line4, color = 'k')
    ax4.text(6, -0.1, r'y = 0.036x - 0.144', fontsize=8)
    ax4.text(6, -0.15, r'$r^2$ = 0.677', fontsize=8)
    ax4.title.set_text('Guji')
    ax5.scatter(mdf.N_SM_westarsi, mdf.D_NDVIm_westarsi, color = 'coral')
    ax5.plot(mdf.N_SM_westarsi, line5, color = 'k')
    ax5.text(5, -0.1, r'y = 0.020x - 0.067', fontsize=8)
    ax5.text(5, -0.15, r'$r^2$ = 0.761', fontsize=8)
    ax5.title.set_text('West Arsi')
    fig.text(0.5, 0.04, r'$(NDVI_{max} - NDVI_{t-1})*SM_{t-1}$', ha='center', va='center') #common xlabel (keep xy position)
    fig.text(0.06, 0.5, r'$\Delta NDVI$', ha='center', va='center', rotation='vertical') #common ylabel
    
# =============================================================================
#     fig, (ax1, ax2, ax3) = plt.subplots(3, sharey =True)
#     ax1.scatter(ddf.N_P, ddf.D_NDVI, color = 'darkcyan')
#     ax1.plot(ddf.N_P, line1, color = 'k')
#     ax1.set(xlabel = r'$(NDVI_{max} - NDVI_{t-1})*P_{t-1}$', ylabel = r'$\Delta NDVI$')
#     
# =============================================================================
    
    
    fig = plt.figure()
    sns.regplot(x= mdf.N_SM, y=mdf.D_NDVIm)
    
    #mask
    mask_sm = ~np.isnan(mdf.N_SM) & ~np.isnan(mdf.D_NDVIm)
    #r, p = stats.pearsonr(D_NDVI[mask], N_P[mask])
    
    #stats.lingress (X,Y)
    slope, intercept, r, p, std = stats.linregress(mdf.N_SM[mask_sm], mdf.D_NDVIm[mask_sm])
    print("r-squared: %f" % r**2)
    
    #sm.OLS (Y,X)
    #ordinary least squares regression fit
    model = sm.OLS(mdf.D_NDVIm, sm.add_constant(mdf.N_SM)).fit()
    #predict values of Y
    Y_pred = model.predict()
    #summary table
    model.summary()

    #%%

    #Autocorrelation and partial autocorrelation
    fig = plt.figure()
    plot_acf(arsi.NDVI, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_pacf(arsi.NDVI, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_acf(arsi.P, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_pacf(arsi.P, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_acf(arsi.ET, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_pacf(arsi.ET, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_acf(arsi.LST, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_pacf(arsi.LST, alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_acf(arsi.SM[0:195], alpha = 0.5, lags = 50)
    plt.show()
    
    fig = plt.figure()
    plot_pacf(arsi.SM[0:195], alpha = 0.5, lags = 50)
    plt.show()
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharey =True)
    plot_pacf(arsi.NDVI, ax = ax1, title = 'NDVI', alpha = 0.5, lags = 10)
    plot_pacf(arsi.P, ax = ax2, title = 'Precipitation', alpha = 0.5, lags = 10)
    plot_pacf(arsi.ET, ax = ax3, title = 'Evapotranspiration', alpha = 0.5, lags = 10)
    plot_pacf(arsi.LST, ax = ax4, title = 'Land Surface Temperature', alpha = 0.5, lags = 10)
    plot_pacf(arsi.SM[0:195], ax = ax5, title = 'Soil Moisture', alpha = 0.5, lags = 10)
    fig.suptitle('Partial Autocorrelation - Arsi')
    fig.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharey =True)
    plot_acf(arsi.NDVI, ax = ax1, title = 'NDVI', alpha = 0.5, lags = 10)
    plot_acf(arsi.P, ax = ax2, title = 'Precipitation', alpha = 0.5, lags = 10)
    plot_acf(arsi.ET, ax = ax3, title = 'Evapotranspiration', alpha = 0.5, lags = 10)
    plot_acf(arsi.LST, ax = ax4, title = 'Land Surface Temperature', alpha = 0.5, lags = 10)
    plot_acf(arsi.SM[0:195], ax = ax5, title = 'Soil Moisture', alpha = 0.5, lags = 10)
    fig.suptitle('Autocorrelation - Arsi')
    fig.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')
    
    fig, ax = plt.subplots(5, 2) #, sharex='col', sharey='row')
    ax[0,0].plot_pacf(arsi.NDVI, title = 'NDVI', alpha = 0.5, lags = 10)
    plot_pacf(arsi.P, ax = ax2, title = 'Precipitation', alpha = 0.5, lags = 10)
    plot_pacf(arsi.ET, ax = ax3, title = 'Evapotranspiration', alpha = 0.5, lags = 10)
    plot_pacf(arsi.LST, ax = ax4, title = 'Land Surface Temperature', alpha = 0.5, lags = 10)
    plot_pacf(arsi.SM[0:195], ax = ax5, title = 'Soil Moisture', alpha = 0.5, lags = 10)
    fig.suptitle('Partial Autocorrelation - Arsi')
    fig.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')
    plot_acf(arsi.NDVI, ax = ax1, title = 'NDVI', alpha = 0.5, lags = 10)
    plot_acf(arsi.P, ax = ax2, title = 'Precipitation', alpha = 0.5, lags = 10)
    plot_acf(arsi.ET, ax = ax3, title = 'Evapotranspiration', alpha = 0.5, lags = 10)
    plot_acf(arsi.LST, ax = ax4, title = 'Land Surface Temperature', alpha = 0.5, lags = 10)
    plot_acf(arsi.SM[0:195], ax = ax5, title = 'Soil Moisture', alpha = 0.5, lags = 10)
    fig.suptitle('Autocorrelation - Arsi')
    fig.text(0.06, 0.5, 'Correlation', ha='center', va='center', rotation='vertical')
    
    
    
#%% 3. FFT

    #To get plots in same format as MATLAB output
    
    # FFT to calculate power spectra of X and Y
    fft1024 = pd.DataFrame()
    fft512 = pd.DataFrame({'F': np.linspace(0,0.5,512)})
    
    for var in ['NDVI','P']:
        # Standardize time series (subtract mean)
        z = var + '_STD'
        arsi[z] = arsi[var] - arsi[var].mean() 
        # Compute FFT (raw)
        xxx = var + '_FFT'
        fft1024[xxx] = fft(arsi[z],1024)
        # Square absolute value and divide by # of sample points
        pxx2 = var + '_SQN'
        fft1024[pxx2] = (abs(fft1024[xxx])**2)/792
        # Select first 512 values
        pxx1 = var + '_POW'
        fft512[pxx1] = fft1024[pxx2][:512] * 2
        fft512[pxx1][0] = fft512[pxx1][0]/2
        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.plot(fft512.F,fft512[pxx1])
        ax1.set_xlabel('Frequency (1/dekad)')
        ax1.set_ylabel('Power')
        ax1.set_xlim(right = 0.1)
        
        # Plot period vs. power 
        ax2.plot(1/fft512.F,fft512[pxx1])
        ax2.set_xlabel('Period (dekads)')
        ax2.set_ylabel('Power')
        ax2.set_xlim(right = 10)



#%% 2. PERIODOGRAM
 
    

    # Standardize X (subtract mean)
    arsi['NDVI_STD'] = arsi.NDVI - arsi.NDVI.mean()
    arsi['P_STD'] = arsi.P - arsi.P.mean()
    arsi['ET_STD'] = arsi.ET - arsi.ET.mean()
    arsi['LST_STD'] = arsi.LST - arsi.LST.mean()
    
       
    # Plot X vs. time  #redo all time series with this standardization
    fig = plt.figure()
    plt.plot(arsi.DT, arsi.NDVI_STD)
    plt.xlabel('Time')
    plt.ylabel('X')

    # Use periodogram to compute Fourier transform
    f, pxx = signal.periodogram(arsi.NDVI, fs=1, nfft=1024, scaling='spectrum')
    
    #NDVI arsi periodogram
    # Plot period vs. power
    fig = plt.figure()
    plt.plot(1/f,pxx*792)
    plt.xlabel('Period (months)')
    plt.ylabel('Power')
    plt.xlim(0, 40)
    plt.title('NDVI Arsi Periodogram')
    #frequency vs.power
# =============================================================================
#     ax1 = fig.add_subplot(1,2,1)
#     ax2 = fig.add_subplot(1,2,2)
#     
#     ax1.plot(f,pxx*792)
#     ax1.set_xlabel('Frequency (1/month)')
#     ax1.set_ylabel('Power')        
# =============================================================================


 
    

#%%
#Wavelets
#tutorial: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 128)

[cfs, frequencies] = pywt.cwt(arsi.NDVI, scales, wavelet, 1)
power = (abs(cfs)) ** 2

period = 1. / frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
f, ax = plt.subplots(figsize=(15, 10))
cf = ax.contourf(arsi.index.values, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')

ax.set_title('Arsi NDVI Wavelet Power Spectrum, \n Fitted mother Morlet')
ax.set_ylabel('Period (dekads)')
ax.set_xlabel('Power')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                        np.ceil(np.log2(period.max())))
ax.set_yticks(np.log2(Yticks))
ax.set_yticklabels(Yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], -1)
cbar = plt.colorbar(cf)

plt.show()       
    
#%%

data_norm = waipy.normalize(arsi.NDVI)
result = waipy.cwt(data_norm, 1, 1, 0.125, 2, 4/0.125, 0.72, 6,mother='Morlet',name='X')
waipy.wavelet_plot('Wavelet Signal', arsi.index.values, data_norm, 0.03125, result)

#%%
    
    #Augmented Dickey-Fuller test
    #conclusion: all data are stationary
    
    result = adfuller(arsi.NDVI)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value)) 

    result = adfuller(arsi.ET)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value)) #
        
    result = adfuller(arsi.LST)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
    result = adfuller(arsi.SM[0:195])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    
    fig = plt.figure()
    plt.plot(arsi.DT, arsi.NDVI)
    plt.plot(arsi.DT, arsi.LST)
    
    #fft
    
    #convolution/wavelet
    
    #granger test

    #auto Arima
    
    #load the data
data = pd.read_csv('international-airline-passengers.csv')

#divide into train and validation set
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]

#preprocessing (since arima takes univariate series as input)
train.drop('Month',axis=1,inplace=True)
valid.drop('Month',axis=1,inplace=True)

#plotting the data
train['International airline passengers'].plot()
valid['International airline passengers'].plot()

#building the model
from pyramid.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)




#%%

#cointegration and vector autoregression
#https://nbviewer.jupyter.org/github/mapsa/seminario-doc-2014/blob/master/cointegration-example.ipynb

def get_johansen(y, p):
        """
        Get the cointegration vectors at 95% level of significance
        given by the trace statistic test.
        """

        N, l = y.shape
        jres = coint_johansen(y, 0, p)
        trstat = jres.lr1                       # trace statistic
        tsignf = jres.cvt                       # critical values

        for i in range(l):
            if trstat[i] > tsignf[i, 1]:     # 0: 90%  1:95% 2: 99%
                r = i + 1
        jres.r = r
        jres.evecr = jres.evec[:, :r]

        return jres

#%%
        
    
#%%
  #Monthly median box and whisker      

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat


series = pd.read_csv('daily-minimum-temperatures.csv', delimiter=',', header=0)
one_year = series['1990']
groups = arsi.groupby(TimeGrouper('M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()
  


#%% 

#Lag plot

from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
values = mdf.D_NDVI
lag_var = mdf.N_LST
lags = 7
main = [values]
forcing = [lag_var]
for i in range(1,(lags + 1)):
	main.append(values.shift(i))
dataframe = concat(main, axis=1)
main = ['t+1']
for i in range(1,(lags + 1)):
	forcing.append('t-' + str(i))
dataframe.forcing = forcing
pyplot.figure(1)
for i in range(1,(lags + 1)):
	ax = pyplot.subplot(240 + i)
	ax.set_title('t+1 vs t-' + str(i))
	pyplot.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].lag_var)
pyplot.show()


#%%

#Auto ARIMA = Auto-Regressive Integrated Moving Averages - 
#auto arima eliminates the need to determine optimal parameters of p and q using ACF and PACF plots by finding the optimal fit based on AIC adnd BIC values
#Arima describes the correlation betwen data points and takes into account the difference of the values. SARIMA is an improvement from ARIMA as it takes into accoun
#seasonality of trends
# =============================================================================
# In the above code, we simply used the .fit() command to fit the model without having 
# to select the combination of p, q, d. But how did the model figure out the best combination 
# of these parameters? Auto ARIMA takes into account the AIC and BIC values generated (as you can 
# see in the code) to determine the best combination of parameters. AIC (Akaike Information Criterion) 
# and BIC (Bayesian Information Criterion) values are estimators to compare models. 
# The lower these values, the better is the model.
# =============================================================================

#load the data


#divide into train and validation set
train = arsi.NDVI[:int(0.7*(len(arsi.NDVI)))]
valid = arsi.NDVI[int(0.7*(len(arsi.NDVI))):]

#preprocessing (since arima takes univariate series as input)
#train.drop('Month',axis=1,inplace=True)
#valid.drop('Month',axis=1,inplace=True)

#plotting the data
train.plot()
valid.plot()

#building the model
from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()
plt.title('Auto-ARIMA NDVI Arsi')
plt.legend(loc = 'upper right')

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)


#%%

#VAR

arsi_d = arsi.drop(['Time','DT', 'Unnamed: 5','DT.1','NDVI.1','SM'], axis = 1)
arsi_d.index = arsi.DT
cols = arsi_d.columns

#creating the train and validation set
train = arsi_d[:int(0.8*(len(arsi_d)))]
valid = arsi_d[int(0.8*(len(arsi_d))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,4):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))
    
# =============================================================================
#     rmse value for NDVI is :  0.10655367496536057
#     rmse value for P is :  22.550839652448367
#     rmse value for ET is :  6.58456060283453
#     rmse value for LST is :  5.190747160439124
# =============================================================================
    
    #make final predictions
model = VAR(endog=arsi_d)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)
#%%  

#%%

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot


#ARIMA 
NDVI = arsi.NDVI
autocorrelation_plot(NDVI)

size = int(len(NDVI) * 0.7)
train, test = NDVI[0:size], NDVI[size:len(NDVI)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,0,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#%%
    
if __name__ == "__main__":
    main()



#%%
