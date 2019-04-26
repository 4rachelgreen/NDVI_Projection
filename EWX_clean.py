#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Tue Apr 23 15:41:52 2019'
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


    import statsmodels.api as sm


    from scipy import stats
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl



    %matplotlib qt5 

#%% FUNCTIONS



#%% MAIN
def main():

#%%
    #Dekadal Data
    eth = pd.read_csv('/Users/rgreen/Downloads/export-Ethiopia+Oromia+Arsi-1-Dekad-2.csv')
    
    eth.insert(2, 'M', [str(x) for x in range(1,13) for i in range(3)])
    
    d = ['10','20'] * 12
    
    for i,x in enumerate(['31','28','31','30','31','30','31','31','30','31','30','31']):
        ind = i + 2*(i+1)
        d.insert(ind, x)
        
    eth.insert(3, 'D', d)
        
    P_arsi = pd.DataFrame(columns=['DT','NDVI'])
    
    for col in list(eth.columns)[4:-1]:
        dts =  eth.M + '/' + eth.D + '/' + col
        dt = pd.to_datetime(dts)
        dtdf = pd.concat([dt,eth[col]],axis=1).reset_index(drop=True)
        dtdf.columns = ['DT','NDVI']
        P_arsi = P_arsi.append(dtdf, ignore_index=True)

        
   #ehti.to_csv(index=False)     
        
        
    

    

#%%
   
   #Monthly Data
    eth = pd.read_csv('/Users/rgreen/Downloads/export-Ethiopia+Oromia+Arsi-1-Monthly.csv')
    
    eth.insert(2, 'M', [str(x) for x in range(1,13)])

        
        
    prec = pd.DataFrame(columns=['DT','P'])
    
    for col in list(eth.columns)[3:-1]:
        dts =  eth.M + '/1/' + col
        dt = pd.to_datetime(dts)
        dtdf = pd.concat([dt,eth[col]],axis=1).reset_index(drop=True)
        dtdf.columns = ['DT','P']
        prec = prec.append(dtdf, ignore_index=True)

        
#%%
    #to combine everything    
    westarsi_full = pd.DataFrame({'DT': NDVI_westarsi.DT, 
                              'NDVI': NDVI_westarsi.NDVI, \
                              'P': P_westarsi.NDVI, \
                              'ET': ET_westarsi.NDVI, \
                              'LST': LST_westarsi.NDVI
                             })
    
    westarsi = westarsi_full.iloc[0:585]
#%%        
    eth_full = prec.copy()
    
    eth_full['NDVI'] = ndvi.NDVI
    eth_full['SM'] = sm.SM
    
#%%    

    arsi.to_csv('arsi.csv', index=False)     
    bale.to_csv('bale.csv', index=False)     
    borena.to_csv('borema.csv', index=False)     
    guji.to_csv('guji.csv', index=False)     
    westarsi.to_csv('westarsi.csv', index=False)     
    
        
        
    


#%%

if __name__ == "__main__":
    main()



#%
