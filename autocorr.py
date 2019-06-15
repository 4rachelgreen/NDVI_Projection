#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Mon Apr 29 18:16:08 2019'
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
import matplotlib.pyplot as plt

import io
import requests
import zipfile

#%% FUNCTIONS

def get_value(name, gender, year):
    """Return the number of babies born a given year,
    with a given gender and a given name."""
    dy = data[year]
    try:
        return dy[dy['Gender'] == gender] \
                 ['Number'][name]
    except KeyError:
        return 0

def get_evolution(name, gender):
    """Return the evolution of a baby name over
    the years."""
    return np.array([get_value(name, gender, year)
                     for year in years])
    
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


def autocorr_name(name, gender, color, axes=None):
    x = get_evolution(name, gender)
    z = autocorr(x)

    # Evolution of the name.
    axes[0].plot(years, x, '-o' + color,
                 label=name)
    axes[0].set_title("Baby names")
    axes[0].legend()

    # Autocorrelation.
    axes[1].plot(z / float(z.max()), #Normalize by max 
                 '-' + color, label=name)
    axes[1].legend()
    axes[1].set_title("Autocorrelation")
    
    

#%% MAIN

url = ('https://github.com/ipython-books/'
       'cookbook-2nd-data/blob/master/'
       'babies.zip?raw=true')
r = io.BytesIO(requests.get(url).content)
zipfile.ZipFile(r).extractall('babies')

#%%

files = [file for file in os.listdir('babies')
         if file.startswith('yob')]

years = np.array(sorted([int(file[3:7])
                         for file in files]))

data = {year:
        pd.read_csv('babies/yob%d.txt' % year,
                    index_col=0, header=None,
                    names=['First name',
                           'Gender',
                           'Number'])
        for year in years}
    
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
autocorr_name('Olivia', 'F', 'k', axes=axes)
autocorr_name('Maria', 'F', 'y', axes=axes)


#%%%
def main():
    



#%%
if __name__ == "__main__":
    main()



#%
