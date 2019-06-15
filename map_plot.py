#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Wed May  8 11:25:02 2019'
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


from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl

import geopandas as gpd
import fiona


%matplotlib qt5 



#%% FUNCTIONS
os.chdir('/Users/rgreen/Documents/Github/NDVI_Projection/Shapefiles')

f = fiona.open('g2008_2.shp')


fp = '/Users/rgreen/Documents/Github/NDVI_Projection/Shapefiles/g2008_2.shp'

africa_adm2 = gpd.read_file(fp)

type(africa_adm2)

ethiopia = africa_adm2[africa_adm2.ADM0_NAME == 'Ethiopia'].set_index('ADM2_NAME')
study = ethiopia.loc[['Arsi', 'Bale', 'Borena', 'Guji', 'West Arsi'], :]

study = study.to_crs(epsg=3857)


ax = study.plot()
ax.set_axis_off()


# =============================================================================
# def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
#     xmin, xmax, ymin, ymax = ax.axis()
#     basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
#     ax.imshow(basemap, extent=extent, interpolation='bilinear')
#     # restore original x/y limits
#     ax.axis((xmin, xmax, ymin, ymax))
# 
# 
# 
# 
# add_basemap(ax, zoom=11, url=ctx.sources.ST_TONER_LITE)
# =============================================================================



# =============================================================================
# study.plot(column='ADM2_NAME', cmap='RdYlBu', linewidth=0, scheme="quantiles", k=9, alpha=0.6, edgecolor='k')
# ctx.add_basemap(ax, url=ctx.tile_providers.OSM_A)
# =============================================================================

#%% MAIN
def main():
    



#%%
if __name__ == "__main__":
    main()



#%
