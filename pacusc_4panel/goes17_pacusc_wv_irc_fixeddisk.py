#!/home/poker/miniconda3/bin/python

# This is the script that does 4 panel wv/ir

import netCDF4
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
#import os.rename
#import os.remove
import shutil
import sys
import glob
import datetime
from time import sleep
from PIL import Image

aoslogo = Image.open('/home/poker/uw-aoslogo.png')
aoslogoheight = aoslogo.size[1]
aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
aoslogo = np.array(aoslogo).astype(np.float) / 255

# set CONUS and date
prod_id = "TIRW"
#dt="201703051957"
dt = sys.argv[1]

# Start of Upper-level water vapor
#
band="08"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

# read in latest file and get time
e = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/latest.nc")
#print(e)
dtjulian = e.start_date_time[:-2]
print(dtjulian)
dt = datetime.datetime.strptime(dtjulian,'%Y%j%H%M').strftime('%Y%m%d%H%M')
print (dt)

sleep(30)

print ("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")


f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


#print(f)

data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]

#print(data_var)

x = f.variables['x'][:]
y = f.variables['y'][:]
xa[f.tile_column_offset:f.tile_column_offset+f.product_tile_width] = x[:]
ya[f.tile_row_offset:f.tile_row_offset+f.product_tile_height] = y[:]

if f.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,f.number_product_tiles):
        if os.path.isfile("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc")
            data_var2 = g.variables['Sectorized_CMI']
            a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]

# swap zeros for ones
a[a==0.] = 1.
#
#print(np.average(a))
#if np.average(a) > 0.7:
#    quit()

xa=xa*35.785831
ya=ya*35.785831


proj_var = f.variables[data_var.grid_mapping]

import cartopy.crs as ccrs

### OLD ###
# Create a Globe specifying a spherical earth with the correct radius
#globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
#
#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)
globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
proj = ccrs.Geostationary(central_longitude=-137.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns

# West Coast Close-up Crop
westcoast_image_crop_top=0
westcoast_image_crop_bottom=-750
westcoast_image_crop_left=1200
westcoast_image_crop_right=-50

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
westcoast_image_size_x=float(westcoast_image_size_x)/40.
westcoast_image_size_y=float(westcoast_image_size_y)/40.

# Same stuff for Hawaii crop
hi_image_crop_top=900
hi_image_crop_bottom=-1
hi_image_crop_left=0
hi_image_crop_right=-1900


hi_image_size_y=(image_rows+hi_image_crop_bottom-hi_image_crop_top)
hi_image_size_x=(image_columns+hi_image_crop_right-hi_image_crop_left)

print("hi image size")
print(hi_image_size_x, hi_image_size_y)

hi_image_size_x=float(hi_image_size_x)/75.
hi_image_size_y=float(hi_image_size_y)/75.

ne_image_crop_top=25
ne_image_crop_bottom=-725
ne_image_crop_left=1275
ne_image_crop_right=-125

ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)

print("ne image size")
print(ne_image_size_x, ne_image_size_y)

ne_image_size_x=float(ne_image_size_x)/100.
ne_image_size_y=float(ne_image_size_y)/100.

conus_image_crop_top=0
conus_image_crop_bottom=0
conus_image_crop_left=0
conus_image_crop_right=0

conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)

print("conus image size")
print(conus_image_size_x, conus_image_size_y)

conus_image_size_x=float(conus_image_size_x)/150.
conus_image_size_y=float(conus_image_size_y)/150.

gulf_image_crop_top=428
gulf_image_crop_bottom=-5
gulf_image_crop_left=649
gulf_image_crop_right=-76

gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)

print("gulf image size")
print(gulf_image_size_x, gulf_image_size_y)

gulf_image_size_x=float(gulf_image_size_x)/120.
gulf_image_size_y=float(gulf_image_size_y)/120.

sw_image_crop_top=220
sw_image_crop_bottom=-600
sw_image_crop_left=1500
sw_image_crop_right=-1


sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

print("sw image size")
print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/10.
sw_image_size_y=float(sw_image_size_y)/10.

nw_image_crop_top=0
nw_image_crop_bottom=-1100
nw_image_crop_left=1500
nw_image_crop_right=-101

nw_image_size_y=(image_rows+nw_image_crop_bottom-nw_image_crop_top)
nw_image_size_x=(image_columns+nw_image_crop_right-nw_image_crop_left)

print("nw image size")
print(nw_image_size_x, nw_image_size_y)

nw_image_size_x=float(sw_image_size_x)/10.
nw_image_size_y=float(sw_image_size_y)/10.



# Create a new figure with size 10" by 10"
# West Coast Crop
fig = plt.figure(figsize=(15.0,10.599))
# Hawaii crop
fig2 = plt.figure(figsize=(10.522,12.379))
# CONUS
fig3 = plt.figure(figsize=(15.0,10.599))
# Southwest US
fig13 = plt.figure(figsize=(13.440,10.831))
# Northwest US
fig14 = plt.figure(figsize=(15.916,8.489))


# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 1, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 1, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 1, projection=proj)
ax3.outline_patch.set_edgecolor('none')
#ax4 = fig4.add_subplot(2, 2, 1, projection=proj, xmargin=0, ymargin=0, xbound=0, ybound=0, frame_on="True")
#ax4.outline_patch.set_edgecolor('none')
#ax8 = fig8.add_subplot(2, 2, 1, projection=proj, xmargin=0, ymargin=0, xbound=0, ybound=0, frame_on="True")
#ax8.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(2, 2, 1, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(2, 2, 1, projection=proj)
ax14.outline_patch.set_edgecolor('none')
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig13.tight_layout()
fig14.tight_layout()
#fig4.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig2.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig3.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig13.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig14.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
#ax9 = fig9.add_subplot(2, 2, 1, projection=proj)


cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

import matplotlib as mpl

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,2048)

cdict2 = {'red': ((0.0, 0.0, 0.0),
                 (0.208, 0.0, 0.0),
                 (0.379, 1.0, 1.0),
                 (0.483, 0.0, 0.0),
                 (0.572, 1.0, 1.0),
                 (0.667, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.208, .423, .423),
                   (0.379, 1.0, 1.0),
                   (0.483, 0.0, 0.0),
                   (0.572, 1.0, 1.0),
                   (0.667, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.208, 0.0, 0.0),
                  (0.379, 1.0, 1.0),
                  (0.483,0.651, 0.651),
                  (0.572, 0.0, 0.0),
                  (0.667, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

cdict3 = {'red': ((0.0, 0.0, 0.0),
                 (0.290, 0.263, .263),
                 (0.385, 1.0, 1.0),
                 (0.475, 0.443, .443),
                 (0.515, 0.0, 0.0),
                 (0.575, 1.0, 1.0),
                 (0.664, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.290, .513, .513),
                   (0.385, 1.0, 1.0),
                   (0.475, .443, .443),
                   (0.515, 0., 0.0),
                   (0.575, 1.0, 1.0),
                   (0.664, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.290, .137, .137),
                  (0.385, 1.0, 1.0),
                  (0.475,0.694, 0.694),
                  (0.515, .451, .451),
                  (0.552, 0.0, 0.0),
                  (0.664, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

my_cmap2 = mpl.colors.LinearSegmentedColormap('my_colormap2',cdict3,2048)

# Plot the data using a simple greyscale colormap (with black for low values);
# set the colormap to extend over a range of values from 140 to 255.
# Note, we save the image returned by imshow for later...
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#               cmap='Greys_r', norm=plt.Normalize(0, 256))
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#im = ax.imshow(a[:], extent=(xa[0], xa[-1], ya[-1], ya[0]), origin='upper',
#               cmap='Greys_r')
#im = ax.imshow(a[250:-3000,2000:-2000], extent=(xa[2000],xa[-2000],ya[-3000],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(a[250:-3500,2500:-2200], extent=(xa[2500],xa[-2200],ya[-3500],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper')
#im = ax2.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap='Greys_r')

im = ax.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat

ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Upper-Level WV (ABI 8) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"

if f.satellite_id == "GOES-17":
    time_string = 'GOES-17 Upper-Level WV (ABI 8)'
elif f.satellite_id == "GOES-18":
    time_string = 'GOES-18 Upper-Level WV (ABI 8)'
else:
    time_string = 'GOES-West Upper-Level WV (ABI 8)'
print(time_string)

#time_string = 'GOES-17 Upper-Level WV (ABI 8)'
#print(time_string)

from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

#2017/065 20:04:00:30
text = ax.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text.set_path_effects(outline_effect)

text2 = ax2.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text2.set_path_effects(outline_effect)

text3 = ax3.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='yellow', fontsize='large', weight='bold')

text3.set_path_effects(outline_effect)

text13 = ax13.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text13.set_path_effects(outline_effect)

text14 = ax14.text(0.50, 0.93, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text14.set_path_effects(outline_effect)


# START MID-LEVEL WV

band="09"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']


#f = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


#print(f)

data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

#print(data_var)

x = f.variables['x'][:]
y = f.variables['y'][:]
xa[f.tile_column_offset:f.tile_column_offset+f.product_tile_width] = x[:]
ya[f.tile_row_offset:f.tile_row_offset+f.product_tile_height] = y[:]

if f.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,f.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


#a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
#print(a)

# swap zeros for ones
a[a==0.] = 1.
#
#print(np.average(a))
#if np.average(a) > 0.7:
#    quit()

xa=xa*35.785831
ya=ya*35.785831


proj_var = f.variables[data_var.grid_mapping]

import cartopy.crs as ccrs

### OLD ###
# Create a Globe specifying a spherical earth with the correct radius
#globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
#
#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)
globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
proj = ccrs.Geostationary(central_longitude=-137.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns

# START CROP 2

# West Coast Close-up Crop
westcoast_image_crop_top=0
westcoast_image_crop_bottom=-750
westcoast_image_crop_left=1200
westcoast_image_crop_right=-50

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
westcoast_image_size_x=float(westcoast_image_size_x)/40.
westcoast_image_size_y=float(westcoast_image_size_y)/40.

# Same stuff for Hawaii crop
hi_image_crop_top=900
hi_image_crop_bottom=-1
hi_image_crop_left=0
hi_image_crop_right=-1900


hi_image_size_y=(image_rows+hi_image_crop_bottom-hi_image_crop_top)
hi_image_size_x=(image_columns+hi_image_crop_right-hi_image_crop_left)

print("hi image size")
print(hi_image_size_x, hi_image_size_y)

hi_image_size_x=float(hi_image_size_x)/75.
hi_image_size_y=float(hi_image_size_y)/75.

ne_image_crop_top=25
ne_image_crop_bottom=-725
ne_image_crop_left=1275
ne_image_crop_right=-125

ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)

print("ne image size")
print(ne_image_size_x, ne_image_size_y)

ne_image_size_x=float(ne_image_size_x)/100.
ne_image_size_y=float(ne_image_size_y)/100.

conus_image_crop_top=0
conus_image_crop_bottom=0
conus_image_crop_left=0
conus_image_crop_right=0

conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)

print("conus image size")
print(conus_image_size_x, conus_image_size_y)

conus_image_size_x=float(conus_image_size_x)/150.
conus_image_size_y=float(conus_image_size_y)/150.

gulf_image_crop_top=428
gulf_image_crop_bottom=-5
gulf_image_crop_left=649
gulf_image_crop_right=-76

gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)

print("gulf image size")
print(gulf_image_size_x, gulf_image_size_y)

gulf_image_size_x=float(gulf_image_size_x)/120.
gulf_image_size_y=float(gulf_image_size_y)/120.

sw_image_crop_top=220
sw_image_crop_bottom=-600
sw_image_crop_left=1500
sw_image_crop_right=-1


sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

print("sw image size")
print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/10.
sw_image_size_y=float(sw_image_size_y)/10.

nw_image_crop_top=0
nw_image_crop_bottom=-1100
nw_image_crop_left=1500
nw_image_crop_right=-101

nw_image_size_y=(image_rows+nw_image_crop_bottom-nw_image_crop_top)
nw_image_size_x=(image_columns+nw_image_crop_right-nw_image_crop_left)

print("nw image size")
print(nw_image_size_x, nw_image_size_y)

nw_image_size_x=float(sw_image_size_x)/10.
nw_image_size_y=float(sw_image_size_y)/10.

# END CROP 2

# NO need to create new figure here - we are adding to one above
# Create a new figure with size 10" by 10"
#fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=80.)
#fig2 = plt.figure(figsize=(mw_image_size_x,mw_image_size_y),dpi=160.)
#fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=160.)
#fig4 = plt.figure(figsize=(ne_image_size_x,ne_image_size_y),dpi=160.)
#fig2 = plt.figure(figsize=(image_columns/200.,image_rows/200.))
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 2, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 2, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 2, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(2, 2, 2, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(2, 2, 2, projection=proj)
ax14.outline_patch.set_edgecolor('none')
#ax9 = fig9.add_subplot(1, 1, 1, projection=proj)


cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

import matplotlib as mpl

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,2048)

cdict2 = {'red': ((0.0, 0.0, 0.0),
                 (0.208, 0.0, 0.0),
                 (0.379, 1.0, 1.0),
                 (0.483, 0.0, 0.0),
                 (0.572, 1.0, 1.0),
                 (0.667, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.208, .423, .423),
                   (0.379, 1.0, 1.0),
                   (0.483, 0.0, 0.0),
                   (0.572, 1.0, 1.0),
                   (0.667, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.208, 0.0, 0.0),
                  (0.379, 1.0, 1.0),
                  (0.483,0.651, 0.651),
                  (0.572, 0.0, 0.0),
                  (0.667, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

cdict3 = {'red': ((0.0, 0.0, 0.0),
                 (0.290, 0.263, .263),
                 (0.385, 1.0, 1.0),
                 (0.475, 0.443, .443),
                 (0.515, 0.0, 0.0),
                 (0.575, 1.0, 1.0),
                 (0.664, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.290, .513, .513),
                   (0.385, 1.0, 1.0),
                   (0.475, .443, .443),
                   (0.515, 0., 0.0),
                   (0.575, 1.0, 1.0),
                   (0.664, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.290, .137, .137),
                  (0.385, 1.0, 1.0),
                  (0.475,0.694, 0.694),
                  (0.515, .451, .451),
                  (0.552, 0.0, 0.0),
                  (0.664, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}


my_cmap2 = mpl.colors.LinearSegmentedColormap('my_colormap2',cdict3,2048)

# Plot the data using a simple greyscale colormap (with black for low values);
# set the colormap to extend over a range of values from 140 to 255.
# Note, we save the image returned by imshow for later...

im = ax.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Mid-Level WV (ABI 9) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"

if f.satellite_id == "GOES-17":
    time_string = 'GOES-17 Mid-Level WV (ABI 9)'
elif f.satellite_id == "GOES-18":
    time_string = 'GOES-18 Mid-Level WV (ABI 9)'
else:
    time_string = 'GOES-West Mid-Level WV (ABI 9)'
print(time_string)

#time_string = 'GOES-17 Mid-Level WV (ABI 9)'
#print(time_string)

#2017/065 20:04:00:30
text21 = ax.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text21.set_path_effects(outline_effect)

text22 = ax2.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text22.set_path_effects(outline_effect)

text23 = ax3.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='yellow', fontsize='large', weight='bold')

text23.set_path_effects(outline_effect)

text213 = ax13.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text213.set_path_effects(outline_effect)

text214 = ax14.text(0.50, 0.93, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text214.set_path_effects(outline_effect)


# START LOW-LEVEL WV

band="10"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']


#f = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


#print(f)

data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

#print(data_var)

x = f.variables['x'][:]
y = f.variables['y'][:]
xa[f.tile_column_offset:f.tile_column_offset+f.product_tile_width] = x[:]
ya[f.tile_row_offset:f.tile_row_offset+f.product_tile_height] = y[:]

if f.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,f.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


#a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
#print(a)

# swap zeros for ones
a[a==0.] = 1.
#
#print(np.average(a))
#if np.average(a) > 0.7:
#    quit()

xa=xa*35.785831
ya=ya*35.785831


proj_var = f.variables[data_var.grid_mapping]

import cartopy.crs as ccrs

### OLD ###
# Create a Globe specifying a spherical earth with the correct radius
#globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
#
#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)
globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
proj = ccrs.Geostationary(central_longitude=-137.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns

# START CROP 3
# West Coast Close-up Crop
westcoast_image_crop_top=0
westcoast_image_crop_bottom=-750
westcoast_image_crop_left=1200
westcoast_image_crop_right=-50

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
westcoast_image_size_x=float(westcoast_image_size_x)/40.
westcoast_image_size_y=float(westcoast_image_size_y)/40.

# Same stuff for Hawaii crop
hi_image_crop_top=900
hi_image_crop_bottom=-1
hi_image_crop_left=0
hi_image_crop_right=-1900


hi_image_size_y=(image_rows+hi_image_crop_bottom-hi_image_crop_top)
hi_image_size_x=(image_columns+hi_image_crop_right-hi_image_crop_left)

print("hi image size")
print(hi_image_size_x, hi_image_size_y)

hi_image_size_x=float(hi_image_size_x)/75.
hi_image_size_y=float(hi_image_size_y)/75.

ne_image_crop_top=25
ne_image_crop_bottom=-725
ne_image_crop_left=1275
ne_image_crop_right=-125

ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)

print("ne image size")
print(ne_image_size_x, ne_image_size_y)

ne_image_size_x=float(ne_image_size_x)/100.
ne_image_size_y=float(ne_image_size_y)/100.

conus_image_crop_top=0
conus_image_crop_bottom=0
conus_image_crop_left=0
conus_image_crop_right=0

conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)

print("conus image size")
print(conus_image_size_x, conus_image_size_y)

conus_image_size_x=float(conus_image_size_x)/150.
conus_image_size_y=float(conus_image_size_y)/150.

gulf_image_crop_top=428
gulf_image_crop_bottom=-5
gulf_image_crop_left=649
gulf_image_crop_right=-76

gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)

print("gulf image size")
print(gulf_image_size_x, gulf_image_size_y)

gulf_image_size_x=float(gulf_image_size_x)/120.
gulf_image_size_y=float(gulf_image_size_y)/120.

sw_image_crop_top=220
sw_image_crop_bottom=-600
sw_image_crop_left=1500
sw_image_crop_right=-1


sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

print("sw image size")
print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/10.
sw_image_size_y=float(sw_image_size_y)/10.

nw_image_crop_top=0
nw_image_crop_bottom=-1100
nw_image_crop_left=1500
nw_image_crop_right=-101

nw_image_size_y=(image_rows+nw_image_crop_bottom-nw_image_crop_top)
nw_image_size_x=(image_columns+nw_image_crop_right-nw_image_crop_left)

print("nw image size")
print(nw_image_size_x, nw_image_size_y)

nw_image_size_x=float(sw_image_size_x)/10.
nw_image_size_y=float(sw_image_size_y)/10.

#END CROP

# NO need to create new figure here - we are adding to one above
# Create a new figure with size 10" by 10"
#fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=80.)
#fig2 = plt.figure(figsize=(mw_image_size_x,mw_image_size_y),dpi=160.)
#fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=160.)
#fig4 = plt.figure(figsize=(ne_image_size_x,ne_image_size_y),dpi=160.)
#fig2 = plt.figure(figsize=(image_columns/200.,image_rows/200.))
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 3, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 3, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 3, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(2, 2, 3, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(2, 2, 3, projection=proj)
ax14.outline_patch.set_edgecolor('none')
#ax9 = fig9.add_subplot(1, 1, 1, projection=proj)


cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

import matplotlib as mpl

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,2048)

cdict2 = {'red': ((0.0, 0.0, 0.0),
                 (0.208, 0.0, 0.0),
                 (0.379, 1.0, 1.0),
                 (0.483, 0.0, 0.0),
                 (0.572, 1.0, 1.0),
                 (0.667, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.208, .423, .423),
                   (0.379, 1.0, 1.0),
                   (0.483, 0.0, 0.0),
                   (0.572, 1.0, 1.0),
                   (0.667, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.208, 0.0, 0.0),
                  (0.379, 1.0, 1.0),
                  (0.483,0.651, 0.651),
                  (0.572, 0.0, 0.0),
                  (0.667, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

cdict3 = {'red': ((0.0, 0.0, 0.0),
                 (0.290, 0.263, .263),
                 (0.385, 1.0, 1.0),
                 (0.475, 0.443, .443),
                 (0.515, 0.0, 0.0),
                 (0.575, 1.0, 1.0),
                 (0.664, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.290, .513, .513),
                   (0.385, 1.0, 1.0),
                   (0.475, .443, .443),
                   (0.515, 0., 0.0),
                   (0.575, 1.0, 1.0),
                   (0.664, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.290, .137, .137),
                  (0.385, 1.0, 1.0),
                  (0.475,0.694, 0.694),
                  (0.515, .451, .451),
                  (0.552, 0.0, 0.0),
                  (0.664, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}


my_cmap2 = mpl.colors.LinearSegmentedColormap('my_colormap2',cdict3,2048)

# Plot the data using a simple greyscale colormap (with black for low values);
# set the colormap to extend over a range of values from 140 to 255.
# Note, we save the image returned by imshow for later...

im = ax.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

# axes for westcoast
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
cbaxes1 = fig.add_axes([0.09,0.12,0.395,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')

# axes for hi
#cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
cbaxes2 = fig2.add_axes([0.09,0.12,0.395,0.02])
cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
font_size = 14
#cbar2.set_label('Brightness Temperature (K)',size=18)
cbar2.ax.tick_params(labelsize=font_size)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

# axes for conus
cbaxes3 = fig3.add_axes([0.09,0.12,0.395,0.02])
cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar3.ax.tick_params(labelsize=font_size)
cbar3.ax.xaxis.set_ticks_position('top')
cbar3.ax.xaxis.set_label_position('top')

# axes for sw
cbaxes13 = fig13.add_axes([0.09,0.12,0.395,0.02])
cbar13 = fig13.colorbar(im, cax=cbaxes13, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar13.ax.tick_params(labelsize=font_size)
cbar13.ax.xaxis.set_ticks_position('top')
cbar13.ax.xaxis.set_label_position('top')

# axes for nw
cbaxes14 = fig14.add_axes([0.09,0.12,0.395,0.02])
cbar14 = fig14.colorbar(im, cax=cbaxes14, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar14.ax.tick_params(labelsize=font_size)
cbar14.ax.xaxis.set_ticks_position('top')
cbar14.ax.xaxis.set_label_position('top')


import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Low-Level WV (ABI 10) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"

if f.satellite_id == "GOES-17":
    time_string = 'GOES-17 Low-Level WV (ABI 10)'
elif f.satellite_id == "GOES-18":
    time_string = 'GOES-18 Low-Level WV (ABI 10)'
else:
    time_string = 'GOES-West Low-Level WV (ABI 10)'
print(time_string)

#time_string = 'GOES-17 Low-Level WV (ABI 10)'
#print(time_string)

#2017/065 20:04:00:30
text31 = ax.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text31.set_path_effects(outline_effect)

text32 = ax2.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text32.set_path_effects(outline_effect)

text33 = ax3.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='yellow', fontsize='large', weight='bold')

text33.set_path_effects(outline_effect)

text313 = ax13.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text313.set_path_effects(outline_effect)

text314 = ax14.text(0.50, 0.93, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text314.set_path_effects(outline_effect)

# START IR 

band="13"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

# read in first tile, create numpy arrays big enough to hold entire tile data, x and y coords (a, xa, ya)
#f = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


# print out some metadata for info. This can be commented out
#print(f)

# read in brightness values/temps from first sector, copy to the whole image array
data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]


# read in sector x, y coordinate variables, copy to whole image x,y variables

x = f.variables['x'][:]
y = f.variables['y'][:]
xa[f.tile_column_offset:f.tile_column_offset+f.product_tile_width] = x[:]
ya[f.tile_row_offset:f.tile_row_offset+f.product_tile_height] = y[:]

# if there are more than one tile in this image
if f.number_product_tiles > 1:
# read in the data, x coord and y coord values from each sector, plug into a, xa and ya arrays
# this goes from 1 to number of tiles - 1
    for i in range(1,f.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRW/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


#a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
#
# not sure I need this next line, might be redundant but I don't want to delete it right now because the script is working
    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
#
# debug print statement
#print(a)

#################################
# This is an ugly hack that originated with the channel 2 visible data - I noticed 
# early on that pixels at the top of convective clouds were coming through black. I
# think this is a calibration issue that the pixels are brighter than the software or
# sensor thinks they should be. They therefore get coded with a 'zero' missing value. 
# Unfortunately, dark (night) pixels are also coded with the same zero missing value.
# I made the executive decision that I'd prefer to have those missing pixels in the
# cloud tops show up white, at the expense of night pixels also being white. I suspect
# as they get closer to operational this will get fixed and this fix won't be necessary
# anymore. One side effect is that since I initialize the a array to all zeros, if a
# sector is missing, that entire block will be white instead of black. 
#################################
#
# swap zeros for ones
a[a==0.] = 1.
#

# This if statement also originated from the visible script that I copied 
# over to become this IR one.
#
# For my loops, I don't want to insert a boatload of all white night visible images, so
# I check the average of all elements in the a array. If that average is > 0.7 (values for
# visible are reflectance values from 0. to 1. - IR values are brightness temperature 
# values) then I assume that most of the image is missing or night values, and therefore
# bail out and don't process it.
#
# 
#print(np.average(a))
#if np.average(a) > 0.7:
#    quit()

xa=xa*35.785831
ya=ya*35.785831


# get projection info from the metadata
proj_var = f.variables[data_var.grid_mapping]

# Set up mapping using cartopy - this is different for full disk imagery vs 
# the CONUS Lamert Conformal
import cartopy.crs as ccrs

### OLD ###
# Create a Globe specifying a spherical earth with the correct radius
#globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
#
#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)
globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
proj = ccrs.Geostationary(central_longitude=-137.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')



image_rows=f.product_rows
image_columns=f.product_columns

# Here I start setting crop values so I can plot multiple images from this one
# image pasted together from the multiple sector netcdf files
#

# Number of pixels to be cropped from the top, bottom, left and right of each 
# image. Bottom and Right are negative numbers (using python's negative array
# indexing - -1 is the last element in the array dimension)
#
# START CROP 4
# West Coast Close-up Crop
westcoast_image_crop_top=0
westcoast_image_crop_bottom=-750
westcoast_image_crop_left=1200
westcoast_image_crop_right=-50

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
westcoast_image_size_x=float(westcoast_image_size_x)/40.
westcoast_image_size_y=float(westcoast_image_size_y)/40.

# Same stuff for Hawaii crop
hi_image_crop_top=900
hi_image_crop_bottom=-1
hi_image_crop_left=0
hi_image_crop_right=-1900


hi_image_size_y=(image_rows+hi_image_crop_bottom-hi_image_crop_top)
hi_image_size_x=(image_columns+hi_image_crop_right-hi_image_crop_left)

print("hi image size")
print(hi_image_size_x, hi_image_size_y)

hi_image_size_x=float(hi_image_size_x)/75.
hi_image_size_y=float(hi_image_size_y)/75.

ne_image_crop_top=25
ne_image_crop_bottom=-725
ne_image_crop_left=1275
ne_image_crop_right=-125

ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)

print("ne image size")
print(ne_image_size_x, ne_image_size_y)

ne_image_size_x=float(ne_image_size_x)/100.
ne_image_size_y=float(ne_image_size_y)/100.

conus_image_crop_top=0
conus_image_crop_bottom=0
conus_image_crop_left=0
conus_image_crop_right=0

conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)

print("conus image size")
print(conus_image_size_x, conus_image_size_y)

conus_image_size_x=float(conus_image_size_x)/150.
conus_image_size_y=float(conus_image_size_y)/150.

gulf_image_crop_top=428
gulf_image_crop_bottom=-5
gulf_image_crop_left=649
gulf_image_crop_right=-76

gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)

print("gulf image size")
print(gulf_image_size_x, gulf_image_size_y)

gulf_image_size_x=float(gulf_image_size_x)/120.
gulf_image_size_y=float(gulf_image_size_y)/120.

sw_image_crop_top=220
sw_image_crop_bottom=-600
sw_image_crop_left=1500
sw_image_crop_right=-1


sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

print("sw image size")
print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/10.
sw_image_size_y=float(sw_image_size_y)/10.

nw_image_crop_top=0
nw_image_crop_bottom=-1100
nw_image_crop_left=1500
nw_image_crop_right=-101

nw_image_size_y=(image_rows+nw_image_crop_bottom-nw_image_crop_top)
nw_image_size_x=(image_columns+nw_image_crop_right-nw_image_crop_left)

print("nw image size")
print(nw_image_size_x, nw_image_size_y)

nw_image_size_x=float(sw_image_size_x)/10.
nw_image_size_y=float(sw_image_size_y)/10.

#END

# These create the figure objects for the Wisconsin (fig), Midwest (fig2), 
# CONUS (fig3) and full resolution (fig9) images. The fig9 is not used
# for any loops, and browsers will scale it down but you can click to zoom
# to get the full resolution in all its beauty.
#

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 4, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 4, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 4, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(2, 2, 4, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(2, 2, 4, projection=proj)
ax14.outline_patch.set_edgecolor('none')
#ax9 = fig9.add_subplot(2, 2, 4, projection=proj)

cdict = {'red': ((0.0, 0.1, 0.1),
                 (.052, 0.07, 0.07),
                 (.055, 0.004, 0.004),
                 (.113, 0.004, 0.004),
                 (.116, 0.85, 0.85),
                 (.162, 0.02, 0.2),
                 (0.165, 0.0, 0.0),
                 (0.229, 0.047, 0.047),
                 (0.232, 0.0, 0.0),
                 (0.297, 0.0, 0.0),
                 (0.30, 0.55, 0.55),
                 (0.355, 0.95, 0.95),
                 (0.358, 0.93, 0.93),
                 (0.416, 0.565, 0.565),
                 (0.419, .373, .373),
                 (0.483, 0.97, 0.97),
                 (0.485, 0.98, 0.98),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (.052, 0.0, 0.0),
                   (.055, 0.0, 0.0),
                   (.113, 0.0, 0.0),
                   (.116, 0.85, 0.85),
                   (.162, 0.0, 0.0),
                   (0.165, .435, .435),
                   (0.229, .97, .97),
                   (0.232, 0.37, 0.37),
                   (0.297, 0.78, 0.78),
                   (0.30, 0.0, 0.0),
                   (0.355, 0.0, 0.0),
                   (0.358, 0.0, 0.0),
                   (0.416, 0.0, 0.0),
                   (0.419, .357, .357),
                   (0.483, 0.95, 0.95),
                   (0.485, 0.98, 0.98),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.04, 0.04),
                  (.052, 0.467, 0.467),
                  (.055, 0.4, 0.4),
                  (.113, 0.97, 0.97),
                  (.116, 0.85, 0.85),
                  (.162, 0.0, 0.0),
                  (0.165, 0.0, 0.0),
                  (0.229, 0.0, 0.0),
                  (0.232,0.816, 0.816),
                  (0.297, 0.565, 0.565),
                  (0.30, .55, .55),
                  (0.355, .97, .97),
                  (0.358, 0.0, 0.0),
                  (0.416, 0., 0.),
                  (0.419, 0., 0.),
                  (0.483, 0., 0.),
                  (0.486, 0.98, 0.98),
                  (1.0, 0.0, 0.0))}

import matplotlib as mpl
my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,2048)


# Plot the data 
# set the colormap to extend over a range of values from 162 to 330 using the
# Greys built-in color map.
# Note, we save the image returned by imshow for later...

im = ax.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax2.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

# axes for westcoast
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
cbaxes1 = fig.add_axes([0.515,0.12,0.395,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')

# axes for hi
#cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
cbaxes2 = fig2.add_axes([0.515,0.12,0.395,0.02])
cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
font_size = 14
#cbar2.set_label('Brightness Temperature (K)',size=18)
cbar2.ax.tick_params(labelsize=font_size)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

## axes for pacus
cbaxes3 = fig3.add_axes([0.515,0.12,0.395,0.02])
#cbaxes3.get_xaxis().set_visible(False)
#cbaxes3.get_yaxis().set_visible(False)
#cbaxes3.axis('off')
cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar3.ax.tick_params(labelsize=font_size)
cbar3.ax.xaxis.set_ticks_position('top')
cbar3.ax.xaxis.set_label_position('top')

## axes for sw
cbaxes13 = fig13.add_axes([0.515,0.12,0.395,0.02])
cbar13 = fig13.colorbar(im, cax=cbaxes13, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar13.ax.tick_params(labelsize=font_size)
cbar13.ax.xaxis.set_ticks_position('top')
cbar13.ax.xaxis.set_label_position('top')

## axes for nw
cbaxes14 = fig14.add_axes([0.515,0.12,0.395,0.02])
cbar14 = fig14.colorbar(im, cax=cbaxes14, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar14.ax.tick_params(labelsize=font_size)
cbar14.ax.xaxis.set_ticks_position('top')
cbar14.ax.xaxis.set_label_position('top')

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 IR (ABI 14) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"

if f.satellite_id == "GOES-17":
    time_string = 'GOES-17 IR (ABI 13)'
elif f.satellite_id == "GOES-18":
    time_string = 'GOES-18 IR (ABI 13)'
else:
    time_string = 'GOES-West IR (ABI 13)'
print(time_string)

#time_string = 'GOES-17 IR (ABI 14)'
#print(time_string)

#2017/065 20:04:00:30
text41 = ax.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text41.set_path_effects(outline_effect)

text42 = ax2.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text42.set_path_effects(outline_effect)

text43 = ax3.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='yellow', fontsize='large', weight='bold')

text43.set_path_effects(outline_effect)

text413 = ax13.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text413.set_path_effects(outline_effect)

text414 = ax14.text(0.50, 0.93, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text414.set_path_effects(outline_effect)

time_string = 'Valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
print(time_string)


# put the time label on the bottom center
txtaxes1 = fig.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes1.axis('off')
txt1 = txtaxes1.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt1.set_path_effects(outline_effect)

txtaxes2 = fig2.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes2.axis('off')
txt2 = txtaxes2.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt2.set_path_effects(outline_effect)

txtaxes3 = fig3.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes3.axis('off')
txt3 = txtaxes3.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt3.set_path_effects(outline_effect)

txtaxes13 = fig13.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes13.axis('off')
txt13 = txtaxes13.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt4.set_path_effects(outline_effect)

txtaxes14 = fig14.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes14.axis('off')
txt14 = txtaxes14.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt8.set_path_effects(outline_effect)


filename1="/whirlwind/goes17/4panel/westcoast/"+dt+"_westcoast.jpg"
filename2="/whirlwind/goes17/4panel/hi/"+dt+"_hi.jpg"
filename3="/whirlwind/goes17/4panel/pacus/"+dt+"_pacus.jpg"
filename13="/whirlwind/goes17/4panel/sw/"+dt+"_sw.jpg"
filename14="/whirlwind/goes17/4panel/nw/"+dt+"_nw.jpg"
#filename9="/whirlwind/goes16/4panel/full/"+dt+"_full.jpg"

fig.figimage(aoslogo,  0, 0  , zorder=10)
fig2.figimage(aoslogo,  0, 0  , zorder=10)
fig3.figimage(aoslogo,  0, 0  , zorder=10)
fig13.figimage(aoslogo,  0, 0  , zorder=10)
fig14.figimage(aoslogo,  0, 0  , zorder=10)

print("filename1 ",filename1)
fig.savefig(filename1, bbox_inches='tight', pad_inches=0)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)
#fig2.savefig(filename2jpg, bbox_inches='tight')
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
fig13.savefig(filename13, bbox_inches='tight', pad_inches=0)
fig14.savefig(filename14, bbox_inches='tight', pad_inches=0)
#fig9.savefig(filename9, bbox_inches='tight')


# quit()

#import os.rename    # os.rename(src,dest)
#import os.remove    # os.remove path
#import shutil.copy  # shutil.copy(src, dest)

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def silentrename(filename1, filename2):
    try:
        os.rename(filename1, filename2)
    except OSError:
        pass

shutil.copy(filename1, "/whirlwind/goes17/4panel/westcoast/latest_westcoast_1.jpg")

# West Coast
file_list = glob.glob('/whirlwind/goes17/4panel/westcoast/2*.jpg')
file_list.sort()
#print("file list is ",file_list)

# 6h // 24 (36) images
thefile = open('/whirlwind/goes17/4panel/westcoast_24_temp.list', 'w')
thelist = file_list[-36:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/westcoast_24_temp.list','/whirlwind/goes17/4panel/goes17_westcoast_loop_3h.list')

# 12h // 60 (72)images
thefile = open('/whirlwind/goes17/4panel/westcoast_60_temp.list', 'w')
thelist = file_list[-72:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/westcoast_60_temp.list','/whirlwind/goes17/4panel/goes17_westcoast_loop.list')


#silentremove("/whirlwind/goes16/4panel/wi/latest_wi_72.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_71.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_72.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_70.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_71.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_69.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_70.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_68.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_69.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_67.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_68.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_66.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_67.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_65.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_66.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_64.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_65.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_63.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_64.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_62.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_63.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_61.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_62.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_60.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_61.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_59.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_60.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_58.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_59.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_57.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_58.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_56.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_57.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_55.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_56.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_54.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_55.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_53.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_54.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_52.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_53.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_51.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_52.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_50.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_51.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_49.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_50.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_48.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_49.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_47.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_48.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_46.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_47.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_45.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_46.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_44.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_45.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_43.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_44.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_42.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_43.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_41.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_42.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_40.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_41.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_39.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_40.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_38.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_39.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_37.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_38.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_36.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_37.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_35.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_36.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_34.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_35.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_33.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_34.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_32.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_33.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_31.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_32.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_30.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_31.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_29.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_30.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_28.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_29.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_27.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_28.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_26.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_27.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_25.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_26.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_24.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_25.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_23.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_24.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_22.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_23.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_21.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_22.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_20.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_21.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_19.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_20.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_18.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_19.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_17.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_18.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_16.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_17.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_15.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_16.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_14.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_15.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_13.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_14.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_12.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_13.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_11.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_12.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_10.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_11.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_9.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_10.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_8.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_9.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_7.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_8.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_6.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_7.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_5.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_6.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_4.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_5.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_3.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_4.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_2.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_3.jpg")
#silentrename("/whirlwind/goes16/4panel/wi/latest_wi_1.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_2.jpg")



# Hawaii 

shutil.copy(filename2, "/whirlwind/goes17/4panel/hi/latest_hi_1.jpg")

file_list = glob.glob('/whirlwind/goes17/4panel/hi/2*.jpg')
file_list.sort()
#print("file list is ",file_list)

# 6h // 24 (36) images
thefile = open('/whirlwind/goes17/4panel/hi_24_temp.list', 'w')
thelist = file_list[-36:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/hi_24_temp.list','/whirlwind/goes17/4panel/goes17_hi_loop_3h.list')

# 12h // 60 (72)images
thefile = open('/whirlwind/goes17/4panel/hi_60_temp.list', 'w')
thelist = file_list[-72:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/hi_60_temp.list','/whirlwind/goes17/4panel/goes17_hi_loop.list')


# PACUS 

shutil.copy(filename3, "/whirlwind/goes17/4panel/pacus/latest_pacus_1.jpg")

file_list = glob.glob('/whirlwind/goes17/4panel/pacus/2*.jpg')
file_list.sort()
#print("file list is ",file_list)

# 6h // 24 (36) images
thefile = open('/whirlwind/goes17/4panel/pacus_24_temp.list', 'w')
thelist = file_list[-36:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/pacus_24_temp.list','/whirlwind/goes17/4panel/goes17_pacus_loop_3h.list')

# 12h // 60 (72)images
thefile = open('/whirlwind/goes17/4panel/pacus_60_temp.list', 'w')
thelist = file_list[-72:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/pacus_60_temp.list','/whirlwind/goes17/4panel/goes17_pacus_loop.list')

# Southwest 

shutil.copy(filename13, "/whirlwind/goes17/4panel/sw/latest_sw_1.jpg")

file_list = glob.glob('/whirlwind/goes17/4panel/sw/2*.jpg')
file_list.sort()
#print("file list is ",file_list)

# 6h // 24 (36) images
thefile = open('/whirlwind/goes17/4panel/sw_24_temp.list', 'w')
thelist = file_list[-36:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/sw_24_temp.list','/whirlwind/goes17/4panel/goes17_sw_loop_3h.list')

# 12h // 60 (72)images
thefile = open('/whirlwind/goes17/4panel/sw_60_temp.list', 'w')
thelist = file_list[-72:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/sw_60_temp.list','/whirlwind/goes17/4panel/goes17_sw_loop.list')

# Northwest 

shutil.copy(filename14, "/whirlwind/goes17/4panel/nw/latest_nw_1.jpg")

file_list = glob.glob('/whirlwind/goes17/4panel/nw/2*.jpg')
file_list.sort()
#print("file list is ",file_list)

# 6h // 24 (36) images
thefile = open('/whirlwind/goes17/4panel/nw_24_temp.list', 'w')
thelist = file_list[-36:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/nw_24_temp.list','/whirlwind/goes17/4panel/goes17_nw_loop_3h.list')

# 12h // 60 (72)images
thefile = open('/whirlwind/goes17/4panel/nw_60_temp.list', 'w')
thelist = file_list[-72:]
#print ("thelist is ",thelist)

for item in thelist:
    head, tail = os.path.split(item)
    head, mid = os.path.split(head)
    thefile.write(mid + '/' + tail + '\n')
thefile.close
os.rename('/whirlwind/goes17/4panel/nw_60_temp.list','/whirlwind/goes17/4panel/goes17_nw_loop.list')


