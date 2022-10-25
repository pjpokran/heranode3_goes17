#!/home/poker/miniconda3/envs/goes16_201710/bin/python

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
prod_id = "TIRE"
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
        if os.path.isfile("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc")
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
proj = ccrs.Geostationary(central_longitude=-75.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns
wi_image_crop_top=48
wi_image_crop_bottom=-1153
wi_image_crop_left=1040
wi_image_crop_right=-990

wi_image_size_y=(image_rows+wi_image_crop_bottom-wi_image_crop_top)
wi_image_size_x=(image_columns+wi_image_crop_right-wi_image_crop_left)

print("wi image size")
print(wi_image_size_x, wi_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
wi_image_size_x=float(wi_image_size_x)/40.
wi_image_size_y=float(wi_image_size_y)/40.

mw_image_crop_top=23
mw_image_crop_bottom=-953
mw_image_crop_left=775
mw_image_crop_right=-978

mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)

print("mw image size")
print(mw_image_size_x, mw_image_size_y)

mw_image_size_x=float(mw_image_size_x)/75.
mw_image_size_y=float(mw_image_size_y)/75.

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


# Create a new figure with size 10" by 10"
fig = plt.figure(figsize=(15.,9.9))
fig2 = plt.figure(figsize=(15.,10.9))
fig3 = plt.figure(figsize=(15.,9.4))
fig4 = plt.figure(figsize=(15.,10.6))
fig8 = plt.figure(figsize=(15.,9.4))
#fig2 = plt.figure(figsize=(image_columns/200.,image_rows/200.))
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 1, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 1, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 1, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax4 = fig4.add_subplot(2, 2, 1, projection=proj, xmargin=0, ymargin=0, xbound=0, ybound=0, frame_on="True")
ax4.outline_patch.set_edgecolor('none')
ax8 = fig8.add_subplot(2, 2, 1, projection=proj, xmargin=0, ymargin=0, xbound=0, ybound=0, frame_on="True")
ax8.outline_patch.set_edgecolor('none')
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig8.tight_layout()
#fig4.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig2.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig3.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig4.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
fig8.subplots_adjust(bottom=0.17, left=0.08, right=0.92,  wspace=0.01, hspace=0.01)
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

im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat

ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax4.coastlines(resolution='50m', color='green')
ax8.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax4.add_feature(state_boundaries, linestyle=':')
ax8.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

## axes for wi
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
#cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
#font_size = 14
##cbar1.set_label('Brightness Temperature (K)',size=18)
#cbar1.ax.tick_params(labelsize=font_size)
#cbar1.ax.xaxis.set_ticks_position('top')
#cbar1.ax.xaxis.set_label_position('top')

## axes for mw
#cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
#cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
#font_size = 14
##cbar2.set_label('Brightness Temperature (K)',size=18)
#cbar2.ax.tick_params(labelsize=font_size)
#cbar2.ax.xaxis.set_ticks_position('top')
#cbar2.ax.xaxis.set_label_position('top')

## axes for conus
#cbaxes3 = fig3.add_axes([0.135,0.12,0.755,0.02])
#cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
#font_size = 14
##cbar3.set_label('Brightness Temperature (K)',size=18)
#cbar3.ax.tick_params(labelsize=font_size)
#cbar3.ax.xaxis.set_ticks_position('top')
#cbar3.ax.xaxis.set_label_position('top')

## axes for ne
#cbaxes4 = fig4.add_axes([0.135,0.12,0.755,0.02])
#cbar4 = fig4.colorbar(im, cax=cbaxes4, orientation='horizontal')
#font_size = 14
##cbar4.set_label('Brightness Temperature (K)',size=18)
#cbar4.ax.tick_params(labelsize=font_size)
#cbar4.ax.xaxis.set_ticks_position('top')
#cbar4.ax.xaxis.set_label_position('top')

## axes for full
#cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
#cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
#font_size = 18
##cbar9.set_label('Brightness Temperature (K)',size=18)
#cbar9.ax.tick_params(labelsize=font_size)
#cbar9.ax.xaxis.set_ticks_position('top')
#cbar9.ax.xaxis.set_label_position('top')


time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Upper-Level WV (ABI 8) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
time_string = 'GOES-16 Upper-Level WV (ABI 8)'
print(time_string)
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

text4 = ax4.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax4.transAxes,
    color='yellow', fontsize='large', weight='bold')

text4.set_path_effects(outline_effect)

text8 = ax8.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax8.transAxes,
    color='yellow', fontsize='large', weight='bold')

text8.set_path_effects(outline_effect)

#text = ax9.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax9.transAxes,
#    color='black', fontsize='large', weight='bold')



# START MID-LEVEL WV

band="09"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']


#f = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_PAA.nc")
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
        if os.path.isfile("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc")
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
proj = ccrs.Geostationary(central_longitude=-75.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns
wi_image_crop_top=48
wi_image_crop_bottom=-1153
wi_image_crop_left=1040
wi_image_crop_right=-990

wi_image_size_y=(image_rows+wi_image_crop_bottom-wi_image_crop_top)
wi_image_size_x=(image_columns+wi_image_crop_right-wi_image_crop_left)

print("wi image size")
print(wi_image_size_x, wi_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
wi_image_size_x=float(wi_image_size_x)/40.
wi_image_size_y=float(wi_image_size_y)/40.

mw_image_crop_top=23
mw_image_crop_bottom=-953
mw_image_crop_left=775
mw_image_crop_right=-978

mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)

print("mw image size")
print(mw_image_size_x, mw_image_size_y)

mw_image_size_x=float(mw_image_size_x)/75.
mw_image_size_y=float(mw_image_size_y)/75.

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

gulf_image_size_x=float(gulf_image_size_x)/60.
gulf_image_size_y=float(gulf_image_size_y)/60.


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
ax4 = fig4.add_subplot(2, 2, 2, projection=proj)
ax4.outline_patch.set_edgecolor('none')
ax8 = fig8.add_subplot(2, 2, 2, projection=proj)
ax8.outline_patch.set_edgecolor('none')
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
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#               cmap='Greys_r', norm=plt.Normalize(0, 256))
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#im = ax.imshow(a[:], extent=(xa[0], xa[-1], ya[-1], ya[0]), origin='upper',
#               cmap='Greys_r')
#im = ax.imshow(a[250:-3000,2000:-2000], extent=(xa[2000],xa[-2000],ya[-3000],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(a[250:-3500,2500:-2200], extent=(xa[2500],xa[-2200],ya[-3500],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper')
#im = ax2.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap='Greys_r')

im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax4.coastlines(resolution='50m', color='green')
ax8.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax4.add_feature(state_boundaries, linestyle=':')
ax8.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

## axes for wi
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
#cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
#font_size = 14
##cbar1.set_label('Brightness Temperature (K)',size=18)
#cbar1.ax.tick_params(labelsize=font_size)
#cbar1.ax.xaxis.set_ticks_position('top')
#cbar1.ax.xaxis.set_label_position('top')

## axes for mw
#cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
#cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
#font_size = 14
##cbar2.set_label('Brightness Temperature (K)',size=18)
#cbar2.ax.tick_params(labelsize=font_size)
#cbar2.ax.xaxis.set_ticks_position('top')
#cbar2.ax.xaxis.set_label_position('top')

## axes for conus
#cbaxes3 = fig3.add_axes([0.135,0.12,0.755,0.02])
#cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
#font_size = 14
##cbar3.set_label('Brightness Temperature (K)',size=18)
#cbar3.ax.tick_params(labelsize=font_size)
#cbar3.ax.xaxis.set_ticks_position('top')
#cbar3.ax.xaxis.set_label_position('top')

## axes for ne
#cbaxes4 = fig4.add_axes([0.135,0.12,0.755,0.02])
#cbar4 = fig4.colorbar(im, cax=cbaxes4, orientation='horizontal')
#font_size = 14
##cbar4.set_label('Brightness Temperature (K)',size=18)
#cbar4.ax.tick_params(labelsize=font_size)
#cbar4.ax.xaxis.set_ticks_position('top')
#cbar4.ax.xaxis.set_label_position('top')

## axes for full
#cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
#cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
#font_size = 18
##cbar9.set_label('Brightness Temperature (K)',size=18)
#cbar9.ax.tick_params(labelsize=font_size)
#cbar9.ax.xaxis.set_ticks_position('top')
#cbar9.ax.xaxis.set_label_position('top')


# Redisplay modified figure
#fig
#fig2

import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Mid-Level WV (ABI 9) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
time_string = 'GOES-16 Mid-Level WV (ABI 9)'
print(time_string)

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

text24 = ax4.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax4.transAxes,
    color='yellow', fontsize='large', weight='bold')

text24.set_path_effects(outline_effect)

text28 = ax8.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax8.transAxes,
    color='yellow', fontsize='large', weight='bold')

text28.set_path_effects(outline_effect)

#text = ax9.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax9.transAxes,
#    color='black', fontsize='large', weight='bold')


# START LOW-LEVEL WV

band="10"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']


#f = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_PAA.nc")
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
        if os.path.isfile("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc")
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
proj = ccrs.Geostationary(central_longitude=-75.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns
wi_image_crop_top=48
wi_image_crop_bottom=-1153
wi_image_crop_left=1040
wi_image_crop_right=-990

wi_image_size_y=(image_rows+wi_image_crop_bottom-wi_image_crop_top)
wi_image_size_x=(image_columns+wi_image_crop_right-wi_image_crop_left)

print("wi image size")
print(wi_image_size_x, wi_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
wi_image_size_x=float(wi_image_size_x)/40.
wi_image_size_y=float(wi_image_size_y)/40.

mw_image_crop_top=23
mw_image_crop_bottom=-953
mw_image_crop_left=775
mw_image_crop_right=-978

mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)

print("mw image size")
print(mw_image_size_x, mw_image_size_y)

mw_image_size_x=float(mw_image_size_x)/75.
mw_image_size_y=float(mw_image_size_y)/75.

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

gulf_image_size_x=float(gulf_image_size_x)/60.
gulf_image_size_y=float(gulf_image_size_y)/60.


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
ax4 = fig4.add_subplot(2, 2, 3, projection=proj)
ax4.outline_patch.set_edgecolor('none')
ax8 = fig8.add_subplot(2, 2, 3, projection=proj)
ax8.outline_patch.set_edgecolor('none')
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
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#               cmap='Greys_r', norm=plt.Normalize(0, 256))
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#im = ax.imshow(a[:], extent=(xa[0], xa[-1], ya[-1], ya[0]), origin='upper',
#               cmap='Greys_r')
#im = ax.imshow(a[250:-3000,2000:-2000], extent=(xa[2000],xa[-2000],ya[-3000],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(a[250:-3500,2500:-2200], extent=(xa[2500],xa[-2200],ya[-3500],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper')
#im = ax2.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap='Greys_r')

im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap2, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap2, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax4.coastlines(resolution='50m', color='green')
ax8.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax4.add_feature(state_boundaries, linestyle=':')
ax8.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

# axes for wi
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
cbaxes1 = fig.add_axes([0.09,0.12,0.395,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')

# axes for mw
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

# axes for ne
cbaxes4 = fig4.add_axes([0.09,0.12,0.395,0.02])
cbar4 = fig4.colorbar(im, cax=cbaxes4, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar4.ax.tick_params(labelsize=font_size)
cbar4.ax.xaxis.set_ticks_position('top')
cbar4.ax.xaxis.set_label_position('top')

# axes for gulf
cbaxes8 = fig8.add_axes([0.09,0.12,0.395,0.02])
cbar8 = fig8.colorbar(im, cax=cbaxes8, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar8.ax.tick_params(labelsize=font_size)
cbar8.ax.xaxis.set_ticks_position('top')
cbar8.ax.xaxis.set_label_position('top')


## axes for full
#cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
#cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
#font_size = 18
##cbar9.set_label('Brightness Temperature (K)',size=18)
#cbar9.ax.tick_params(labelsize=font_size)
#cbar9.ax.xaxis.set_ticks_position('top')
#cbar9.ax.xaxis.set_label_position('top')


# Redisplay modified figure
#fig
#fig2

#fig4.tight_layout()
#fig4.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 Low-Level WV (ABI 10) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
time_string = 'GOES-16 Low-Level WV (ABI 10)'
print(time_string)

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

text34 = ax4.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax4.transAxes,
    color='yellow', fontsize='large', weight='bold')

text34.set_path_effects(outline_effect)

text38 = ax8.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax8.transAxes,
    color='yellow', fontsize='large', weight='bold')

text38.set_path_effects(outline_effect)

#text = ax9.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax9.transAxes,
#    color='black', fontsize='large', weight='bold')

# START IR 

band="14"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

# read in first tile, create numpy arrays big enough to hold entire tile data, x and y coords (a, xa, ya)
#f = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_PAA.nc")
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
        if os.path.isfile("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_P"+filechar[i]+".nc")
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
proj = ccrs.Geostationary(central_longitude=-75.0,
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
# Wisconsin close-up crop
wi_image_crop_top=48
wi_image_crop_bottom=-1153
wi_image_crop_left=1040
wi_image_crop_right=-990

wi_image_size_y=(image_rows+wi_image_crop_bottom-wi_image_crop_top)
wi_image_size_x=(image_columns+wi_image_crop_right-wi_image_crop_left)

print("wi image size")
print(wi_image_size_x, wi_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.

# This determines the size of the picture, and based on how many elements
# in the array, the resolution of each pixel. I experimented to come up 
# with 80 for wi, 150 for mw, etc.. You can change those to see how they
# affect the final image.
#
wi_image_size_x=float(wi_image_size_x)/40.
wi_image_size_y=float(wi_image_size_y)/40.

# Same stuff for Midwest crop
mw_image_crop_top=23
mw_image_crop_bottom=-953
mw_image_crop_left=775
mw_image_crop_right=-978

mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)

print("mw image size")
print(mw_image_size_x, mw_image_size_y)

mw_image_size_x=float(mw_image_size_x)/75.
mw_image_size_y=float(mw_image_size_y)/75.

# Same stuff for Northeast crop
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

# Same stuff for CONUS crop - basically there is some white empty space
# around the edges, especially at the bottom. I tried to crop that off
# and create an image that would fit on an average screen.
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

gulf_image_size_x=float(gulf_image_size_x)/60.
gulf_image_size_y=float(gulf_image_size_y)/60.


# These create the figure objects for the Wisconsin (fig), Midwest (fig2), 
# CONUS (fig3) and full resolution (fig9) images. The fig9 is not used
# for any loops, and browsers will scale it down but you can click to zoom
# to get the full resolution in all its beauty.
#
#fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=80.)
#fig2 = plt.figure(figsize=(mw_image_size_x,mw_image_size_y),dpi=160.)
#fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=160.)
#fig4 = plt.figure(figsize=(ne_image_size_x,ne_image_size_y),dpi=160.)
#fig2 = plt.figure(figsize=(image_columns/200.,image_rows/200.))
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(2, 2, 4, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(2, 2, 4, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(2, 2, 4, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax4 = fig4.add_subplot(2, 2, 4, projection=proj)
ax4.outline_patch.set_edgecolor('none')
ax8 = fig8.add_subplot(2, 2, 4, projection=proj)
ax8.outline_patch.set_edgecolor('none')
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

im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.0)

import cartopy.feature as cfeat
ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax4.coastlines(resolution='50m', color='green')
ax8.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax4.add_feature(state_boundaries, linestyle=':')
ax8.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')

# axes for wi
#cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
cbaxes1 = fig.add_axes([0.515,0.12,0.395,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')

# axes for mw
#cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
cbaxes2 = fig2.add_axes([0.515,0.12,0.395,0.02])
cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
font_size = 14
#cbar2.set_label('Brightness Temperature (K)',size=18)
cbar2.ax.tick_params(labelsize=font_size)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

## axes for conus
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

## axes for ne
cbaxes4 = fig4.add_axes([0.515,0.12,0.395,0.02])
cbar4 = fig4.colorbar(im, cax=cbaxes4, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar4.ax.tick_params(labelsize=font_size)
cbar4.ax.xaxis.set_ticks_position('top')
cbar4.ax.xaxis.set_label_position('top')

## axes for gulf
cbaxes8 = fig8.add_axes([0.515,0.12,0.395,0.02])
cbar8 = fig8.colorbar(im, cax=cbaxes8, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar8.ax.tick_params(labelsize=font_size)
cbar8.ax.xaxis.set_ticks_position('top')
cbar8.ax.xaxis.set_label_position('top')

## axes for full
#cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
#cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
#font_size = 18
##cbar9.set_label('Brightness Temperature (K)',size=18)
#cbar9.ax.tick_params(labelsize=font_size)
#cbar9.ax.xaxis.set_ticks_position('top')
#cbar9.ax.xaxis.set_label_position('top')

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

#time_string = 'GOES-16 IR (ABI 14) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
time_string = 'GOES-16 IR (ABI 14)'
print(time_string)

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

text44 = ax4.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax4.transAxes,
    color='yellow', fontsize='large', weight='bold')

text44.set_path_effects(outline_effect)

text48 = ax8.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax8.transAxes,
    color='yellow', fontsize='large', weight='bold')

text48.set_path_effects(outline_effect)

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

txtaxes4 = fig4.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes4.axis('off')
txt4 = txtaxes4.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt4.set_path_effects(outline_effect)

txtaxes8 = fig8.add_axes([0.135,0.07,0.755,0.02], visible='false')
txtaxes8.axis('off')
txt8 = txtaxes8.text(0.50, 0.50, time_string,
    horizontalalignment='center', 
    color='black', fontsize='large', weight='bold')

#txt8.set_path_effects(outline_effect)


filename1="/whirlwind/goes16/4panel/wi/"+dt+"_wi.jpg"
filename2="/whirlwind/goes16/4panel/mw/"+dt+"_mw.jpg"
filename3="/whirlwind/goes16/4panel/conus/"+dt+"_conus.jpg"
filename4="/whirlwind/goes16/4panel/ne/"+dt+"_ne.jpg"
filename8="/whirlwind/goes16/4panel/gulf/"+dt+"_gulf.jpg"
#filename9="/whirlwind/goes16/4panel/full/"+dt+"_full.jpg"

#filename1="wvh/"+dt+"_wi.jpg"
#filename2="wvh/"+dt+"_mw.jpg"
#filename3="wvh/"+dt+"_conus.jpg"
#filename4="wvh/"+dt+"_ne.jpg"
#filename9="wvh/"+dt+"_full.jpg"

fig.figimage(aoslogo,  0, 0  , zorder=10)
fig2.figimage(aoslogo,  0, 0  , zorder=10)
fig3.figimage(aoslogo,  0, 0  , zorder=10)
fig4.figimage(aoslogo,  0, 0  , zorder=10)
fig8.figimage(aoslogo,  0, 0  , zorder=10)

fig.savefig(filename1, bbox_inches='tight', pad_inches=0)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)
#fig2.savefig(filename2jpg, bbox_inches='tight')
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
fig4.savefig(filename4, bbox_inches='tight', pad_inches=0)
fig8.savefig(filename8, bbox_inches='tight', pad_inches=0)
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


silentremove("/whirlwind/goes16/4panel/wi/latest_wi_72.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_71.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_72.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_70.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_71.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_69.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_70.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_68.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_69.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_67.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_68.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_66.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_67.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_65.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_66.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_64.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_65.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_63.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_64.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_62.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_63.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_61.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_62.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_60.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_61.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_59.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_60.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_58.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_59.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_57.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_58.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_56.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_57.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_55.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_56.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_54.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_55.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_53.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_54.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_52.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_53.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_51.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_52.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_50.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_51.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_49.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_50.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_48.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_49.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_47.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_48.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_46.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_47.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_45.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_46.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_44.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_45.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_43.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_44.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_42.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_43.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_41.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_42.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_40.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_41.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_39.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_40.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_38.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_39.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_37.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_38.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_36.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_37.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_35.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_36.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_34.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_35.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_33.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_34.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_32.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_33.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_31.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_32.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_30.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_31.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_29.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_30.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_28.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_29.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_27.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_28.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_26.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_27.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_25.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_26.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_24.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_25.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_23.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_24.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_22.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_23.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_21.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_22.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_20.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_21.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_19.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_20.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_18.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_19.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_17.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_18.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_16.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_17.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_15.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_16.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_14.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_15.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_13.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_14.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_12.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_13.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_11.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_12.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_10.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_11.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_9.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_10.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_8.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_9.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_7.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_8.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_6.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_7.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_5.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_6.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_4.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_5.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_3.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_4.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_2.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_3.jpg")
silentrename("/whirlwind/goes16/4panel/wi/latest_wi_1.jpg", "/whirlwind/goes16/4panel/wi/latest_wi_2.jpg")

shutil.copy(filename1, "/whirlwind/goes16/4panel/wi/latest_wi_1.jpg")


silentremove("/whirlwind/goes16/4panel/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_71.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_70.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_71.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_69.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_70.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_68.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_69.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_67.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_68.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_66.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_67.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_65.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_66.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_64.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_65.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_63.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_64.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_62.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_63.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_61.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_62.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_60.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_61.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_59.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_60.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_58.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_59.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_57.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_58.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_56.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_57.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_55.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_56.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_54.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_55.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_53.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_54.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_52.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_53.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_51.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_52.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_50.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_51.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_49.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_50.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_48.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_49.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_47.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_48.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_46.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_47.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_45.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_46.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_44.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_45.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_43.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_44.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_42.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_43.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_41.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_42.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_40.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_41.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_39.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_40.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_38.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_39.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_37.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_38.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_36.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_37.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_35.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_36.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_34.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_35.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_33.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_34.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_32.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_33.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_31.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_32.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_30.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_31.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_29.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_30.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_28.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_29.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_27.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_28.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_26.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_27.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_25.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_26.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_24.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_25.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_23.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_24.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_22.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_23.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_21.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_22.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_20.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_21.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_19.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_20.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_18.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_19.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_17.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_18.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_16.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_17.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_15.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_16.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_14.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_15.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_13.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_14.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_12.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_13.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_11.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_12.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_10.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_11.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_9.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_10.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_8.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_9.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_7.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_8.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_6.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_7.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_5.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_6.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_4.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_5.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_3.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_4.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_2.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_3.jpg")
silentrename("/whirlwind/goes16/4panel/mw/latest_mw_1.jpg", "/whirlwind/goes16/4panel/mw/latest_mw_2.jpg")

shutil.copy(filename2, "/whirlwind/goes16/4panel/mw/latest_mw_1.jpg")


silentremove("/whirlwind/goes16/4panel/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_71.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_70.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_71.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_69.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_70.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_68.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_69.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_67.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_68.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_66.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_67.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_65.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_66.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_64.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_65.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_63.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_64.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_62.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_63.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_61.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_62.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_60.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_61.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_59.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_60.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_58.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_59.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_57.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_58.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_56.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_57.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_55.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_56.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_54.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_55.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_53.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_54.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_52.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_53.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_51.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_52.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_50.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_51.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_49.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_50.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_48.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_49.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_47.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_48.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_46.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_47.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_45.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_46.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_44.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_45.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_43.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_44.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_42.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_43.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_41.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_42.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_40.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_41.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_39.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_40.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_38.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_39.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_37.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_38.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_36.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_37.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_35.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_36.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_34.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_35.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_33.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_34.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_32.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_33.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_31.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_32.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_30.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_31.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_29.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_30.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_28.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_29.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_27.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_28.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_26.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_27.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_25.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_26.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_24.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_25.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_23.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_24.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_22.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_23.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_21.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_22.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_20.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_21.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_19.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_20.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_18.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_19.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_17.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_18.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_16.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_17.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_15.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_16.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_14.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_15.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_13.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_14.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_12.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_13.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_11.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_12.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_10.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_11.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_9.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_10.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_8.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_9.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_7.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_8.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_6.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_7.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_5.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_6.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_4.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_5.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_3.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_4.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_2.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_3.jpg")
silentrename("/whirlwind/goes16/4panel/ne/latest_ne_1.jpg", "/whirlwind/goes16/4panel/ne/latest_ne_2.jpg")

shutil.copy(filename4, "/whirlwind/goes16/4panel/ne/latest_ne_1.jpg")


silentremove("/whirlwind/goes16/4panel/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_71.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_70.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_71.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_69.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_70.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_68.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_69.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_67.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_68.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_66.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_67.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_65.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_66.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_64.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_65.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_63.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_64.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_62.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_63.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_61.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_62.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_60.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_61.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_59.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_60.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_58.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_59.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_57.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_58.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_56.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_57.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_55.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_56.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_54.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_55.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_53.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_54.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_52.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_53.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_51.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_52.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_50.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_51.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_49.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_50.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_48.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_49.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_47.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_48.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_46.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_47.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_45.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_46.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_44.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_45.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_43.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_44.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_42.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_43.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_41.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_42.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_40.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_41.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_39.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_40.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_38.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_39.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_37.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_38.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_36.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_37.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_35.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_36.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_34.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_35.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_33.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_34.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_32.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_33.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_31.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_32.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_30.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_31.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_29.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_30.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_28.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_29.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_27.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_28.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_26.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_27.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_25.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_26.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_24.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_25.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_23.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_24.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_22.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_23.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_21.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_22.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_20.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_21.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_19.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_20.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_18.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_19.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_17.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_18.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_16.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_17.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_15.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_16.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_14.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_15.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_13.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_14.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_12.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_13.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_11.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_12.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_10.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_11.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_9.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_10.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_8.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_9.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_7.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_8.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_6.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_7.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_5.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_6.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_4.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_5.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_3.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_4.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_2.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_3.jpg")
silentrename("/whirlwind/goes16/4panel/conus/latest_conus_1.jpg", "/whirlwind/goes16/4panel/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes16/4panel/conus/latest_conus_1.jpg")

# quit()

silentremove("/whirlwind/goes16/4panel/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_71.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_70.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_71.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_69.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_70.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_68.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_69.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_67.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_68.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_66.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_67.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_65.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_66.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_64.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_65.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_63.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_64.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_62.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_63.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_61.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_62.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_60.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_61.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_59.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_60.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_58.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_59.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_57.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_58.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_56.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_57.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_55.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_56.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_54.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_55.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_53.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_54.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_52.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_53.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_51.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_52.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_50.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_51.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_49.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_50.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_48.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_49.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_47.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_48.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_46.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_47.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_45.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_46.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_44.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_45.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_43.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_44.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_42.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_43.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_41.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_42.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_40.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_41.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_39.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_40.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_38.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_39.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_37.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_38.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_36.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_37.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_35.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_36.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_34.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_35.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_33.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_34.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_32.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_33.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_31.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_32.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_30.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_31.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_29.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_30.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_28.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_29.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_27.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_28.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_26.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_27.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_25.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_26.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_24.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_25.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_23.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_24.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_22.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_23.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_21.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_22.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_20.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_21.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_19.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_20.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_18.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_19.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_17.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_18.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_16.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_17.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_15.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_16.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_14.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_15.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_13.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_14.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_12.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_13.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_11.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_12.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_10.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_11.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_9.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_10.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_8.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_9.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_7.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_8.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_6.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_7.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_5.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_6.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_4.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_5.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_3.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_4.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_2.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_3.jpg")
silentrename("/whirlwind/goes16/4panel/gulf/latest_gulf_1.jpg", "/whirlwind/goes16/4panel/gulf/latest_gulf_2.jpg")

shutil.copy(filename8, "/whirlwind/goes16/4panel/gulf/latest_gulf_1.jpg")

