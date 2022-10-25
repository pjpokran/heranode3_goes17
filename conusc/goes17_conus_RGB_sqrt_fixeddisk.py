#!/home/poker/miniconda3/envs/goes16_201710/bin/python

import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os,errno
#import os.rename
#import os.remove
import shutil
import sys
import datetime
#from time import sleep
from scipy import interpolate
from PIL import Image

aoslogo = Image.open('/home/poker/uw-aoslogo.png')
aoslogoheight = aoslogo.size[1]
aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
aoslogo = np.array(aoslogo).astype(np.float) / 255

#def interpolate_red_channel(red_ds, resampled_ds):
#    """
#    Interpolate the red data channel to the same grid as another channel.
#    """
#    x_new = resampled_ds.variables['x'][:]
#    y_new = resampled_ds.variables['y'][::-1]
#
#    f = interpolate.interp2d(red_ds.variables['x'][:], red_ds.variables['y'][::-1],
#                             red_ds.variables['Sectorized_CMI'][::-1,], fill_value=0)
#    red_interpolated = f(x_new, y_new[::-1])
#    return x_new, y_new, red_interpolated



filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ',
          'CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM',
          'CN','CO','CP','CQ','CR','CS','CT','CU','CV','CW','CX','CY','CZ',
          'DA','DB','DC','DD','DE','DF','DG','DH','DI','DJ','DK','DL','DM',
          'DN','DO','DP','DQ','DR','DS','DT','DU','DV','DW','DX','DY','DZ',
          'EA','EB','EC','ED','EE','EF','EG','EH','EI','EJ','EK','EL','EM',
          'EN','EO','EP','EQ','ER','ES','ET','EU','EV','EW','EX','EY','EZ']

#print(filechar[1])

prod_id = "TIRW"
#dt="201703051957"
dt = sys.argv[1]
# read in latest file and get time
band = "02"
e = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/latest.nc")
#e = netCDF4.Dataset(dt)
print(e)
dtjulian = e.start_date_time[:-2]
print(dtjulian)
dt = datetime.datetime.strptime(dtjulian,'%Y%j%H%M').strftime('%Y%m%d%H%M')
print (dt)


# Start red band
band="02"
#f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
red_ds = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
red_data = np.zeros(shape=(red_ds.product_rows,red_ds.product_columns))
red_xa= np.zeros(shape=(red_ds.product_columns))
red_ya= np.zeros(shape=(red_ds.product_rows))


print("RED DS")
#print(red_ds)

data_var = red_ds.variables['Sectorized_CMI']
red_data[0:red_ds.product_tile_height,0:red_ds.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

#print(data_var)

x = red_ds.variables['x'][:]
y = red_ds.variables['y'][:]
red_xa[red_ds.tile_column_offset:red_ds.tile_column_offset+red_ds.product_tile_width] = x[:]
red_ya[red_ds.tile_row_offset:red_ds.tile_row_offset+red_ds.product_tile_height] = y[:]

if red_ds.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,red_ds.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            red_data[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            red_xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            red_ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


##a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
#    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
##print(a)

# swap zeros for ones
#red_data[red_data==0.] = 1.

# Start blue band
band="01"
#f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
blue_ds = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
blue_data = np.zeros(shape=(blue_ds.product_rows,blue_ds.product_columns))
blue_xa= np.zeros(shape=(blue_ds.product_columns))
blue_ya= np.zeros(shape=(blue_ds.product_rows))


print("BLUE DS")
#print(blue_ds)

blue_data_var = blue_ds.variables['Sectorized_CMI']
blue_data[0:blue_ds.product_tile_height,0:blue_ds.product_tile_width] = blue_data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

#print(blue_data_var)

x = blue_ds.variables['x'][:]
y = blue_ds.variables['y'][:]
blue_xa[blue_ds.tile_column_offset:blue_ds.tile_column_offset+blue_ds.product_tile_width] = x[:]
blue_ya[blue_ds.tile_row_offset:blue_ds.tile_row_offset+blue_ds.product_tile_height] = y[:]

if blue_ds.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,blue_ds.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            blue_data[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            blue_xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            blue_ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


##a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
#    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
##print(a)

# swap zeros for ones
#blue_data[blue_data==0.] = 1.

# Start veggie band
band="03"
#f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
veggie_ds = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
veggie_data = np.zeros(shape=(veggie_ds.product_rows,veggie_ds.product_columns))
veggie_xa= np.zeros(shape=(veggie_ds.product_columns))
veggie_ya= np.zeros(shape=(veggie_ds.product_rows))


print("VEGGIE DS")
#print(veggie_ds)

data_var = veggie_ds.variables['Sectorized_CMI']
veggie_data[0:veggie_ds.product_tile_height,0:veggie_ds.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

#print(data_var)

x = veggie_ds.variables['x'][:]
y = veggie_ds.variables['y'][:]
veggie_xa[veggie_ds.tile_column_offset:veggie_ds.tile_column_offset+veggie_ds.product_tile_width] = x[:]
veggie_ya[veggie_ds.tile_row_offset:veggie_ds.tile_row_offset+veggie_ds.product_tile_height] = y[:]

if red_ds.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,veggie_ds.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            veggie_data[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            veggie_xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            veggie_ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


#a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
#    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
#print(a)

# swap zeros for ones
#veggie_data[veggie_data==0.] = 1.





print("Blue Data average")
#print(np.average(blue_data))
#if np.average(blue_data) > 0.7:
#    quit()
print(np.average(blue_data))
if np.average(blue_data) < 0.0035:
    quit()


proj_var = blue_ds.variables[blue_data_var.grid_mapping]

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


image_rows=blue_ds.product_rows
image_columns=blue_ds.product_columns

import matplotlib as mpl
import cartopy.feature as cfeat
# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

state_boundaries2 = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='10m', facecolor='none', edgecolor='red')

import datetime

time_var = blue_ds.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

if red_ds.satellite_id == "GOES-17":
    time_string = 'GOES-17 Natural color visible   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
elif red_ds.satellite_id == "GOES-18":
    time_string = 'GOES-18 Natural color visible   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
else:
    time_string = 'GOES-West Natural color visible   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
print(time_string)

from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]


print("interpolate red")

x_new = blue_xa[:]
y_new = blue_ya[::-1]

fint = interpolate.interp2d(red_xa[:], red_ya[::-1],
                         red_data[::-1,], fill_value=0)
red_interpolated = fint(x_new, y_new[::-1])


# Part one of Kaba's pseudo green

print("calculate green")
green_data = (.1*veggie_data) + (.45*blue_data) + (.45*red_interpolated[::-1,:])
#green_data = (.2*veggie_data) + (.45*blue_data) + (.35*red_interpolated[::-1,:])
#green_data = (.15*veggie_data) + (.27*blue_data) + (.58*red_interpolated[::-1,:])
#green_data = (.15*veggie_data) + (.10*blue_data) + (.75*red_interpolated[::-1,:])

print("calc sqrt of red")
red_interpolated = np.sqrt(red_interpolated)
print("calc sqrt of blue")
blue_data = np.sqrt(blue_data)
print("calc sqrt of green")
green_data = np.sqrt(green_data)

# Kaba's second magic contrast part

# This may need to change when NOAAPORT files get fixed
maxValue=1.0
acont=0.1
amax=1.0067
amid=0.5
afact=(amax*(acont+maxValue)/(maxValue*(amax-acont)))
# Red part

print("Kaba part 2 red")

red_interpolated = (afact*(red_interpolated-amid)+amid)
red_interpolated[red_interpolated <= 0.0392] = 0
red_interpolated[red_interpolated >=1.0] = 1.0

# Blue part

print("Kaba part 2 blue")

blue_data = (afact*(blue_data-amid)+amid)
blue_data[blue_data <= 0.0392] = 0
blue_data[blue_data >=1.0] = 1.0

# Green part
print("Kaba part 2 green")

green_data = (afact*(green_data-amid)+amid)
green_data[green_data <= 0.0392] = 0
green_data[green_data >=1.0] = 1.0

print("stack 3 colors")
rgb_data = np.dstack([red_interpolated[::-1,:], green_data, blue_data])

blue_xa = blue_xa * 35.785831
blue_ya = blue_ya * 35.785831

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

# WestCoast Crop [no suffix]

westcoast_image_crop_top=0
westcoast_image_crop_bottom=-1500
westcoast_image_crop_left=2400
westcoast_image_crop_right=-100

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

westcoast_image_size_x=float(westcoast_image_size_x)/80.
westcoast_image_size_y=float(westcoast_image_size_y)/80.

# WI crop
fig = plt.figure(figsize=(20.,11.987))
ax = fig.add_subplot(1, 1, 1, projection=proj)
im = ax.imshow(rgb_data[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(blue_xa[westcoast_image_crop_left],blue_xa[westcoast_image_crop_right],blue_ya[westcoast_image_crop_bottom],blue_ya[westcoast_image_crop_top]), origin='upper')
ax.coastlines(resolution='50m', color='green')
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax.add_feature(state_boundaries, linestyle=':')

text = ax.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text.set_path_effects(outline_effect)

filename1="/whirlwind/goes17/vis_color/westcoast/"+dt+"_westcoast.jpg"
isize = fig.get_size_inches()*fig.dpi
ysize=int(isize[1]*0.77)
fig.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig.savefig(filename1, bbox_inches='tight', pad_inches=0)

# END WI


# START Hawaii [suffix 2]
hi_image_crop_top=1799
hi_image_crop_bottom=-1
hi_image_crop_left=0
hi_image_crop_right=-3800

hi_image_size_y=(image_rows+hi_image_crop_bottom-hi_image_crop_top)
hi_image_size_x=(image_columns+hi_image_crop_right-hi_image_crop_left)

print("hi image size")
print(hi_image_size_x, hi_image_size_y)

hi_image_size_x=float(hi_image_size_x)/150.
hi_image_size_y=float(hi_image_size_y)/150.

# Hawaii crop
fig2 = plt.figure(figsize=(14.,14.))
ax2 = fig2.add_subplot(1, 1, 1, projection=proj)

im = ax2.imshow(rgb_data[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(blue_xa[hi_image_crop_left],blue_xa[hi_image_crop_right],blue_ya[hi_image_crop_bottom],blue_ya[hi_image_crop_top]), origin='upper')
ax2.coastlines(resolution='50m', color='red')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='red')
ax2.add_feature(state_boundaries, linestyle=':')
text2 = ax2.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text2.set_path_effects(outline_effect)
filename2="/whirlwind/goes17/vis_color/hi/"+dt+"_hi.jpg"
isize = fig2.get_size_inches()*fig2.dpi
ysize=int(isize[1]*0.77)
fig2.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)


# END MW

# Start CONUS [suffix 3]
conus_image_crop_top=0
conus_image_crop_bottom=0
conus_image_crop_left=0
conus_image_crop_right=0

conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)

#print("conus image size")
#print(conus_image_size_x, conus_image_size_y)

conus_image_size_x=float(conus_image_size_x)/300.
conus_image_size_y=float(conus_image_size_y)/300.

fig3 = plt.figure(figsize=(20.,11.987))
ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
im = ax3.imshow(rgb_data[:], extent=(blue_xa[0],blue_xa[-1],blue_ya[-1],blue_ya[0]), origin='upper')
ax3.coastlines(resolution='50m', color='green')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax3.add_feature(state_boundaries, linestyle=':')
text3 = ax3.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text3.set_path_effects(outline_effect)

filename3="/whirlwind/goes17/vis_color/conus/"+dt+"_conus.jpg"
isize = fig3.get_size_inches()*fig3.dpi
ysize=int(isize[1]*0.77)
fig3.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)

# END CONUS

## Start NE [suffix 4]
## Northeast sector
#ne_image_crop_top=50
#ne_image_crop_bottom=-1450
#ne_image_crop_left=2550
#ne_image_crop_right=-250
#
#ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
#ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)
#
#ne_image_size_x=float(ne_image_size_x)/400.
#ne_image_size_y=float(ne_image_size_y)/400.
#
## Northeast crop
#fig4 = plt.figure(figsize=(18.,12.27))
#ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
#
#im = ax4.imshow(rgb_data[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(blue_xa[ne_image_crop_left],blue_xa[ne_image_crop_right],blue_ya[ne_image_crop_bottom],blue_ya[ne_image_crop_top]), origin='upper')
#
#ax4.coastlines(resolution='50m', color='green')
#ax4.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax4.add_feature(state_boundaries, linestyle=':')
#text4 = ax4.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax4.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#text4.set_path_effects(outline_effect)
#filename4="/whirlwind/goes16/vis_color/ne/"+dt+"_ne.jpg"
#isize = fig4.get_size_inches()*fig4.dpi
#ysize=int(isize[1]*0.77)
#fig4.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#fig4.savefig(filename4, bbox_inches='tight', pad_inches=0)
#
# END NE

## UNUSED IN THIS SCRIPT ###
#im = ax5.imshow(rgb_data[swi_image_crop_top:swi_image_crop_bottom,swi_image_crop_left:swi_image_crop_right], extent=(blue_xa[swi_image_crop_left],blue_xa[swi_image_crop_right],blue_ya[swi_image_crop_bottom],blue_ya[swi_image_crop_top]), origin='upper')
#im = ax6.imshow(rgb_data[co_image_crop_top:co_image_crop_bottom,co_image_crop_left:co_image_crop_right], extent=(blue_xa[co_image_crop_left],blue_xa[co_image_crop_right],blue_ya[co_image_crop_bottom],blue_ya[co_image_crop_top]), origin='upper')
#im = ax7.imshow(rgb_data[fl_image_crop_top:fl_image_crop_bottom,fl_image_crop_left:fl_image_crop_right], extent=(blue_xa[fl_image_crop_left],blue_xa[fl_image_crop_right],blue_ya[fl_image_crop_bottom],blue_ya[fl_image_crop_top]), origin='upper')
## UNUSED IN THIS SCRIPT ###

## start GULF [suffix 8]
## Gulf of Mexico sector
#
#gulf_image_crop_top=856
#gulf_image_crop_bottom=-10
#gulf_image_crop_left=1298
#gulf_image_crop_right=-152
#
#gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
#gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)
#
#gulf_image_size_x=float(gulf_image_size_x)/120.
#gulf_image_size_y=float(gulf_image_size_y)/120.
#
## Gulf of Mexico region
#fig8 = plt.figure(figsize=(30.,18.026))
#ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
#
#im = ax8.imshow(rgb_data[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(blue_xa[gulf_image_crop_left],blue_xa[gulf_image_crop_right],blue_ya[gulf_image_crop_bottom],blue_ya[gulf_image_crop_top]), origin='upper')
#ax8.coastlines(resolution='50m', color='green')
#ax8.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax8.add_feature(state_boundaries, linestyle=':')
#text8 = ax8.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax8.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text8.set_path_effects(outline_effect)
#
#filename8="/whirlwind/goes16/vis_color/gulf/"+dt+"_gulf.jpg"
#isize = fig8.get_size_inches()*fig2.dpi
#ysize=int(isize[1]*0.77)
#fig8.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#fig8.savefig(filename8, bbox_inches='tight', pad_inches=0)
## END GULF

# START Southwest [suffix 13]
sw_image_crop_top=440
sw_image_crop_bottom=-1200
sw_image_crop_left=2999
sw_image_crop_right=-1

sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

#print("sw image size")
#print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/150.
sw_image_size_y=float(sw_image_size_y)/150.

fig13 = plt.figure(figsize=(18.,12.25))
ax13 = fig13.add_subplot(1, 1, 1, projection=proj)

im = ax13.imshow(rgb_data[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(blue_xa[sw_image_crop_left],blue_xa[sw_image_crop_right],blue_ya[sw_image_crop_bottom],blue_ya[sw_image_crop_top]), origin='upper')
ax13.coastlines(resolution='50m', color='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(state_boundaries, linestyle=':')
text13 = ax13.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text13.set_path_effects(outline_effect)
filename13="/whirlwind/goes17/vis_color/sw/"+dt+"_sw.jpg"
isize = fig13.get_size_inches()*fig13.dpi
ysize=int(isize[1]*0.77)
fig13.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig13.savefig(filename13, bbox_inches='tight', pad_inches=0)


# END Southwest

# START Northwest [suffix 14]
nw_image_crop_top=0
nw_image_crop_bottom=-2200
nw_image_crop_left=3000
nw_image_crop_right=-202

nw_image_size_y=(image_rows+nw_image_crop_bottom-nw_image_crop_top)
nw_image_size_x=(image_columns+nw_image_crop_right-nw_image_crop_left)

#print("nw image size")
#print(nw_image_size_x, nw_image_size_y)

nw_image_size_x=float(nw_image_size_x)/150.
nw_image_size_y=float(nw_image_size_y)/150.

fig14 = plt.figure(figsize=(18.,8.009))
ax14 = fig14.add_subplot(1, 1, 1, projection=proj)

im = ax14.imshow(rgb_data[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(blue_xa[nw_image_crop_left],blue_xa[nw_image_crop_right],blue_ya[nw_image_crop_bottom],blue_ya[nw_image_crop_top]), origin='upper')
ax14.coastlines(resolution='50m', color='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(state_boundaries, linestyle=':')
text14 = ax14.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text14.set_path_effects(outline_effect)
filename14="/whirlwind/goes17/vis_color/nw/"+dt+"_nw.jpg"
isize = fig14.get_size_inches()*fig14.dpi
ysize=int(isize[1]*0.77)
fig14.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig14.savefig(filename14, bbox_inches='tight', pad_inches=0)


# END Northwest

## START Greatlakes [suffix 15]
#gtlakes_image_crop_top=8
#gtlakes_image_crop_bottom=-2160
#gtlakes_image_crop_left=2282
#gtlakes_image_crop_right=-1322
#
#gtlakes_image_size_y=(image_rows+gtlakes_image_crop_bottom-gtlakes_image_crop_top)
#gtlakes_image_size_x=(image_columns+gtlakes_image_crop_right-gtlakes_image_crop_left)
#
##print("gtlakes image size")
##print(gtlakes_image_size_x, gtlakes_image_size_y)
#
#gtlakes_image_size_x=float(gtlakes_image_size_x)/150.
#gtlakes_image_size_y=float(gtlakes_image_size_y)/150.
#
#fig15 = plt.figure(figsize=(18.,12.62))
#ax15 = fig15.add_subplot(1, 1, 1, projection=proj)
#
#im = ax15.imshow(rgb_data[gtlakes_image_crop_top:gtlakes_image_crop_bottom,gtlakes_image_crop_left:gtlakes_image_crop_right], extent=(blue_xa[gtlakes_image_crop_left],blue_xa[gtlakes_image_crop_right],blue_ya[gtlakes_image_crop_bottom],blue_ya[gtlakes_image_crop_top]), origin='upper')
#ax15.coastlines(resolution='50m', color='green')
#ax15.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax15.add_feature(state_boundaries, linestyle=':')
#text15 = ax15.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax15.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text15.set_path_effects(outline_effect)
#filename15="/whirlwind/goes16/vis_color/gtlakes/"+dt+"_gtlakes.jpg"
#isize = fig15.get_size_inches()*fig15.dpi
#ysize=int(isize[1]*0.77)
#fig15.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#fig15.savefig(filename15, bbox_inches='tight', pad_inches=0)
#
#
# END GreatLakes

# Start full size [suffix 9]
# Full res
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))
#ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
#im = ax9.imshow(rgb_data[:], extent=(blue_xa[0],blue_xa[-1],blue_ya[-1],blue_ya[0]), origin='upper')
#ax9.coastlines(resolution='50m', color='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(state_boundaries, linestyle=':')
#text9 = ax9.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax9.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text9.set_path_effects(outline_effect)
#
#filename9="/whirlwind/goes16/vis_color/full/"+dt+"_full.jpg"
#isize = fig9.get_size_inches()*fig9.dpi
#ysize=int(isize[1]*0.77)
#fig9.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#fig9.savefig(filename9, bbox_inches='tight', pad_inches=0)
### UNUSED IN THIS SCRIPT ###
#im = ax10.imshow(rgb_data[alex_image_crop_top:alex_image_crop_bottom,alex_image_crop_left:alex_image_crop_right], extent=(blue_xa[alex_image_crop_left],blue_xa[alex_image_crop_right],blue_ya[alex_image_crop_bottom],blue_ya[alex_image_crop_top]), origin='upper')
### UNUSED IN THIS SCRIPT ###


#ax5.coastlines(resolution='50m', color='green')
#ax6.coastlines(resolution='50m', color='green')
#ax7.coastlines(resolution='50m', color='green')
#ax10.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
#ax5.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax6.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax7.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax10.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Add the feature with dotted lines, denoted by ':'
#ax5.add_feature(state_boundaries2, linewidth=2)
#ax6.add_feature(state_boundaries2, linewidth=2)
#ax7.add_feature(state_boundaries2, linewidth=2)
#ax10.add_feature(state_boundaries, linestyle=':')


#2017/065 20:04:00:30
#text5 = ax5.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax5.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text6 = ax6.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax6.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#
#text7 = ax7.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax7.transAxes,
#    color='darkorange', fontsize='large', weight='bold')
#


#text = ax10.text(0.50, 0.95, time_string,
#    horizontalalignment='center', transform = ax10.transAxes,
#    color='darkorange', fontsize='large', weight='bold')

#filename5="/whirlwind/goes16/vis_color/swi/"+dt+"_swi.jpg"
#filename6="/whirlwind/goes16/vis_color/co/"+dt+"_co.jpg"
#filename7="/whirlwind/goes16/vis_color/fl/"+dt+"_fl.jpg"
#filename10="/whirlwind/goes16/vis_color/alex/"+dt+"_alex.jpg"

#fig5.figimage(aoslogo,  10, int(fig5.bbox.ymax*.96386) - aoslogoheight - 18  , zorder=10)
#fig6.figimage(aoslogo,  10, int(fig6.bbox.ymax*.96386) - aoslogoheight - 18  , zorder=10)
#fig7.figimage(aoslogo,  10, int(fig7.bbox.ymax*.96386) - aoslogoheight - 18  , zorder=10)
#print("fig8.bbox.ymax/.96386",fig8.bbox.ymax/.96386)
#fig10.figimage(aoslogo,  10, int(fig10.bbox.ymax*.96386) - aoslogoheight - 18  , zorder=10)

#fig2.savefig(filename2jpg, bbox_inches='tight')
#fig5.savefig(filename5, bbox_inches='tight')
#fig6.savefig(filename6, bbox_inches='tight')
#fig7.savefig(filename7, bbox_inches='tight')
#fig10.savefig(filename10, bbox_inches='tight')

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


# quit()

# West Coast
silentremove("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_72.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_71.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_72.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_70.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_71.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_69.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_70.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_68.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_69.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_67.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_68.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_66.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_67.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_65.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_66.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_64.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_65.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_63.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_64.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_62.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_63.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_61.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_62.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_60.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_61.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_59.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_60.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_58.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_59.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_57.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_58.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_56.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_57.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_55.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_56.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_54.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_55.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_53.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_54.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_52.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_53.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_51.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_52.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_50.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_51.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_49.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_50.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_48.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_49.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_47.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_48.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_46.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_47.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_45.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_46.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_44.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_45.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_43.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_44.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_42.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_43.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_41.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_42.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_40.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_41.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_39.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_40.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_38.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_39.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_37.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_38.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_36.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_37.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_35.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_36.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_34.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_35.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_33.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_34.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_32.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_33.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_31.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_32.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_30.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_31.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_29.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_30.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_28.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_29.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_27.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_28.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_26.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_27.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_25.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_26.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_24.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_25.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_23.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_24.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_22.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_23.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_21.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_22.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_20.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_21.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_19.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_20.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_18.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_19.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_17.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_18.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_16.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_17.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_15.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_16.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_14.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_15.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_13.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_14.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_12.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_13.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_11.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_12.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_10.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_11.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_9.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_10.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_8.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_9.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_7.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_8.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_6.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_7.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_5.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_6.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_4.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_5.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_3.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_4.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_2.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_3.jpg")
silentrename("/whirlwind/goes17/vis_color/westcoast/latest_westcoast_1.jpg", "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_2.jpg")

shutil.copy(filename1, "/whirlwind/goes17/vis_color/westcoast/latest_westcoast_1.jpg")

# Hawaii
silentremove("/whirlwind/goes17/vis_color/hi/latest_hi_72.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_71.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_72.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_70.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_71.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_69.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_70.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_68.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_69.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_67.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_68.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_66.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_67.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_65.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_66.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_64.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_65.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_63.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_64.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_62.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_63.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_61.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_62.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_60.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_61.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_59.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_60.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_58.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_59.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_57.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_58.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_56.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_57.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_55.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_56.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_54.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_55.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_53.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_54.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_52.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_53.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_51.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_52.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_50.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_51.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_49.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_50.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_48.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_49.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_47.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_48.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_46.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_47.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_45.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_46.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_44.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_45.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_43.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_44.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_42.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_43.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_41.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_42.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_40.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_41.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_39.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_40.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_38.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_39.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_37.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_38.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_36.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_37.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_35.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_36.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_34.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_35.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_33.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_34.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_32.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_33.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_31.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_32.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_30.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_31.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_29.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_30.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_28.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_29.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_27.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_28.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_26.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_27.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_25.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_26.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_24.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_25.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_23.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_24.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_22.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_23.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_21.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_22.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_20.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_21.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_19.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_20.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_18.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_19.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_17.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_18.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_16.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_17.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_15.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_16.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_14.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_15.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_13.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_14.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_12.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_13.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_11.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_12.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_10.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_11.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_9.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_10.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_8.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_9.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_7.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_8.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_6.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_7.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_5.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_6.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_4.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_5.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_3.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_4.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_2.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_3.jpg")
silentrename("/whirlwind/goes17/vis_color/hi/latest_hi_1.jpg", "/whirlwind/goes17/vis_color/hi/latest_hi_2.jpg")

shutil.copy(filename2, "/whirlwind/goes17/vis_color/hi/latest_hi_1.jpg")


silentremove("/whirlwind/goes17/vis_color/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_71.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_70.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_71.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_69.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_70.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_68.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_69.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_67.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_68.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_66.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_67.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_65.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_66.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_64.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_65.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_63.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_64.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_62.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_63.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_61.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_62.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_60.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_61.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_59.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_60.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_58.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_59.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_57.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_58.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_56.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_57.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_55.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_56.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_54.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_55.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_53.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_54.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_52.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_53.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_51.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_52.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_50.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_51.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_49.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_50.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_48.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_49.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_47.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_48.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_46.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_47.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_45.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_46.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_44.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_45.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_43.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_44.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_42.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_43.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_41.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_42.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_40.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_41.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_39.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_40.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_38.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_39.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_37.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_38.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_36.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_37.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_35.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_36.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_34.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_35.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_33.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_34.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_32.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_33.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_31.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_32.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_30.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_31.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_29.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_30.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_28.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_29.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_27.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_28.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_26.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_27.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_25.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_26.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_24.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_25.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_23.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_24.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_22.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_23.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_21.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_22.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_20.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_21.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_19.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_20.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_18.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_19.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_17.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_18.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_16.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_17.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_15.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_16.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_14.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_15.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_13.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_14.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_12.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_13.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_11.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_12.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_10.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_11.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_9.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_10.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_8.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_9.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_7.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_8.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_6.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_7.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_5.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_6.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_4.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_5.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_3.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_4.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_2.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_3.jpg")
silentrename("/whirlwind/goes17/vis_color/conus/latest_conus_1.jpg", "/whirlwind/goes17/vis_color/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes17/vis_color/conus/latest_conus_1.jpg")

# Southwest
silentremove("/whirlwind/goes17/vis_color/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_71.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_70.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_71.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_69.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_70.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_68.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_69.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_67.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_68.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_66.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_67.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_65.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_66.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_64.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_65.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_63.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_64.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_62.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_63.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_61.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_62.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_60.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_61.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_59.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_60.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_58.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_59.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_57.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_58.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_56.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_57.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_55.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_56.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_54.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_55.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_53.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_54.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_52.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_53.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_51.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_52.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_50.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_51.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_49.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_50.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_48.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_49.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_47.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_48.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_46.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_47.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_45.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_46.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_44.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_45.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_43.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_44.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_42.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_43.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_41.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_42.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_40.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_41.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_39.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_40.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_38.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_39.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_37.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_38.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_36.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_37.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_35.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_36.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_34.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_35.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_33.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_34.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_32.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_33.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_31.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_32.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_30.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_31.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_29.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_30.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_28.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_29.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_27.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_28.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_26.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_27.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_25.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_26.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_24.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_25.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_23.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_24.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_22.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_23.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_21.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_22.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_20.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_21.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_19.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_20.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_18.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_19.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_17.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_18.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_16.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_17.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_15.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_16.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_14.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_15.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_13.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_14.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_12.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_13.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_11.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_12.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_10.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_11.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_9.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_10.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_8.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_9.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_7.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_8.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_6.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_7.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_5.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_6.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_4.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_5.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_3.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_4.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_2.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_3.jpg")
silentrename("/whirlwind/goes17/vis_color/sw/latest_sw_1.jpg", "/whirlwind/goes17/vis_color/sw/latest_sw_2.jpg")

shutil.copy(filename13, "/whirlwind/goes17/vis_color/sw/latest_sw_1.jpg")

# Northwest
silentremove("/whirlwind/goes17/vis_color/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_71.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_70.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_71.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_69.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_70.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_68.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_69.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_67.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_68.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_66.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_67.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_65.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_66.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_64.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_65.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_63.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_64.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_62.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_63.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_61.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_62.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_60.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_61.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_59.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_60.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_58.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_59.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_57.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_58.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_56.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_57.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_55.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_56.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_54.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_55.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_53.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_54.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_52.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_53.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_51.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_52.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_50.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_51.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_49.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_50.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_48.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_49.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_47.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_48.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_46.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_47.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_45.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_46.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_44.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_45.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_43.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_44.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_42.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_43.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_41.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_42.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_40.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_41.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_39.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_40.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_38.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_39.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_37.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_38.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_36.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_37.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_35.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_36.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_34.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_35.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_33.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_34.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_32.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_33.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_31.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_32.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_30.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_31.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_29.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_30.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_28.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_29.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_27.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_28.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_26.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_27.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_25.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_26.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_24.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_25.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_23.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_24.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_22.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_23.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_21.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_22.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_20.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_21.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_19.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_20.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_18.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_19.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_17.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_18.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_16.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_17.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_15.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_16.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_14.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_15.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_13.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_14.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_12.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_13.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_11.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_12.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_10.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_11.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_9.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_10.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_8.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_9.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_7.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_8.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_6.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_7.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_5.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_6.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_4.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_5.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_3.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_4.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_2.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_3.jpg")
silentrename("/whirlwind/goes17/vis_color/nw/latest_nw_1.jpg", "/whirlwind/goes17/vis_color/nw/latest_nw_2.jpg")

shutil.copy(filename14, "/whirlwind/goes17/vis_color/nw/latest_nw_1.jpg")

#shutil.copy(filename9, "/whirlwind/goes17/vis_color/full/latest_full_1.jpg")


quit()

