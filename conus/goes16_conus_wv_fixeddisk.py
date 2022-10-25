#!/home/poker/miniconda3/envs/goes16_201710/bin/python

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
from PIL import Image

aoslogo = Image.open('/home/poker/uw-aoslogo.png')
aoslogoheight = aoslogo.size[1]
aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
aoslogo = np.array(aoslogo).astype(np.float) / 255


band="09"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

prod_id = "TIRE"
#dt="201703051957"
dt = sys.argv[1]

#f = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


print(f)

data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

print(data_var)

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

# Create a new figure with size 10" by 10"
fig = plt.figure(figsize=(16.,10.17))
fig2 = plt.figure(figsize=(18.,12.62))
fig3 = plt.figure(figsize=(20.0, 11.987))
fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
ax9 = fig9.add_subplot(1, 1, 1, projection=proj)


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

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,1024)

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

im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.0)

import cartopy.feature as cfeat

ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax9.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax9.add_feature(state_boundaries, linestyle=':')

# axes for wi
cbaxes1 = fig.add_axes([0.135,0.12,0.755,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')

# axes for mw
cbaxes2 = fig2.add_axes([0.135,0.12,0.755,0.02])
cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
font_size = 14
#cbar2.set_label('Brightness Temperature (K)',size=18)
cbar2.ax.tick_params(labelsize=font_size)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

# axes for conus
cbaxes3 = fig3.add_axes([0.135,0.12,0.755,0.02])
cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar3.ax.tick_params(labelsize=font_size)
cbar3.ax.xaxis.set_ticks_position('top')
cbar3.ax.xaxis.set_label_position('top')

# axes for full
cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
font_size = 18
#cbar9.set_label('Brightness Temperature (K)',size=18)
cbar9.ax.tick_params(labelsize=font_size)
cbar9.ax.xaxis.set_ticks_position('top')
cbar9.ax.xaxis.set_label_position('top')

# Redisplay modified figure
#fig
#fig2

import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

time_string = 'GOES16 Mid-level Water Vapor (ABI ch 9)\n%s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
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
    color='darkorange', fontsize='large', weight='bold')

text3.set_path_effects(outline_effect)

text9 = ax9.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax9.transAxes,
    color='black', fontsize='large', weight='bold')

text9.set_path_effects(outline_effect)



filename1="/whirlwind/goes16/wv/wi/"+dt+"_wi.jpg"
filename2="/whirlwind/goes16/wv/mw/"+dt+"_mw.jpg"
filename3="/whirlwind/goes16/wv/conus/"+dt+"_conus.jpg"
filename9="/whirlwind/goes16/wv/full/"+dt+"_full.jpg"

isize = fig.get_size_inches()*fig.dpi
ysize=int(isize[1]*0.97)
fig.figimage(aoslogo,  0, ysize - aoslogoheight  , zorder=10)
isize = fig2.get_size_inches()*fig2.dpi
ysize=int(isize[1]*0.97)
fig2.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
isize = fig3.get_size_inches()*fig3.dpi
ysize=int(isize[1]*0.97)
fig3.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
isize = fig9.get_size_inches()*fig9.dpi
ysize=int(isize[1]*0.97)
fig9.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)

fig.savefig(filename1, bbox_inches='tight', pad_inches=0)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)
#fig2.savefig(filename2jpg, bbox_inches='tight', pad_inches=0)
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
fig9.savefig(filename9, bbox_inches='tight', pad_inches=0)

#import os.rename    # os.rename(src,dest)
#import os.remove    # os.remove path
#import shutil.copy  # shutil.copy(src, dest)

os.remove("/whirlwind/goes16/wv/wi/latest_wi_72.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_71.jpg", "/whirlwind/goes16/wv/wi/latest_wi_72.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_70.jpg", "/whirlwind/goes16/wv/wi/latest_wi_71.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_69.jpg", "/whirlwind/goes16/wv/wi/latest_wi_70.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_68.jpg", "/whirlwind/goes16/wv/wi/latest_wi_69.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_67.jpg", "/whirlwind/goes16/wv/wi/latest_wi_68.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_66.jpg", "/whirlwind/goes16/wv/wi/latest_wi_67.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_65.jpg", "/whirlwind/goes16/wv/wi/latest_wi_66.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_64.jpg", "/whirlwind/goes16/wv/wi/latest_wi_65.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_63.jpg", "/whirlwind/goes16/wv/wi/latest_wi_64.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_62.jpg", "/whirlwind/goes16/wv/wi/latest_wi_63.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_61.jpg", "/whirlwind/goes16/wv/wi/latest_wi_62.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_60.jpg", "/whirlwind/goes16/wv/wi/latest_wi_61.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_59.jpg", "/whirlwind/goes16/wv/wi/latest_wi_60.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_58.jpg", "/whirlwind/goes16/wv/wi/latest_wi_59.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_57.jpg", "/whirlwind/goes16/wv/wi/latest_wi_58.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_56.jpg", "/whirlwind/goes16/wv/wi/latest_wi_57.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_55.jpg", "/whirlwind/goes16/wv/wi/latest_wi_56.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_54.jpg", "/whirlwind/goes16/wv/wi/latest_wi_55.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_53.jpg", "/whirlwind/goes16/wv/wi/latest_wi_54.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_52.jpg", "/whirlwind/goes16/wv/wi/latest_wi_53.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_51.jpg", "/whirlwind/goes16/wv/wi/latest_wi_52.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_50.jpg", "/whirlwind/goes16/wv/wi/latest_wi_51.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_49.jpg", "/whirlwind/goes16/wv/wi/latest_wi_50.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_48.jpg", "/whirlwind/goes16/wv/wi/latest_wi_49.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_47.jpg", "/whirlwind/goes16/wv/wi/latest_wi_48.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_46.jpg", "/whirlwind/goes16/wv/wi/latest_wi_47.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_45.jpg", "/whirlwind/goes16/wv/wi/latest_wi_46.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_44.jpg", "/whirlwind/goes16/wv/wi/latest_wi_45.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_43.jpg", "/whirlwind/goes16/wv/wi/latest_wi_44.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_42.jpg", "/whirlwind/goes16/wv/wi/latest_wi_43.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_41.jpg", "/whirlwind/goes16/wv/wi/latest_wi_42.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_40.jpg", "/whirlwind/goes16/wv/wi/latest_wi_41.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_39.jpg", "/whirlwind/goes16/wv/wi/latest_wi_40.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_38.jpg", "/whirlwind/goes16/wv/wi/latest_wi_39.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_37.jpg", "/whirlwind/goes16/wv/wi/latest_wi_38.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_36.jpg", "/whirlwind/goes16/wv/wi/latest_wi_37.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_35.jpg", "/whirlwind/goes16/wv/wi/latest_wi_36.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_34.jpg", "/whirlwind/goes16/wv/wi/latest_wi_35.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_33.jpg", "/whirlwind/goes16/wv/wi/latest_wi_34.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_32.jpg", "/whirlwind/goes16/wv/wi/latest_wi_33.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_31.jpg", "/whirlwind/goes16/wv/wi/latest_wi_32.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_30.jpg", "/whirlwind/goes16/wv/wi/latest_wi_31.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_29.jpg", "/whirlwind/goes16/wv/wi/latest_wi_30.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_28.jpg", "/whirlwind/goes16/wv/wi/latest_wi_29.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_27.jpg", "/whirlwind/goes16/wv/wi/latest_wi_28.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_26.jpg", "/whirlwind/goes16/wv/wi/latest_wi_27.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_25.jpg", "/whirlwind/goes16/wv/wi/latest_wi_26.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_24.jpg", "/whirlwind/goes16/wv/wi/latest_wi_25.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_23.jpg", "/whirlwind/goes16/wv/wi/latest_wi_24.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_22.jpg", "/whirlwind/goes16/wv/wi/latest_wi_23.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_21.jpg", "/whirlwind/goes16/wv/wi/latest_wi_22.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_20.jpg", "/whirlwind/goes16/wv/wi/latest_wi_21.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_19.jpg", "/whirlwind/goes16/wv/wi/latest_wi_20.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_18.jpg", "/whirlwind/goes16/wv/wi/latest_wi_19.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_17.jpg", "/whirlwind/goes16/wv/wi/latest_wi_18.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_16.jpg", "/whirlwind/goes16/wv/wi/latest_wi_17.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_15.jpg", "/whirlwind/goes16/wv/wi/latest_wi_16.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_14.jpg", "/whirlwind/goes16/wv/wi/latest_wi_15.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_13.jpg", "/whirlwind/goes16/wv/wi/latest_wi_14.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_12.jpg", "/whirlwind/goes16/wv/wi/latest_wi_13.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_11.jpg", "/whirlwind/goes16/wv/wi/latest_wi_12.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_10.jpg", "/whirlwind/goes16/wv/wi/latest_wi_11.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_9.jpg", "/whirlwind/goes16/wv/wi/latest_wi_10.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_8.jpg", "/whirlwind/goes16/wv/wi/latest_wi_9.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_7.jpg", "/whirlwind/goes16/wv/wi/latest_wi_8.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_6.jpg", "/whirlwind/goes16/wv/wi/latest_wi_7.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_5.jpg", "/whirlwind/goes16/wv/wi/latest_wi_6.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_4.jpg", "/whirlwind/goes16/wv/wi/latest_wi_5.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_3.jpg", "/whirlwind/goes16/wv/wi/latest_wi_4.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_2.jpg", "/whirlwind/goes16/wv/wi/latest_wi_3.jpg")
os.rename("/whirlwind/goes16/wv/wi/latest_wi_1.jpg", "/whirlwind/goes16/wv/wi/latest_wi_2.jpg")

shutil.copy(filename1, "/whirlwind/goes16/wv/wi/latest_wi_1.jpg")


os.remove("/whirlwind/goes16/wv/mw/latest_mw_72.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_71.jpg", "/whirlwind/goes16/wv/mw/latest_mw_72.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_70.jpg", "/whirlwind/goes16/wv/mw/latest_mw_71.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_69.jpg", "/whirlwind/goes16/wv/mw/latest_mw_70.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_68.jpg", "/whirlwind/goes16/wv/mw/latest_mw_69.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_67.jpg", "/whirlwind/goes16/wv/mw/latest_mw_68.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_66.jpg", "/whirlwind/goes16/wv/mw/latest_mw_67.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_65.jpg", "/whirlwind/goes16/wv/mw/latest_mw_66.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_64.jpg", "/whirlwind/goes16/wv/mw/latest_mw_65.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_63.jpg", "/whirlwind/goes16/wv/mw/latest_mw_64.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_62.jpg", "/whirlwind/goes16/wv/mw/latest_mw_63.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_61.jpg", "/whirlwind/goes16/wv/mw/latest_mw_62.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_60.jpg", "/whirlwind/goes16/wv/mw/latest_mw_61.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_59.jpg", "/whirlwind/goes16/wv/mw/latest_mw_60.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_58.jpg", "/whirlwind/goes16/wv/mw/latest_mw_59.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_57.jpg", "/whirlwind/goes16/wv/mw/latest_mw_58.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_56.jpg", "/whirlwind/goes16/wv/mw/latest_mw_57.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_55.jpg", "/whirlwind/goes16/wv/mw/latest_mw_56.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_54.jpg", "/whirlwind/goes16/wv/mw/latest_mw_55.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_53.jpg", "/whirlwind/goes16/wv/mw/latest_mw_54.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_52.jpg", "/whirlwind/goes16/wv/mw/latest_mw_53.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_51.jpg", "/whirlwind/goes16/wv/mw/latest_mw_52.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_50.jpg", "/whirlwind/goes16/wv/mw/latest_mw_51.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_49.jpg", "/whirlwind/goes16/wv/mw/latest_mw_50.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_48.jpg", "/whirlwind/goes16/wv/mw/latest_mw_49.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_47.jpg", "/whirlwind/goes16/wv/mw/latest_mw_48.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_46.jpg", "/whirlwind/goes16/wv/mw/latest_mw_47.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_45.jpg", "/whirlwind/goes16/wv/mw/latest_mw_46.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_44.jpg", "/whirlwind/goes16/wv/mw/latest_mw_45.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_43.jpg", "/whirlwind/goes16/wv/mw/latest_mw_44.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_42.jpg", "/whirlwind/goes16/wv/mw/latest_mw_43.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_41.jpg", "/whirlwind/goes16/wv/mw/latest_mw_42.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_40.jpg", "/whirlwind/goes16/wv/mw/latest_mw_41.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_39.jpg", "/whirlwind/goes16/wv/mw/latest_mw_40.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_38.jpg", "/whirlwind/goes16/wv/mw/latest_mw_39.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_37.jpg", "/whirlwind/goes16/wv/mw/latest_mw_38.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_36.jpg", "/whirlwind/goes16/wv/mw/latest_mw_37.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_35.jpg", "/whirlwind/goes16/wv/mw/latest_mw_36.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_34.jpg", "/whirlwind/goes16/wv/mw/latest_mw_35.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_33.jpg", "/whirlwind/goes16/wv/mw/latest_mw_34.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_32.jpg", "/whirlwind/goes16/wv/mw/latest_mw_33.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_31.jpg", "/whirlwind/goes16/wv/mw/latest_mw_32.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_30.jpg", "/whirlwind/goes16/wv/mw/latest_mw_31.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_29.jpg", "/whirlwind/goes16/wv/mw/latest_mw_30.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_28.jpg", "/whirlwind/goes16/wv/mw/latest_mw_29.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_27.jpg", "/whirlwind/goes16/wv/mw/latest_mw_28.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_26.jpg", "/whirlwind/goes16/wv/mw/latest_mw_27.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_25.jpg", "/whirlwind/goes16/wv/mw/latest_mw_26.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_24.jpg", "/whirlwind/goes16/wv/mw/latest_mw_25.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_23.jpg", "/whirlwind/goes16/wv/mw/latest_mw_24.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_22.jpg", "/whirlwind/goes16/wv/mw/latest_mw_23.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_21.jpg", "/whirlwind/goes16/wv/mw/latest_mw_22.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_20.jpg", "/whirlwind/goes16/wv/mw/latest_mw_21.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_19.jpg", "/whirlwind/goes16/wv/mw/latest_mw_20.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_18.jpg", "/whirlwind/goes16/wv/mw/latest_mw_19.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_17.jpg", "/whirlwind/goes16/wv/mw/latest_mw_18.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_16.jpg", "/whirlwind/goes16/wv/mw/latest_mw_17.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_15.jpg", "/whirlwind/goes16/wv/mw/latest_mw_16.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_14.jpg", "/whirlwind/goes16/wv/mw/latest_mw_15.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_13.jpg", "/whirlwind/goes16/wv/mw/latest_mw_14.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_12.jpg", "/whirlwind/goes16/wv/mw/latest_mw_13.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_11.jpg", "/whirlwind/goes16/wv/mw/latest_mw_12.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_10.jpg", "/whirlwind/goes16/wv/mw/latest_mw_11.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_9.jpg", "/whirlwind/goes16/wv/mw/latest_mw_10.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_8.jpg", "/whirlwind/goes16/wv/mw/latest_mw_9.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_7.jpg", "/whirlwind/goes16/wv/mw/latest_mw_8.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_6.jpg", "/whirlwind/goes16/wv/mw/latest_mw_7.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_5.jpg", "/whirlwind/goes16/wv/mw/latest_mw_6.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_4.jpg", "/whirlwind/goes16/wv/mw/latest_mw_5.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_3.jpg", "/whirlwind/goes16/wv/mw/latest_mw_4.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_2.jpg", "/whirlwind/goes16/wv/mw/latest_mw_3.jpg")
os.rename("/whirlwind/goes16/wv/mw/latest_mw_1.jpg", "/whirlwind/goes16/wv/mw/latest_mw_2.jpg")

shutil.copy(filename2, "/whirlwind/goes16/wv/mw/latest_mw_1.jpg")


os.remove("/whirlwind/goes16/wv/conus/latest_conus_72.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_71.jpg", "/whirlwind/goes16/wv/conus/latest_conus_72.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_70.jpg", "/whirlwind/goes16/wv/conus/latest_conus_71.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_69.jpg", "/whirlwind/goes16/wv/conus/latest_conus_70.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_68.jpg", "/whirlwind/goes16/wv/conus/latest_conus_69.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_67.jpg", "/whirlwind/goes16/wv/conus/latest_conus_68.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_66.jpg", "/whirlwind/goes16/wv/conus/latest_conus_67.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_65.jpg", "/whirlwind/goes16/wv/conus/latest_conus_66.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_64.jpg", "/whirlwind/goes16/wv/conus/latest_conus_65.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_63.jpg", "/whirlwind/goes16/wv/conus/latest_conus_64.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_62.jpg", "/whirlwind/goes16/wv/conus/latest_conus_63.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_61.jpg", "/whirlwind/goes16/wv/conus/latest_conus_62.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_60.jpg", "/whirlwind/goes16/wv/conus/latest_conus_61.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_59.jpg", "/whirlwind/goes16/wv/conus/latest_conus_60.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_58.jpg", "/whirlwind/goes16/wv/conus/latest_conus_59.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_57.jpg", "/whirlwind/goes16/wv/conus/latest_conus_58.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_56.jpg", "/whirlwind/goes16/wv/conus/latest_conus_57.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_55.jpg", "/whirlwind/goes16/wv/conus/latest_conus_56.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_54.jpg", "/whirlwind/goes16/wv/conus/latest_conus_55.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_53.jpg", "/whirlwind/goes16/wv/conus/latest_conus_54.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_52.jpg", "/whirlwind/goes16/wv/conus/latest_conus_53.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_51.jpg", "/whirlwind/goes16/wv/conus/latest_conus_52.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_50.jpg", "/whirlwind/goes16/wv/conus/latest_conus_51.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_49.jpg", "/whirlwind/goes16/wv/conus/latest_conus_50.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_48.jpg", "/whirlwind/goes16/wv/conus/latest_conus_49.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_47.jpg", "/whirlwind/goes16/wv/conus/latest_conus_48.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_46.jpg", "/whirlwind/goes16/wv/conus/latest_conus_47.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_45.jpg", "/whirlwind/goes16/wv/conus/latest_conus_46.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_44.jpg", "/whirlwind/goes16/wv/conus/latest_conus_45.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_43.jpg", "/whirlwind/goes16/wv/conus/latest_conus_44.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_42.jpg", "/whirlwind/goes16/wv/conus/latest_conus_43.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_41.jpg", "/whirlwind/goes16/wv/conus/latest_conus_42.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_40.jpg", "/whirlwind/goes16/wv/conus/latest_conus_41.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_39.jpg", "/whirlwind/goes16/wv/conus/latest_conus_40.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_38.jpg", "/whirlwind/goes16/wv/conus/latest_conus_39.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_37.jpg", "/whirlwind/goes16/wv/conus/latest_conus_38.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_36.jpg", "/whirlwind/goes16/wv/conus/latest_conus_37.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_35.jpg", "/whirlwind/goes16/wv/conus/latest_conus_36.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_34.jpg", "/whirlwind/goes16/wv/conus/latest_conus_35.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_33.jpg", "/whirlwind/goes16/wv/conus/latest_conus_34.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_32.jpg", "/whirlwind/goes16/wv/conus/latest_conus_33.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_31.jpg", "/whirlwind/goes16/wv/conus/latest_conus_32.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_30.jpg", "/whirlwind/goes16/wv/conus/latest_conus_31.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_29.jpg", "/whirlwind/goes16/wv/conus/latest_conus_30.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_28.jpg", "/whirlwind/goes16/wv/conus/latest_conus_29.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_27.jpg", "/whirlwind/goes16/wv/conus/latest_conus_28.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_26.jpg", "/whirlwind/goes16/wv/conus/latest_conus_27.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_25.jpg", "/whirlwind/goes16/wv/conus/latest_conus_26.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_24.jpg", "/whirlwind/goes16/wv/conus/latest_conus_25.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_23.jpg", "/whirlwind/goes16/wv/conus/latest_conus_24.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_22.jpg", "/whirlwind/goes16/wv/conus/latest_conus_23.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_21.jpg", "/whirlwind/goes16/wv/conus/latest_conus_22.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_20.jpg", "/whirlwind/goes16/wv/conus/latest_conus_21.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_19.jpg", "/whirlwind/goes16/wv/conus/latest_conus_20.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_18.jpg", "/whirlwind/goes16/wv/conus/latest_conus_19.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_17.jpg", "/whirlwind/goes16/wv/conus/latest_conus_18.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_16.jpg", "/whirlwind/goes16/wv/conus/latest_conus_17.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_15.jpg", "/whirlwind/goes16/wv/conus/latest_conus_16.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_14.jpg", "/whirlwind/goes16/wv/conus/latest_conus_15.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_13.jpg", "/whirlwind/goes16/wv/conus/latest_conus_14.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_12.jpg", "/whirlwind/goes16/wv/conus/latest_conus_13.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_11.jpg", "/whirlwind/goes16/wv/conus/latest_conus_12.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_10.jpg", "/whirlwind/goes16/wv/conus/latest_conus_11.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_9.jpg", "/whirlwind/goes16/wv/conus/latest_conus_10.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_8.jpg", "/whirlwind/goes16/wv/conus/latest_conus_9.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_7.jpg", "/whirlwind/goes16/wv/conus/latest_conus_8.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_6.jpg", "/whirlwind/goes16/wv/conus/latest_conus_7.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_5.jpg", "/whirlwind/goes16/wv/conus/latest_conus_6.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_4.jpg", "/whirlwind/goes16/wv/conus/latest_conus_5.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_3.jpg", "/whirlwind/goes16/wv/conus/latest_conus_4.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_2.jpg", "/whirlwind/goes16/wv/conus/latest_conus_3.jpg")
os.rename("/whirlwind/goes16/wv/conus/latest_conus_1.jpg", "/whirlwind/goes16/wv/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes16/wv/conus/latest_conus_1.jpg")

shutil.copy(filename9, "/whirlwind/goes16/wv/full/latest_full_1.jpg")
