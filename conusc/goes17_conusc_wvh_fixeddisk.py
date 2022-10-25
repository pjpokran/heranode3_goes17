#!/home/poker/miniconda3/envs/goes16_201710/bin/python

# GOES 16 WV (ABI Channel 08) plotting script. Based on 
# http://nbviewer.jupyter.org/github/unidata/notebook-gallery/blob/master/notebooks/Plotting_GINI_Water_Vapor_Imagery_Part2.ipynb
# from the Unidata python notebook gallery

# import modules
import netCDF4
import matplotlib
# needed for batch processing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

band="08"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

prod_id = "TIRW"
# pull date/time to process from stdin. YYYYMMDDHHmm of file named /weather/data/goes16/TIRE/##/YYYYMMDDHHmm_PAA.nc
#dt="201703051957"
dt = sys.argv[1]

# read in latest file and get time
e = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/latest.nc")
#e = netCDF4.Dataset(dt)
print(e)
dtjulian = e.start_date_time[:-2]
print(dtjulian)
dt = datetime.datetime.strptime(dtjulian,'%Y%j%H%M').strftime('%Y%m%d%H%M')
print (dt)

print ("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")

# sleep(30)

# read in first tile, create numpy arrays big enough to hold entire tile data, x and y coords (a, xa, ya)
#f = netCDF4.Dataset("/weather/data/goes16/TIRE/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


# print out some metadata for info. This can be commented out
print(f)

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
        if os.path.isfile("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc")
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
#a[a==0.] = 1.
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

print("image_rows, image_columns",image_rows, image_columns)

# Here I start setting crop values so I can plot multiple images from this one
# image pasted together from the multiple sector netcdf files
#

# Number of pixels to be cropped from the top, bottom, left and right of each 
# image. Bottom and Right are negative numbers (using python's negative array
# indexing - -1 is the last element in the array dimension)
#
# West Coast close-up crop
westcoast_image_crop_top=0
westcoast_image_crop_bottom=-750
westcoast_image_crop_left=1200
westcoast_image_crop_right=-50

westcoast_image_size_y=(image_rows+westcoast_image_crop_bottom-westcoast_image_crop_top)
westcoast_image_size_x=(image_columns+westcoast_image_crop_right-westcoast_image_crop_left)

print("westcoast image size")
print(westcoast_image_size_x, westcoast_image_size_y)

# This determines the size of the picture, and based on how many elements
# in the array, the resolution of each pixel. I experimented to come up
# with 80 for wi, 150 for mw, etc.. You can change those to see how they
# affect the final image.
#
westcoast_image_size_x=float(westcoast_image_size_x)/40.
westcoast_image_size_y=float(westcoast_image_size_y)/40.

# Same stuff for Midwest crop
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

gtlakes_image_crop_top=4
gtlakes_image_crop_bottom=-1080
gtlakes_image_crop_left=1141
gtlakes_image_crop_right=-661

gtlakes_image_size_y=(image_rows+gtlakes_image_crop_bottom-gtlakes_image_crop_top)
gtlakes_image_size_x=(image_columns+gtlakes_image_crop_right-gtlakes_image_crop_left)

print("gtlakes image size")
print(gtlakes_image_size_x, gtlakes_image_size_y)

gtlakes_image_size_x=float(gtlakes_image_size_x)/10.
gtlakes_image_size_y=float(gtlakes_image_size_y)/10.

# These create the figure objects for the Wisconsin (fig), Midwest (fig2), 
# CONUS (fig3) and full resolution (fig9) images. The fig9 is not used
# for any loops, and browsers will scale it down but you can click to zoom
# to get the full resolution in all its beauty.
#
# West Coast Crop
fig = plt.figure(figsize=(20.,11.987))
# Hawaii crop
fig2 = plt.figure(figsize=(14.,14))
# CONUS
fig3 = plt.figure(figsize=(20.,11.987))
# Northeast crop
fig4 = plt.figure(figsize=(18.,12.27))
# Gulf of Mexico region
#fig8 = plt.figure(figsize=(30.,18.026))
#fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))

# Southwest US
fig13 = plt.figure(figsize=(18.,12.25))
# Northwest US
fig14 = plt.figure(figsize=(18.,8.009))
## GreatLakes
#fig15 = plt.figure(figsize=(18.,10.73))
print("create axes")

fig0 = plt.figure(figsize=(2.0,2.0))
fig20 = plt.figure(figsize=(2.0,2.0))
fig30 = plt.figure(figsize=(2.0,2.0))
#fig40 = plt.figure(figsize=(2.0,2.0))
#fig80 = plt.figure(figsize=(2.0,2.0))
fig130 = plt.figure(figsize=(2.0,2.0))
fig140 = plt.figure(figsize=(2.0,2.0))
#fig150 = plt.figure(figsize=(2.0,2.0))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
ax3.outline_patch.set_edgecolor('none')
#ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
#ax4.outline_patch.set_edgecolor('none')
#ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
#ax8.outline_patch.set_edgecolor('none')
#ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
#ax9.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(1, 1, 1, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(1, 1, 1, projection=proj)
ax14.outline_patch.set_edgecolor('none')
#ax15 = fig15.add_subplot(1, 1, 1, projection=proj)
#ax15.outline_patch.set_edgecolor('none')
ax0 = fig0.add_subplot(1, 1, 1, projection=proj)
ax0.outline_patch.set_edgecolor('none')
ax20 = fig20.add_subplot(1, 1, 1, projection=proj)
ax20.outline_patch.set_edgecolor('none')
ax30 = fig30.add_subplot(1, 1, 1, projection=proj)
ax30.outline_patch.set_edgecolor('none')
#ax40 = fig40.add_subplot(1, 1, 1, projection=proj)
#ax40.outline_patch.set_edgecolor('none')
#ax80 = fig80.add_subplot(1, 1, 1, projection=proj)
#ax80.outline_patch.set_edgecolor('none')
ax130 = fig130.add_subplot(1, 1, 1, projection=proj)
ax130.outline_patch.set_edgecolor('none')
ax140 = fig140.add_subplot(1, 1, 1, projection=proj)
ax140.outline_patch.set_edgecolor('none')
#ax150 = fig150.add_subplot(1, 1, 1, projection=proj)
#ax150.outline_patch.set_edgecolor('none')

cdict = {'red': ((0.0, 0.0, 0.0),
                 (.001, 1.00, 1.00),
                 (.107, 1.00, 1.00),
                 (.113, 0.498, 0.498),
                 (.173, 1.00, 1.00),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00),
                 (.287, 0.902, 0.902),
                 (.293, 1.00, 1.00),
                 (.346, 1.00, 1.00),
                 (.352, 1.00, 1.00),
                 (.406, 0.101, 0.101),
                 (.412, 0.00, 0.00),
                 (.481, 0.00, 0.00),
                 (.484, 0.00, 0.00),
                 (.543, 0.00, 0.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                 (.001, 1.00, 1.00),
                 (.107, 1.00, 1.00),
                 (.113, 0.00, 0.00),
                 (.173, 0.498, 0.498),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00),
                 (.287, 0.00, 0.00),
                 (.293, 0.00, 0.00),
                 (.346, 0.902, 0.902),
                 (.352, 1.00, 1.00),
                 (.406, 1.00, 1.00),
                 (.412, 1.00, 1.00),
                 (.481, 0.00, 0.00),
                 (.484, 0.00, 0.00),
                 (.543, 1.00, 1.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.00, 0.00),
                 (.001, 1.00, 1.00),
                 (.107, 0.00, 0.00),
                 (.113, 0.498, 0.498),
                 (.173, 0.786, 0.786),
                 (.179, 0.902, 0.902),
                 (.227, 0.102, 0.102),
                 (.233, 0.00, 0.00), 
                 (.287, 0.00, 0.00),
                 (.293, 0.00, 0.00),
                 (.346, 0.00, 0.00),
                 (.352, 0.00, 0.00),
                 (.406, 0.00, 0.00),
                 (.412, 0.00, 0.00),
                 (.481, 0.451, 0.451),
                 (.484, 0.451, 0.451),
                 (.543, 1.00, 1.00),
                 (.546, 0.773, 0.773),
                 (.994, 0.012, 0.012),
                 (.997, 0.004, 0.004),
                  (1.0, 0.0, 0.0))}

cdict2 = {'red': ((0.0, 0.0, 0.0),
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


import matplotlib as mpl
my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict2,2048)


# Plot the data 
# set the colormap to extend over a range of values from 162 to 330 using the
# Greys built-in color map.
# Note, we save the image returned by imshow for later...

im = ax.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax2.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.0)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax15.imshow(a[gtlakes_image_crop_top:gtlakes_image_crop_bottom,gtlakes_image_crop_left:gtlakes_image_crop_right], extent=(xa[gtlakes_image_crop_left],xa[gtlakes_image_crop_right],ya[gtlakes_image_crop_bottom],ya[gtlakes_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax0.imshow(a[westcoast_image_crop_top:westcoast_image_crop_bottom,westcoast_image_crop_left:westcoast_image_crop_right], extent=(xa[westcoast_image_crop_left],xa[westcoast_image_crop_right],ya[westcoast_image_crop_bottom],ya[westcoast_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax20.imshow(a[hi_image_crop_top:hi_image_crop_bottom,hi_image_crop_left:hi_image_crop_right], extent=(xa[hi_image_crop_left],xa[hi_image_crop_right],ya[hi_image_crop_bottom],ya[hi_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax30.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax40.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax80.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax130.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
im = ax140.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)
#im = ax150.imshow(a[gtlakes_image_crop_top:gtlakes_image_crop_bottom,gtlakes_image_crop_left:gtlakes_image_crop_right], extent=(xa[gtlakes_image_crop_left],xa[gtlakes_image_crop_right],ya[gtlakes_image_crop_bottom],ya[gtlakes_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.0)

import cartopy.feature as cfeat

ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='red')
ax3.coastlines(resolution='50m', color='green')
#ax4.coastlines(resolution='50m', color='green')
#ax8.coastlines(resolution='50m', color='green')
#ax9.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
#ax15.coastlines(resolution='50m', color='green')
ax0.coastlines(resolution='50m', color='green')
ax20.coastlines(resolution='50m', color='green')
ax30.coastlines(resolution='50m', color='green')
#ax40.coastlines(resolution='50m', color='green')
#ax80.coastlines(resolution='50m', color='green')
ax130.coastlines(resolution='50m', color='green')
ax140.coastlines(resolution='50m', color='green')
#ax150.coastlines(resolution='50m', color='green')

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='red')
ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax4.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax8.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax15.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax0.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax20.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax30.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax40.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax80.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax130.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
ax140.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#ax150.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
#ax4.add_feature(state_boundaries, linestyle=':')
#ax8.add_feature(state_boundaries, linestyle=':')
#ax9.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
#ax15.add_feature(state_boundaries, linestyle=':')
ax0.add_feature(state_boundaries, linestyle=':')
ax20.add_feature(state_boundaries, linestyle=':')
ax30.add_feature(state_boundaries, linestyle=':')
#ax40.add_feature(state_boundaries, linestyle=':')
#ax80.add_feature(state_boundaries, linestyle=':')
ax130.add_feature(state_boundaries, linestyle=':')
ax140.add_feature(state_boundaries, linestyle=':')
#ax150.add_feature(state_boundaries, linestyle=':')

## axes for wi
cbaxes1 = fig.add_axes([0.135,0.14,0.755,0.02])
cbar1 = fig.colorbar(im, cax=cbaxes1, orientation='horizontal')
font_size = 14
#cbar1.set_label('Brightness Temperature (K)',size=18)
cbar1.ax.tick_params(labelsize=font_size)
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')
#
# axes for mw
cbaxes2 = fig2.add_axes([0.135,0.14,0.755,0.02])
cbar2 = fig2.colorbar(im, cax=cbaxes2, orientation='horizontal')
font_size = 14
#cbar2.set_label('Brightness Temperature (K)',size=18)
cbar2.ax.tick_params(labelsize=font_size)
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')
#
# axes for conus
cbaxes3 = fig3.add_axes([0.135,0.14,0.755,0.02])
cbar3 = fig3.colorbar(im, cax=cbaxes3, orientation='horizontal')
font_size = 14
#cbar3.set_label('Brightness Temperature (K)',size=18)
cbar3.ax.tick_params(labelsize=font_size)
cbar3.ax.xaxis.set_ticks_position('top')
cbar3.ax.xaxis.set_label_position('top')

## axes for ne
#cbaxes4 = fig4.add_axes([0.135,0.12,0.755,0.02])
#cbar4 = fig4.colorbar(im, cax=cbaxes4, orientation='horizontal')
#font_size = 14
##cbar3.set_label('Brightness Temperature (K)',size=18)
#cbar4.ax.tick_params(labelsize=font_size)
#cbar4.ax.xaxis.set_ticks_position('top')
#cbar4.ax.xaxis.set_label_position('top')
#
## axes for gulf
#cbaxes8 = fig8.add_axes([0.135,0.12,0.755,0.02])
#cbar8 = fig8.colorbar(im, cax=cbaxes8, orientation='horizontal')
#font_size = 14
##cbar3.set_label('Brightness Temperature (K)',size=18)
#cbar8.ax.tick_params(labelsize=font_size)
#cbar8.ax.xaxis.set_ticks_position('top')
#cbar8.ax.xaxis.set_label_position('top')
#
## axes for full
#cbaxes9 = fig9.add_axes([0.135,0.15,0.755,0.02])
#cbar9 = fig9.colorbar(im, cax=cbaxes9, orientation='horizontal')
#font_size = 18
##cbar9.set_label('Brightness Temperature (K)',size=18)
#cbar9.ax.tick_params(labelsize=font_size, labelcolor="yellow")
#cbar9.ax.xaxis.set_ticks_position('top')
#cbar9.ax.xaxis.set_label_position('top')
#
# axes for sw
cbaxes13 = fig13.add_axes([0.135,0.14,0.755,0.02])
cbar13 = fig13.colorbar(im, cax=cbaxes13, orientation='horizontal')
font_size = 14
#cbar13.set_label('Brightness Temperature (K)',size=18)
cbar13.ax.tick_params(labelsize=font_size)
cbar13.ax.xaxis.set_ticks_position('top')
cbar13.ax.xaxis.set_label_position('top')

# axes for nw
cbaxes14 = fig14.add_axes([0.135,0.14,0.755,0.02])
cbar14 = fig14.colorbar(im, cax=cbaxes14, orientation='horizontal')
font_size = 14
#cbar14.set_label('Brightness Temperature (K)',size=18)
cbar14.ax.tick_params(labelsize=font_size)
cbar14.ax.xaxis.set_ticks_position('top')
cbar14.ax.xaxis.set_label_position('top')
#
## axes for gtlakes
#cbaxes15 = fig15.add_axes([0.135,0.12,0.755,0.02])
#cbar15 = fig15.colorbar(im, cax=cbaxes15, orientation='horizontal')
#font_size = 14
##cbar15.set_label('Brightness Temperature (K)',size=18)
#cbar15.ax.tick_params(labelsize=font_size)
#cbar15.ax.xaxis.set_ticks_position('top')
#cbar15.ax.xaxis.set_label_position('top')
#
# Grab the valid time from the netcdf metadata and put a label on the picture


time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

if f.satellite_id == "GOES-17":
    time_string = 'GOES-17 upper-level water vapor (ABI ch 8)   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
elif f.satellite_id == "GOES-18":
    time_string = 'GOES-18 upper-level water vapor (ABI ch 8)   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
else:
    time_string = 'GOES-West upper-level water vapor (ABI ch 8)   %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
print(time_string)

from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

##2017/065 20:04:00:30
text = ax.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text.set_path_effects(outline_effect)
#
text2 = ax2.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text2.set_path_effects(outline_effect)

text3 = ax3.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='yellow', fontsize='large', weight='bold')

text3.set_path_effects(outline_effect)

#text4 = ax4.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax4.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text4.set_path_effects(outline_effect)
#
#text8 = ax8.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax8.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text8.set_path_effects(outline_effect)
#
#text9 = ax9.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax9.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text9.set_path_effects(outline_effect)
#
text13 = ax13.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text13.set_path_effects(outline_effect)

text14 = ax14.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text14.set_path_effects(outline_effect)
#
#text15 = ax15.text(0.50, 0.97, time_string,
#    horizontalalignment='center', transform = ax15.transAxes,
#    color='yellow', fontsize='large', weight='bold')
#
#text15.set_path_effects(outline_effect)
#
#
# set file names for each file. Paths should be changed to reflect
# your setup. I'm using jpg because the file sizes are smaller, but
# change the extension to png or something else to get other file types


filename1="/whirlwind/goes17/wvh/westcoast/"+dt+"_westcoast.jpg"
filename2="/whirlwind/goes17/wvh/hi/"+dt+"_hi.jpg"
filename3="/whirlwind/goes17/wvh/conus/"+dt+"_conus.jpg"
#filename4="/whirlwind/goes17/wvh/ne/"+dt+"_ne.jpg"
#filename8="/whirlwind/goes17/wvh/gulf/"+dt+"_gulf.jpg"
#filename9="/whirlwind/goes17/wvh/full/"+dt+"_full.jpg"
filename13="/whirlwind/goes17/wvh/sw/"+dt+"_sw.jpg"
filename14="/whirlwind/goes17/wvh/nw/"+dt+"_nw.jpg"
#filename15="/whirlwind/goes17/wvh/gtlakes/"+dt+"_gtlakes.jpg"
filename0="/whirlwind/goes17/wvh/westcoast/latest_westcoast_1s.jpg"
filename20="/whirlwind/goes17/wvh/hi/latest_hi_1s.jpg"
filename30="/whirlwind/goes17/wvh/conus/latest_conus_1s.jpg"
#filename40="/whirlwind/goes17/wvh/ne/latest_ne_1s.jpg"
#filename80="/whirlwind/goes17/wvh/gulf/latest_gulf_1s.jpg"
filename130="/whirlwind/goes17/wvh/sw/latest_sw_1s.jpg"
filename140="/whirlwind/goes17/wvh/nw/latest_nw_1s.jpg"
#filename150="/whirlwind/goes17/wvh/gtlakes/latest_gtlakes_1s.jpg"

isize = fig.get_size_inches()*fig.dpi
ysize=int(isize[1]*0.77)
fig.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
isize = fig2.get_size_inches()*fig2.dpi
ysize=int(isize[1]*0.77)
print("hi isize,ysize",isize,ysize)
#fig2.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig2.figimage(aoslogo,  0, 1029   , zorder=10)
isize = fig3.get_size_inches()*fig3.dpi
ysize=int(isize[1]*0.77)
print("conus isize,ysize",isize,ysize)
#fig3.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
fig3.figimage(aoslogo,  0, 874   , zorder=10)
#isize = fig4.get_size_inches()*fig4.dpi
#ysize=int(isize[1]*0.97)
#fig4.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#isize = fig8.get_size_inches()*fig8.dpi
#ysize=int(isize[1]*0.97)
#fig8.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#isize = fig9.get_size_inches()*fig9.dpi
#ysize=int(isize[1]*0.97)
#fig9.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
isize = fig13.get_size_inches()*fig13.dpi
ysize=int(isize[1]*0.77)
fig13.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
isize = fig14.get_size_inches()*fig14.dpi
ysize=int(isize[1]*0.77)
fig14.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#isize = fig15.get_size_inches()*fig15.dpi
#ysize=int(isize[1]*0.97)
#fig15.figimage(aoslogo,  0, ysize - aoslogoheight   , zorder=10)
#
fig.savefig(filename1, bbox_inches='tight', pad_inches=0)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
#fig3.savefig(filename3, bbox_inches=None, pad_inches=0)
#fig3.savefig("test.png")
#fig4.savefig(filename4, bbox_inches='tight', pad_inches=0)
#fig8.savefig(filename8, bbox_inches='tight', pad_inches=0)
#fig9.savefig(filename9, bbox_inches='tight', pad_inches=0)
fig13.savefig(filename13, bbox_inches='tight', pad_inches=0)
fig14.savefig(filename14, bbox_inches='tight', pad_inches=0)
#fig15.savefig(filename15, bbox_inches='tight', pad_inches=0)
#fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
fig0.savefig(filename0, bbox_inches='tight', pad_inches=0)
fig20.savefig(filename20, bbox_inches='tight', pad_inches=0)
fig30.savefig(filename30, bbox_inches='tight', pad_inches=0)
#fig40.savefig(filename40, bbox_inches='tight', pad_inches=0)
#fig80.savefig(filename80, bbox_inches='tight', pad_inches=0)
fig130.savefig(filename130, bbox_inches='tight', pad_inches=0)
fig140.savefig(filename140, bbox_inches='tight', pad_inches=0)
#fig150.savefig(filename150, bbox_inches='tight', pad_inches=0)

#quit()

# below is my image cycling stuff for my hanis looper

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


silentremove("/whirlwind/goes17/wvh/westcoast/latest_westcoast_72.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_71.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_72.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_70.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_71.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_69.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_70.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_68.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_69.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_67.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_68.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_66.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_67.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_65.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_66.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_64.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_65.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_63.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_64.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_62.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_63.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_61.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_62.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_60.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_61.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_59.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_60.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_58.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_59.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_57.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_58.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_56.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_57.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_55.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_56.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_54.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_55.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_53.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_54.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_52.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_53.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_51.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_52.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_50.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_51.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_49.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_50.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_48.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_49.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_47.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_48.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_46.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_47.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_45.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_46.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_44.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_45.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_43.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_44.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_42.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_43.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_41.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_42.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_40.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_41.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_39.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_40.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_38.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_39.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_37.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_38.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_36.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_37.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_35.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_36.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_34.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_35.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_33.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_34.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_32.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_33.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_31.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_32.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_30.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_31.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_29.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_30.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_28.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_29.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_27.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_28.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_26.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_27.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_25.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_26.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_24.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_25.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_23.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_24.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_22.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_23.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_21.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_22.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_20.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_21.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_19.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_20.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_18.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_19.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_17.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_18.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_16.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_17.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_15.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_16.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_14.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_15.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_13.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_14.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_12.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_13.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_11.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_12.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_10.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_11.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_9.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_10.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_8.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_9.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_7.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_8.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_6.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_7.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_5.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_6.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_4.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_5.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_3.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_4.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_2.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_3.jpg")
silentrename("/whirlwind/goes17/wvh/westcoast/latest_westcoast_1.jpg", "/whirlwind/goes17/wvh/westcoast/latest_westcoast_2.jpg")

shutil.copy(filename1, "/whirlwind/goes17/wvh/westcoast/latest_westcoast_1.jpg")


silentremove("/whirlwind/goes17/wvh/hi/latest_hi_72.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_71.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_72.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_70.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_71.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_69.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_70.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_68.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_69.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_67.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_68.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_66.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_67.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_65.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_66.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_64.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_65.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_63.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_64.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_62.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_63.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_61.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_62.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_60.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_61.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_59.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_60.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_58.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_59.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_57.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_58.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_56.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_57.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_55.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_56.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_54.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_55.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_53.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_54.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_52.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_53.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_51.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_52.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_50.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_51.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_49.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_50.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_48.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_49.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_47.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_48.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_46.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_47.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_45.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_46.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_44.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_45.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_43.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_44.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_42.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_43.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_41.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_42.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_40.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_41.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_39.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_40.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_38.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_39.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_37.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_38.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_36.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_37.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_35.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_36.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_34.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_35.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_33.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_34.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_32.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_33.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_31.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_32.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_30.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_31.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_29.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_30.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_28.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_29.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_27.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_28.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_26.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_27.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_25.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_26.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_24.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_25.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_23.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_24.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_22.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_23.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_21.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_22.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_20.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_21.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_19.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_20.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_18.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_19.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_17.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_18.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_16.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_17.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_15.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_16.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_14.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_15.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_13.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_14.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_12.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_13.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_11.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_12.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_10.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_11.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_9.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_10.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_8.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_9.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_7.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_8.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_6.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_7.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_5.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_6.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_4.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_5.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_3.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_4.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_2.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_3.jpg")
silentrename("/whirlwind/goes17/wvh/hi/latest_hi_1.jpg", "/whirlwind/goes17/wvh/hi/latest_hi_2.jpg")

shutil.copy(filename2, "/whirlwind/goes17/wvh/hi/latest_hi_1.jpg")

silentremove("/whirlwind/goes17/wvh/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_71.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_70.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_71.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_69.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_70.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_68.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_69.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_67.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_68.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_66.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_67.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_65.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_66.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_64.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_65.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_63.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_64.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_62.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_63.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_61.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_62.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_60.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_61.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_59.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_60.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_58.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_59.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_57.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_58.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_56.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_57.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_55.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_56.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_54.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_55.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_53.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_54.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_52.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_53.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_51.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_52.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_50.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_51.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_49.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_50.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_48.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_49.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_47.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_48.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_46.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_47.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_45.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_46.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_44.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_45.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_43.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_44.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_42.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_43.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_41.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_42.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_40.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_41.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_39.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_40.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_38.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_39.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_37.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_38.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_36.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_37.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_35.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_36.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_34.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_35.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_33.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_34.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_32.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_33.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_31.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_32.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_30.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_31.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_29.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_30.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_28.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_29.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_27.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_28.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_26.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_27.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_25.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_26.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_24.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_25.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_23.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_24.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_22.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_23.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_21.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_22.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_20.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_21.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_19.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_20.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_18.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_19.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_17.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_18.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_16.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_17.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_15.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_16.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_14.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_15.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_13.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_14.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_12.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_13.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_11.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_12.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_10.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_11.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_9.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_10.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_8.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_9.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_7.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_8.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_6.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_7.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_5.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_6.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_4.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_5.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_3.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_4.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_2.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_3.jpg")
silentrename("/whirlwind/goes17/wvh/conus/latest_conus_1.jpg", "/whirlwind/goes17/wvh/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes17/wvh/conus/latest_conus_1.jpg")

silentremove("/whirlwind/goes17/wvh/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_71.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_70.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_71.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_69.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_70.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_68.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_69.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_67.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_68.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_66.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_67.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_65.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_66.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_64.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_65.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_63.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_64.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_62.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_63.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_61.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_62.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_60.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_61.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_59.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_60.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_58.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_59.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_57.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_58.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_56.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_57.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_55.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_56.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_54.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_55.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_53.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_54.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_52.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_53.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_51.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_52.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_50.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_51.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_49.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_50.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_48.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_49.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_47.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_48.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_46.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_47.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_45.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_46.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_44.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_45.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_43.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_44.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_42.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_43.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_41.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_42.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_40.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_41.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_39.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_40.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_38.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_39.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_37.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_38.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_36.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_37.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_35.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_36.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_34.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_35.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_33.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_34.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_32.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_33.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_31.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_32.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_30.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_31.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_29.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_30.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_28.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_29.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_27.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_28.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_26.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_27.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_25.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_26.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_24.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_25.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_23.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_24.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_22.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_23.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_21.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_22.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_20.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_21.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_19.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_20.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_18.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_19.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_17.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_18.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_16.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_17.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_15.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_16.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_14.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_15.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_13.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_14.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_12.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_13.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_11.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_12.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_10.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_11.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_9.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_10.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_8.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_9.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_7.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_8.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_6.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_7.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_5.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_6.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_4.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_5.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_3.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_4.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_2.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_3.jpg")
silentrename("/whirlwind/goes17/wvh/sw/latest_sw_1.jpg", "/whirlwind/goes17/wvh/sw/latest_sw_2.jpg")

shutil.copy(filename13, "/whirlwind/goes17/wvh/sw/latest_sw_1.jpg")

silentremove("/whirlwind/goes17/wvh/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_71.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_70.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_71.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_69.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_70.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_68.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_69.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_67.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_68.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_66.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_67.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_65.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_66.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_64.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_65.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_63.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_64.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_62.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_63.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_61.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_62.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_60.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_61.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_59.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_60.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_58.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_59.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_57.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_58.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_56.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_57.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_55.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_56.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_54.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_55.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_53.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_54.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_52.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_53.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_51.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_52.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_50.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_51.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_49.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_50.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_48.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_49.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_47.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_48.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_46.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_47.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_45.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_46.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_44.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_45.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_43.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_44.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_42.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_43.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_41.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_42.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_40.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_41.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_39.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_40.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_38.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_39.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_37.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_38.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_36.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_37.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_35.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_36.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_34.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_35.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_33.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_34.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_32.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_33.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_31.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_32.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_30.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_31.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_29.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_30.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_28.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_29.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_27.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_28.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_26.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_27.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_25.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_26.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_24.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_25.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_23.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_24.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_22.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_23.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_21.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_22.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_20.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_21.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_19.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_20.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_18.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_19.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_17.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_18.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_16.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_17.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_15.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_16.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_14.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_15.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_13.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_14.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_12.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_13.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_11.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_12.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_10.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_11.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_9.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_10.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_8.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_9.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_7.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_8.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_6.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_7.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_5.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_6.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_4.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_5.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_3.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_4.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_2.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_3.jpg")
silentrename("/whirlwind/goes17/wvh/nw/latest_nw_1.jpg", "/whirlwind/goes17/wvh/nw/latest_nw_2.jpg")

shutil.copy(filename14, "/whirlwind/goes17/wvh/nw/latest_nw_1.jpg")

