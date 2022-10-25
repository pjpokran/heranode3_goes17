#!/home/poker/miniconda3/envs/goes16_201710/bin/python

import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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


band="14"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

prod_id = "TIRE"
# pull date/time to process from stdin. YYYYMMDDHHmm of file named /weather/data/goes16/TIRE/##/YYYYMMDDHHmm_PAA.nc
#dt="201703051957"
dt = sys.argv[1]

#f = netCDF4.Dataset("/weather/data/goes16/TIRC/"+band+"/"+dt+"_PAA.nc")
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
#a[a==0.] = 1.

#print(np.average(a))
#if np.average(a) > 0.7:
#    quit()

print('np.average is ', np.average(a))
#if np.average(a) < 0.3:
if np.average(a) < 0.0035:
    quit()

# xa = xa / 1000000.0  * 35785831
xa=xa*35.785831
ya=ya*35.785831

print("xa is ",xa)
print("ya is ",ya)

proj_var = f.variables[data_var.grid_mapping]
print ("proj_var",proj_var )

import cartopy.crs as ccrs

# Create a Globe specifying a spherical earth with the correct radius
### OLD VERSION  ###
#globe = ccrs.Globe(ellipse='sphere', semimajor_axis=proj_var.semi_major,
#                   semiminor_axis=proj_var.semi_minor)
#
#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)
### OLD VERSION  ###

globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
proj = ccrs.Geostationary(central_longitude=-75.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')


image_rows=f.product_rows
image_columns=f.product_columns
#wi_image_crop_top=1100
#wi_image_crop_bottom=-6300
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

#
# Northeast sector
ne_image_crop_top=25
ne_image_crop_bottom=-725
ne_image_crop_left=1275
ne_image_crop_right=-125


ne_image_size_y=(image_rows+ne_image_crop_bottom-ne_image_crop_top)
ne_image_size_x=(image_columns+ne_image_crop_right-ne_image_crop_left)

ne_image_size_x=float(ne_image_size_x)/100.
ne_image_size_y=float(ne_image_size_y)/100.



# Fullres Southern WI sector

#swi_image_crop_top=1900
#swi_image_crop_bottom=-6400
#swi_image_crop_top=2000
#swi_image_crop_bottom=-2000
#swi_image_crop_left=1500
#swi_image_crop_right=-1500
swi_image_crop_top=48
swi_image_crop_bottom=-1153
swi_image_crop_left=1040
swi_image_crop_right=-990

swi_image_size_y=(image_rows+swi_image_crop_bottom-swi_image_crop_top)
swi_image_size_x=(image_columns+swi_image_crop_right-swi_image_crop_left)

swi_image_size_x=float(swi_image_size_x)/16.25
swi_image_size_y=float(swi_image_size_y)/16.25

# Fullres Colorado sector

#co_image_crop_top=2600
#co_image_crop_bottom=-5350
co_image_crop_top=325
co_image_crop_bottom=-990
co_image_crop_left=450
co_image_crop_right=-1700

co_image_size_y=(image_rows+co_image_crop_bottom-co_image_crop_top)
co_image_size_x=(image_columns+co_image_crop_right-co_image_crop_left)

co_image_size_x=float(co_image_size_x)/16.25
co_image_size_y=float(co_image_size_y)/16.25

# Fullres Florida sector

#fl_image_crop_top=5000
#fl_image_crop_bottom=-2800
fl_image_crop_top=675
fl_image_crop_bottom=-475
fl_image_crop_left=1200
fl_image_crop_right=-875

fl_image_size_y=(image_rows+fl_image_crop_bottom-fl_image_crop_top)
fl_image_size_x=(image_columns+fl_image_crop_right-fl_image_crop_left)

fl_image_size_x=float(fl_image_size_x)/16.25
fl_image_size_y=float(fl_image_size_y)/16.25

# Gulf of Mexico sector

#gulf_image_crop_top=4000
#gulf_image_crop_bottom=-500

gulf_image_crop_top=428
gulf_image_crop_bottom=-5
gulf_image_crop_left=649
gulf_image_crop_right=-76


gulf_image_size_y=(image_rows+gulf_image_crop_bottom-gulf_image_crop_top)
gulf_image_size_x=(image_columns+gulf_image_crop_right-gulf_image_crop_left)

gulf_image_size_x=float(gulf_image_size_x)/60.
gulf_image_size_y=float(gulf_image_size_y)/60.

# Alex OK/KS/MO subsection
alex_image_crop_top=500
alex_image_crop_bottom=-500
alex_image_crop_left=750
alex_image_crop_right=-750

alex_image_size_y=(image_rows+alex_image_crop_bottom-alex_image_crop_top)
alex_image_size_x=(image_columns+alex_image_crop_right-alex_image_crop_left)

alex_image_size_x=float(alex_image_size_x)/16.25
alex_image_size_y=float(alex_image_size_y)/16.25

sw_image_crop_top=300
sw_image_crop_bottom=-750
sw_image_crop_left=1
sw_image_crop_right=-1761


sw_image_size_y=(image_rows+sw_image_crop_bottom-sw_image_crop_top)
sw_image_size_x=(image_columns+sw_image_crop_right-sw_image_crop_left)

print("sw image size")
print(sw_image_size_x, sw_image_size_y)

sw_image_size_x=float(sw_image_size_x)/10.
sw_image_size_y=float(sw_image_size_y)/10.

nw_image_crop_top=20
nw_image_crop_bottom=-1080
nw_image_crop_left=101
nw_image_crop_right=-1701


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



print("create figures")


# Create a new figure with size 10" by 10"
# Wisconsin Crop
#fig = plt.figure(figsize=(wi_image_size_x,wi_image_size_y),dpi=80.)
fig = plt.figure(figsize=(16.,10.17))
# Midwest crop
#fig2 = plt.figure(figsize=(mw_image_size_x,mw_image_size_y),dpi=160.)
fig2 = plt.figure(figsize=(18.,12.62))
# CONUS
fig3 = plt.figure(figsize=(20.,11.987))
# Northeast crop
fig4 = plt.figure(figsize=(18.,12.27))
# Wisconsin fullres crop
#fig5 = plt.figure(figsize=(swi_image_size_x,swi_image_size_y),dpi=83.)
fig5 = plt.figure(figsize=(18.,13.31))
# Colorado fullres crop
#fig6 = plt.figure(figsize=(co_image_size_x,co_image_size_y),dpi=83.)
fig6 = plt.figure(figsize=(18.0650,9.5486))
# Florida fullres crop
#fig7 = plt.figure(figsize=(fl_image_size_x,fl_image_size_y),dpi=83.)
fig7 = plt.figure(figsize=(21.9355,18.065))
# Gulf of Mexico region
fig8 = plt.figure(figsize=(gulf_image_size_x,gulf_image_size_y),dpi=83.)
fig8 = plt.figure(figsize=(30.,18.026))
# Full Res
fig9 = plt.figure(figsize=(image_columns/80.,image_rows/80.))
# Alex chase crop
fig10 = plt.figure(figsize=(alex_image_size_x,alex_image_size_y),dpi=83.)
# Southwest US
fig13 = plt.figure(figsize=(18.,10.98))
# Northwest US
fig14 = plt.figure(figsize=(18.,10.32))
# GreatLakes
fig15 = plt.figure(figsize=(18.,10.73))
print("create axes")
# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.outline_patch.set_edgecolor('none')
ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
ax2.outline_patch.set_edgecolor('none')
ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
ax3.outline_patch.set_edgecolor('none')
ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
ax4.outline_patch.set_edgecolor('none')
ax5 = fig5.add_subplot(1, 1, 1, projection=proj)
ax5.set_extent((-91.3,-87.0,40.8,45.0))
ax5.outline_patch.set_edgecolor('none')
ax6 = fig6.add_subplot(1, 1, 1, projection=proj)
ax6.outline_patch.set_edgecolor('none')
ax7 = fig7.add_subplot(1, 1, 1, projection=proj)
ax7.outline_patch.set_edgecolor('none')
ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
ax8.outline_patch.set_edgecolor('none')
ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
ax9.outline_patch.set_edgecolor('none')
ax10 = fig10.add_subplot(1, 1, 1, projection=proj)
ax10.outline_patch.set_edgecolor('none')
ax13 = fig13.add_subplot(1, 1, 1, projection=proj)
ax13.outline_patch.set_edgecolor('none')
ax14 = fig14.add_subplot(1, 1, 1, projection=proj)
ax14.outline_patch.set_edgecolor('none')
ax15 = fig15.add_subplot(1, 1, 1, projection=proj)
ax15.outline_patch.set_edgecolor('none')

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

print("im ax")
im = ax.imshow(a[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right], extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
print("im ax2")
im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
print("im ax3")
im = ax3.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax4.imshow(a[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right], extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax5.imshow(a[swi_image_crop_top:swi_image_crop_bottom,swi_image_crop_left:swi_image_crop_right], extent=(xa[swi_image_crop_left],xa[swi_image_crop_right],ya[swi_image_crop_bottom],ya[swi_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax6.imshow(a[co_image_crop_top:co_image_crop_bottom,co_image_crop_left:co_image_crop_right], extent=(xa[co_image_crop_left],xa[co_image_crop_right],ya[co_image_crop_bottom],ya[co_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax7.imshow(a[fl_image_crop_top:fl_image_crop_bottom,fl_image_crop_left:fl_image_crop_right], extent=(xa[fl_image_crop_left],xa[fl_image_crop_right],ya[fl_image_crop_bottom],ya[fl_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax8.imshow(a[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right], extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax9.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap='Greys', vmin=162., vmax=330.)
im = ax10.imshow(a[alex_image_crop_top:alex_image_crop_bottom,alex_image_crop_left:alex_image_crop_right], extent=(xa[alex_image_crop_left],xa[alex_image_crop_right],ya[alex_image_crop_bottom],ya[alex_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax13.imshow(a[sw_image_crop_top:sw_image_crop_bottom,sw_image_crop_left:sw_image_crop_right], extent=(xa[sw_image_crop_left],xa[sw_image_crop_right],ya[sw_image_crop_bottom],ya[sw_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax14.imshow(a[nw_image_crop_top:nw_image_crop_bottom,nw_image_crop_left:nw_image_crop_right], extent=(xa[nw_image_crop_left],xa[nw_image_crop_right],ya[nw_image_crop_bottom],ya[nw_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)
im = ax15.imshow(a[gtlakes_image_crop_top:gtlakes_image_crop_bottom,gtlakes_image_crop_left:gtlakes_image_crop_right], extent=(xa[gtlakes_image_crop_left],xa[gtlakes_image_crop_right],ya[gtlakes_image_crop_bottom],ya[gtlakes_image_crop_top]), origin='upper',cmap='Greys', vmin=162., vmax=330.)

import cartopy.feature as cfeat
print("coastlines")

ax.coastlines(resolution='50m', color='green')
ax2.coastlines(resolution='50m', color='green')
ax3.coastlines(resolution='50m', color='green')
ax4.coastlines(resolution='50m', color='green')
ax5.coastlines(resolution='50m', color='green')
ax6.coastlines(resolution='50m', color='green')
ax7.coastlines(resolution='50m', color='green')
ax8.coastlines(resolution='50m', color='green')
ax9.coastlines(resolution='50m', color='green')
ax10.coastlines(resolution='50m', color='green')
ax13.coastlines(resolution='50m', color='green')
ax14.coastlines(resolution='50m', color='green')
ax15.coastlines(resolution='50m', color='green')

print("country borders")
# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax5.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax6.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax7.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax10.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax13.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax14.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax15.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

print("state boundaries")
# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

state_boundaries2 = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='10m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
ax3.add_feature(state_boundaries, linestyle=':')
ax4.add_feature(state_boundaries, linestyle=':')
ax5.add_feature(state_boundaries2, linewidth=2)
ax6.add_feature(state_boundaries2, linewidth=2)
ax7.add_feature(state_boundaries2, linewidth=2)
ax8.add_feature(state_boundaries, linestyle=':')
ax9.add_feature(state_boundaries, linestyle=':')
ax10.add_feature(state_boundaries, linestyle=':')
ax13.add_feature(state_boundaries, linestyle=':')
ax14.add_feature(state_boundaries, linestyle=':')
ax15.add_feature(state_boundaries, linestyle=':')

# Redisplay modified figure
#fig
#fig2

from cartopy.io.shapereader import Reader
#fname = '/home/poker/resources/counties.shp'
fname = '/home/poker/resources/cb_2016_us_county_5m.shp'
counties = Reader(fname)

print("counties")
ax5.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
ax6.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
ax7.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')

print("stations")
ax5.plot(-89.4012, 43.0731, 'bo', markersize=3, transform=ccrs.Geodetic())
ax5.text(-89.50, 43.02, 'MSN', transform=ccrs.Geodetic(), color='darkorange')
ax5.plot(-87.9065, 43.0389, 'bo', markersize=3, transform=ccrs.Geodetic())
ax5.text(-88.00, 42.98, 'MKE', transform=ccrs.Geodetic(), color='darkorange')
ax5.plot(-91.2396, 43.8014, 'bo', markersize=3, transform=ccrs.Geodetic())
ax5.text(-91.33, 43.75, 'LSE', transform=ccrs.Geodetic(), color='darkorange')
ax5.plot(-88.0198, 44.5192, 'bo', markersize=3, transform=ccrs.Geodetic())
ax5.text(-88.11, 44.46, 'GRB', transform=ccrs.Geodetic(), color='darkorange')

ax6.plot(-104.9903, 39.7392, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-105.09, 39.68, 'DEN', transform=ccrs.Geodetic(), color='darkorange')
ax6.plot(-105.2705, 40.0150, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-105.37, 39.96, 'BOU', transform=ccrs.Geodetic(), color='darkorange')
ax6.plot(-105.0844, 40.5853, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-105.18, 40.53, 'FNL', transform=ccrs.Geodetic(), color='darkorange')
ax6.plot(-108.5506, 39.0639, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-108.65, 39.01, 'GJT', transform=ccrs.Geodetic(), color='darkorange')
ax6.plot(-104.8214, 38.8339, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-104.92, 38.78, 'COS', transform=ccrs.Geodetic(), color='darkorange')
ax6.plot(-104.6091, 38.2544, 'bo', markersize=3, transform=ccrs.Geodetic())
ax6.text(-104.70, 38.20, 'PUB', transform=ccrs.Geodetic(), color='darkorange')



import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

time_string = 'GOES16 IR (ABI ch 14) valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
print(time_string)

from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

print("text")
#2017/065 20:04:00:30
text = ax.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text.set_path_effects(outline_effect)

text2 = ax2.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text2.set_path_effects(outline_effect)

text3 = ax3.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax3.transAxes,
    color='darkorange', fontsize='large', weight='bold')
text3.set_path_effects(outline_effect)

text4 = ax4.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax4.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text4.set_path_effects(outline_effect)

text5 = ax5.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax5.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text5.set_path_effects(outline_effect)

text6 = ax6.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax6.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text6.set_path_effects(outline_effect)

text7 = ax7.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax7.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text7.set_path_effects(outline_effect)

text8 = ax8.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax8.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text8.set_path_effects(outline_effect)

text9 = ax9.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax9.transAxes,
    color='yellow', fontsize='large', weight='bold')

text9.set_path_effects(outline_effect)

text10 = ax10.text(0.50, 0.95, time_string,
    horizontalalignment='center', transform = ax10.transAxes,
    color='darkorange', fontsize='large', weight='bold')

text10.set_path_effects(outline_effect)

text13 = ax13.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax13.transAxes,
    color='yellow', fontsize='large', weight='bold')

text13.set_path_effects(outline_effect)

text14 = ax14.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax14.transAxes,
    color='yellow', fontsize='large', weight='bold')

text14.set_path_effects(outline_effect)

text15 = ax15.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax15.transAxes,
    color='yellow', fontsize='large', weight='bold')

text15.set_path_effects(outline_effect)



filename1="/whirlwind/goes16/ir/wi/"+dt+"_wi.jpg"
filename2="/whirlwind/goes16/ir/mw/"+dt+"_mw.jpg"
filename3="/whirlwind/goes16/ir/conus/"+dt+"_conus.jpg"
filename4="/whirlwind/goes16/ir/ne/"+dt+"_ne.jpg"
filename5="/whirlwind/goes16/ir/swi/"+dt+"_swi.jpg"
filename6="/whirlwind/goes16/ir/co/"+dt+"_co.jpg"
filename7="/whirlwind/goes16/ir/fl/"+dt+"_fl.jpg"
filename8="/whirlwind/goes16/ir/gulf/"+dt+"_gulf.jpg"
filename9="/whirlwind/goes16/ir/full/"+dt+"_full.jpg"
filename10="/whirlwind/goes16/ir/alex/"+dt+"_alex.jpg"
filename13="/whirlwind/goes16/ir/sw/"+dt+"_sw.jpg"
filename14="/whirlwind/goes16/ir/nw/"+dt+"_nw.jpg"
filename15="/whirlwind/goes16/ir/gtlakes/"+dt+"_gtlakes.jpg"

##fig.figimage(aoslogo,  0, fig.bbox.ymax - aoslogoheight - 26  , zorder=10)
##fig2.figimage(aoslogo,  0, int(fig2.bbox.ymax/2) - aoslogoheight - 26  , zorder=10)
##fig3.figimage(aoslogo,  0, int(fig3.bbox.ymax/2) - aoslogoheight - 30  , zorder=10)
##fig4.figimage(aoslogo,  0, int(fig4.bbox.ymax/.5) - aoslogoheight - 26  , zorder=10)
##fig5.figimage(aoslogo,  0, int(fig5.bbox.ymax*.96386) - aoslogoheight - 34  , zorder=10)
##fig6.figimage(aoslogo,  0, int(fig6.bbox.ymax*.96386) - aoslogoheight - 44  , zorder=10)
##fig7.figimage(aoslogo,  0, int(fig7.bbox.ymax*.96386) - aoslogoheight - 54  , zorder=10)
##fig8.figimage(aoslogo,  0, int(fig8.bbox.ymax*.96386) - aoslogoheight - 46  , zorder=10)
###fig9.figimage(aoslogo,  0, fig9.bbox.ymax - aoslogoheight - 158  , zorder=10)
##print('fig9.bbox.ymax,aoslogoheight = ',fig9.bbox.ymax,aoslogoheight)
##fig9.figimage(aoslogo,  0, fig9.bbox.ymax - 370, zorder=10)
##fig10.figimage(aoslogo,  0, int(fig10.bbox.ymax*.96386) - aoslogoheight - 26  , zorder=10)
isize = fig.get_size_inches()*fig.dpi
ysize=int(isize[1]*0.97)
#fig.figimage(aoslogo,  0, int(fig.bbox.ymax)-aoslogoheight, zorder=10)
fig.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig2.get_size_inches()*fig2.dpi
ysize=int(isize[1]*0.97)
fig2.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig3.get_size_inches()*fig3.dpi
ysize=int(isize[1]*0.97)
fig3.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig4.get_size_inches()*fig4.dpi
ysize=int(isize[1]*0.97)
fig4.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig5.get_size_inches()*fig5.dpi
ysize=int(isize[1]*0.97)
fig5.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig6.get_size_inches()*fig6.dpi
ysize=int(isize[1]*0.97)
fig6.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig7.get_size_inches()*fig7.dpi
ysize=int(isize[1]*0.97)
fig7.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig8.get_size_inches()*fig8.dpi
ysize=int(isize[1]*0.97)
fig8.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig9.get_size_inches()*fig9.dpi
ysize=int(isize[1]*0.97)
#fig9.figimage(aoslogo,  0, fig9.bbox.ymax - aoslogoheight - 158  , zorder=10)
fig9.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)
isize = fig10.get_size_inches()*fig10.dpi
ysize=int(isize[1]*0.97)
fig10.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)

isize = fig13.get_size_inches()*fig13.dpi
ysize=int(isize[1]*0.97)
fig13.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)

isize = fig14.get_size_inches()*fig14.dpi
ysize=int(isize[1]*0.97)
fig14.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)

isize = fig15.get_size_inches()*fig15.dpi
ysize=int(isize[1]*0.97)
fig15.figimage(aoslogo,  0, ysize-aoslogoheight, zorder=10)

fig.savefig(filename1, bbox_inches='tight', pad_inches=0)
fig2.savefig(filename2, bbox_inches='tight', pad_inches=0)
#fig2.savefig(filename2jpg, bbox_inches='tight')
fig3.savefig(filename3, bbox_inches='tight', pad_inches=0)
fig4.savefig(filename4, bbox_inches='tight', pad_inches=0)
fig5.savefig(filename5, bbox_inches='tight', pad_inches=0)
fig6.savefig(filename6, bbox_inches='tight', pad_inches=0)
fig7.savefig(filename7, bbox_inches='tight', pad_inches=0)
fig8.savefig(filename8, bbox_inches='tight', pad_inches=0)
fig9.savefig(filename9, bbox_inches='tight', pad_inches=0)
fig10.savefig(filename10, bbox_inches='tight', pad_inches=0)
fig13.savefig(filename13, bbox_inches='tight', pad_inches=0)
fig14.savefig(filename14, bbox_inches='tight', pad_inches=0)
fig15.savefig(filename15, bbox_inches='tight', pad_inches=0)

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

print("rename wi")
silentremove("/whirlwind/goes16/ir/wi/latest_wi_72.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_71.jpg", "/whirlwind/goes16/ir/wi/latest_wi_72.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_70.jpg", "/whirlwind/goes16/ir/wi/latest_wi_71.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_69.jpg", "/whirlwind/goes16/ir/wi/latest_wi_70.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_68.jpg", "/whirlwind/goes16/ir/wi/latest_wi_69.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_67.jpg", "/whirlwind/goes16/ir/wi/latest_wi_68.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_66.jpg", "/whirlwind/goes16/ir/wi/latest_wi_67.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_65.jpg", "/whirlwind/goes16/ir/wi/latest_wi_66.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_64.jpg", "/whirlwind/goes16/ir/wi/latest_wi_65.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_63.jpg", "/whirlwind/goes16/ir/wi/latest_wi_64.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_62.jpg", "/whirlwind/goes16/ir/wi/latest_wi_63.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_61.jpg", "/whirlwind/goes16/ir/wi/latest_wi_62.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_60.jpg", "/whirlwind/goes16/ir/wi/latest_wi_61.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_59.jpg", "/whirlwind/goes16/ir/wi/latest_wi_60.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_58.jpg", "/whirlwind/goes16/ir/wi/latest_wi_59.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_57.jpg", "/whirlwind/goes16/ir/wi/latest_wi_58.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_56.jpg", "/whirlwind/goes16/ir/wi/latest_wi_57.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_55.jpg", "/whirlwind/goes16/ir/wi/latest_wi_56.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_54.jpg", "/whirlwind/goes16/ir/wi/latest_wi_55.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_53.jpg", "/whirlwind/goes16/ir/wi/latest_wi_54.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_52.jpg", "/whirlwind/goes16/ir/wi/latest_wi_53.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_51.jpg", "/whirlwind/goes16/ir/wi/latest_wi_52.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_50.jpg", "/whirlwind/goes16/ir/wi/latest_wi_51.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_49.jpg", "/whirlwind/goes16/ir/wi/latest_wi_50.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_48.jpg", "/whirlwind/goes16/ir/wi/latest_wi_49.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_47.jpg", "/whirlwind/goes16/ir/wi/latest_wi_48.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_46.jpg", "/whirlwind/goes16/ir/wi/latest_wi_47.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_45.jpg", "/whirlwind/goes16/ir/wi/latest_wi_46.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_44.jpg", "/whirlwind/goes16/ir/wi/latest_wi_45.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_43.jpg", "/whirlwind/goes16/ir/wi/latest_wi_44.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_42.jpg", "/whirlwind/goes16/ir/wi/latest_wi_43.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_41.jpg", "/whirlwind/goes16/ir/wi/latest_wi_42.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_40.jpg", "/whirlwind/goes16/ir/wi/latest_wi_41.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_39.jpg", "/whirlwind/goes16/ir/wi/latest_wi_40.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_38.jpg", "/whirlwind/goes16/ir/wi/latest_wi_39.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_37.jpg", "/whirlwind/goes16/ir/wi/latest_wi_38.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_36.jpg", "/whirlwind/goes16/ir/wi/latest_wi_37.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_35.jpg", "/whirlwind/goes16/ir/wi/latest_wi_36.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_34.jpg", "/whirlwind/goes16/ir/wi/latest_wi_35.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_33.jpg", "/whirlwind/goes16/ir/wi/latest_wi_34.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_32.jpg", "/whirlwind/goes16/ir/wi/latest_wi_33.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_31.jpg", "/whirlwind/goes16/ir/wi/latest_wi_32.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_30.jpg", "/whirlwind/goes16/ir/wi/latest_wi_31.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_29.jpg", "/whirlwind/goes16/ir/wi/latest_wi_30.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_28.jpg", "/whirlwind/goes16/ir/wi/latest_wi_29.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_27.jpg", "/whirlwind/goes16/ir/wi/latest_wi_28.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_26.jpg", "/whirlwind/goes16/ir/wi/latest_wi_27.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_25.jpg", "/whirlwind/goes16/ir/wi/latest_wi_26.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_24.jpg", "/whirlwind/goes16/ir/wi/latest_wi_25.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_23.jpg", "/whirlwind/goes16/ir/wi/latest_wi_24.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_22.jpg", "/whirlwind/goes16/ir/wi/latest_wi_23.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_21.jpg", "/whirlwind/goes16/ir/wi/latest_wi_22.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_20.jpg", "/whirlwind/goes16/ir/wi/latest_wi_21.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_19.jpg", "/whirlwind/goes16/ir/wi/latest_wi_20.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_18.jpg", "/whirlwind/goes16/ir/wi/latest_wi_19.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_17.jpg", "/whirlwind/goes16/ir/wi/latest_wi_18.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_16.jpg", "/whirlwind/goes16/ir/wi/latest_wi_17.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_15.jpg", "/whirlwind/goes16/ir/wi/latest_wi_16.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_14.jpg", "/whirlwind/goes16/ir/wi/latest_wi_15.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_13.jpg", "/whirlwind/goes16/ir/wi/latest_wi_14.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_12.jpg", "/whirlwind/goes16/ir/wi/latest_wi_13.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_11.jpg", "/whirlwind/goes16/ir/wi/latest_wi_12.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_10.jpg", "/whirlwind/goes16/ir/wi/latest_wi_11.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_9.jpg", "/whirlwind/goes16/ir/wi/latest_wi_10.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_8.jpg", "/whirlwind/goes16/ir/wi/latest_wi_9.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_7.jpg", "/whirlwind/goes16/ir/wi/latest_wi_8.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_6.jpg", "/whirlwind/goes16/ir/wi/latest_wi_7.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_5.jpg", "/whirlwind/goes16/ir/wi/latest_wi_6.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_4.jpg", "/whirlwind/goes16/ir/wi/latest_wi_5.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_3.jpg", "/whirlwind/goes16/ir/wi/latest_wi_4.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_2.jpg", "/whirlwind/goes16/ir/wi/latest_wi_3.jpg")
silentrename("/whirlwind/goes16/ir/wi/latest_wi_1.jpg", "/whirlwind/goes16/ir/wi/latest_wi_2.jpg")

shutil.copy(filename1, "/whirlwind/goes16/ir/wi/latest_wi_1.jpg")


print("rename mw")
silentremove("/whirlwind/goes16/ir/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_71.jpg", "/whirlwind/goes16/ir/mw/latest_mw_72.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_70.jpg", "/whirlwind/goes16/ir/mw/latest_mw_71.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_69.jpg", "/whirlwind/goes16/ir/mw/latest_mw_70.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_68.jpg", "/whirlwind/goes16/ir/mw/latest_mw_69.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_67.jpg", "/whirlwind/goes16/ir/mw/latest_mw_68.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_66.jpg", "/whirlwind/goes16/ir/mw/latest_mw_67.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_65.jpg", "/whirlwind/goes16/ir/mw/latest_mw_66.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_64.jpg", "/whirlwind/goes16/ir/mw/latest_mw_65.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_63.jpg", "/whirlwind/goes16/ir/mw/latest_mw_64.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_62.jpg", "/whirlwind/goes16/ir/mw/latest_mw_63.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_61.jpg", "/whirlwind/goes16/ir/mw/latest_mw_62.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_60.jpg", "/whirlwind/goes16/ir/mw/latest_mw_61.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_59.jpg", "/whirlwind/goes16/ir/mw/latest_mw_60.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_58.jpg", "/whirlwind/goes16/ir/mw/latest_mw_59.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_57.jpg", "/whirlwind/goes16/ir/mw/latest_mw_58.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_56.jpg", "/whirlwind/goes16/ir/mw/latest_mw_57.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_55.jpg", "/whirlwind/goes16/ir/mw/latest_mw_56.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_54.jpg", "/whirlwind/goes16/ir/mw/latest_mw_55.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_53.jpg", "/whirlwind/goes16/ir/mw/latest_mw_54.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_52.jpg", "/whirlwind/goes16/ir/mw/latest_mw_53.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_51.jpg", "/whirlwind/goes16/ir/mw/latest_mw_52.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_50.jpg", "/whirlwind/goes16/ir/mw/latest_mw_51.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_49.jpg", "/whirlwind/goes16/ir/mw/latest_mw_50.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_48.jpg", "/whirlwind/goes16/ir/mw/latest_mw_49.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_47.jpg", "/whirlwind/goes16/ir/mw/latest_mw_48.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_46.jpg", "/whirlwind/goes16/ir/mw/latest_mw_47.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_45.jpg", "/whirlwind/goes16/ir/mw/latest_mw_46.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_44.jpg", "/whirlwind/goes16/ir/mw/latest_mw_45.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_43.jpg", "/whirlwind/goes16/ir/mw/latest_mw_44.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_42.jpg", "/whirlwind/goes16/ir/mw/latest_mw_43.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_41.jpg", "/whirlwind/goes16/ir/mw/latest_mw_42.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_40.jpg", "/whirlwind/goes16/ir/mw/latest_mw_41.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_39.jpg", "/whirlwind/goes16/ir/mw/latest_mw_40.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_38.jpg", "/whirlwind/goes16/ir/mw/latest_mw_39.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_37.jpg", "/whirlwind/goes16/ir/mw/latest_mw_38.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_36.jpg", "/whirlwind/goes16/ir/mw/latest_mw_37.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_35.jpg", "/whirlwind/goes16/ir/mw/latest_mw_36.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_34.jpg", "/whirlwind/goes16/ir/mw/latest_mw_35.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_33.jpg", "/whirlwind/goes16/ir/mw/latest_mw_34.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_32.jpg", "/whirlwind/goes16/ir/mw/latest_mw_33.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_31.jpg", "/whirlwind/goes16/ir/mw/latest_mw_32.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_30.jpg", "/whirlwind/goes16/ir/mw/latest_mw_31.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_29.jpg", "/whirlwind/goes16/ir/mw/latest_mw_30.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_28.jpg", "/whirlwind/goes16/ir/mw/latest_mw_29.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_27.jpg", "/whirlwind/goes16/ir/mw/latest_mw_28.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_26.jpg", "/whirlwind/goes16/ir/mw/latest_mw_27.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_25.jpg", "/whirlwind/goes16/ir/mw/latest_mw_26.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_24.jpg", "/whirlwind/goes16/ir/mw/latest_mw_25.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_23.jpg", "/whirlwind/goes16/ir/mw/latest_mw_24.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_22.jpg", "/whirlwind/goes16/ir/mw/latest_mw_23.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_21.jpg", "/whirlwind/goes16/ir/mw/latest_mw_22.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_20.jpg", "/whirlwind/goes16/ir/mw/latest_mw_21.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_19.jpg", "/whirlwind/goes16/ir/mw/latest_mw_20.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_18.jpg", "/whirlwind/goes16/ir/mw/latest_mw_19.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_17.jpg", "/whirlwind/goes16/ir/mw/latest_mw_18.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_16.jpg", "/whirlwind/goes16/ir/mw/latest_mw_17.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_15.jpg", "/whirlwind/goes16/ir/mw/latest_mw_16.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_14.jpg", "/whirlwind/goes16/ir/mw/latest_mw_15.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_13.jpg", "/whirlwind/goes16/ir/mw/latest_mw_14.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_12.jpg", "/whirlwind/goes16/ir/mw/latest_mw_13.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_11.jpg", "/whirlwind/goes16/ir/mw/latest_mw_12.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_10.jpg", "/whirlwind/goes16/ir/mw/latest_mw_11.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_9.jpg", "/whirlwind/goes16/ir/mw/latest_mw_10.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_8.jpg", "/whirlwind/goes16/ir/mw/latest_mw_9.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_7.jpg", "/whirlwind/goes16/ir/mw/latest_mw_8.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_6.jpg", "/whirlwind/goes16/ir/mw/latest_mw_7.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_5.jpg", "/whirlwind/goes16/ir/mw/latest_mw_6.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_4.jpg", "/whirlwind/goes16/ir/mw/latest_mw_5.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_3.jpg", "/whirlwind/goes16/ir/mw/latest_mw_4.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_2.jpg", "/whirlwind/goes16/ir/mw/latest_mw_3.jpg")
silentrename("/whirlwind/goes16/ir/mw/latest_mw_1.jpg", "/whirlwind/goes16/ir/mw/latest_mw_2.jpg")

shutil.copy(filename2, "/whirlwind/goes16/ir/mw/latest_mw_1.jpg")


print("rename conus")
silentremove("/whirlwind/goes16/ir/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_71.jpg", "/whirlwind/goes16/ir/conus/latest_conus_72.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_70.jpg", "/whirlwind/goes16/ir/conus/latest_conus_71.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_69.jpg", "/whirlwind/goes16/ir/conus/latest_conus_70.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_68.jpg", "/whirlwind/goes16/ir/conus/latest_conus_69.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_67.jpg", "/whirlwind/goes16/ir/conus/latest_conus_68.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_66.jpg", "/whirlwind/goes16/ir/conus/latest_conus_67.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_65.jpg", "/whirlwind/goes16/ir/conus/latest_conus_66.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_64.jpg", "/whirlwind/goes16/ir/conus/latest_conus_65.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_63.jpg", "/whirlwind/goes16/ir/conus/latest_conus_64.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_62.jpg", "/whirlwind/goes16/ir/conus/latest_conus_63.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_61.jpg", "/whirlwind/goes16/ir/conus/latest_conus_62.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_60.jpg", "/whirlwind/goes16/ir/conus/latest_conus_61.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_59.jpg", "/whirlwind/goes16/ir/conus/latest_conus_60.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_58.jpg", "/whirlwind/goes16/ir/conus/latest_conus_59.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_57.jpg", "/whirlwind/goes16/ir/conus/latest_conus_58.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_56.jpg", "/whirlwind/goes16/ir/conus/latest_conus_57.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_55.jpg", "/whirlwind/goes16/ir/conus/latest_conus_56.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_54.jpg", "/whirlwind/goes16/ir/conus/latest_conus_55.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_53.jpg", "/whirlwind/goes16/ir/conus/latest_conus_54.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_52.jpg", "/whirlwind/goes16/ir/conus/latest_conus_53.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_51.jpg", "/whirlwind/goes16/ir/conus/latest_conus_52.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_50.jpg", "/whirlwind/goes16/ir/conus/latest_conus_51.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_49.jpg", "/whirlwind/goes16/ir/conus/latest_conus_50.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_48.jpg", "/whirlwind/goes16/ir/conus/latest_conus_49.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_47.jpg", "/whirlwind/goes16/ir/conus/latest_conus_48.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_46.jpg", "/whirlwind/goes16/ir/conus/latest_conus_47.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_45.jpg", "/whirlwind/goes16/ir/conus/latest_conus_46.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_44.jpg", "/whirlwind/goes16/ir/conus/latest_conus_45.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_43.jpg", "/whirlwind/goes16/ir/conus/latest_conus_44.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_42.jpg", "/whirlwind/goes16/ir/conus/latest_conus_43.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_41.jpg", "/whirlwind/goes16/ir/conus/latest_conus_42.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_40.jpg", "/whirlwind/goes16/ir/conus/latest_conus_41.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_39.jpg", "/whirlwind/goes16/ir/conus/latest_conus_40.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_38.jpg", "/whirlwind/goes16/ir/conus/latest_conus_39.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_37.jpg", "/whirlwind/goes16/ir/conus/latest_conus_38.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_36.jpg", "/whirlwind/goes16/ir/conus/latest_conus_37.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_35.jpg", "/whirlwind/goes16/ir/conus/latest_conus_36.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_34.jpg", "/whirlwind/goes16/ir/conus/latest_conus_35.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_33.jpg", "/whirlwind/goes16/ir/conus/latest_conus_34.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_32.jpg", "/whirlwind/goes16/ir/conus/latest_conus_33.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_31.jpg", "/whirlwind/goes16/ir/conus/latest_conus_32.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_30.jpg", "/whirlwind/goes16/ir/conus/latest_conus_31.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_29.jpg", "/whirlwind/goes16/ir/conus/latest_conus_30.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_28.jpg", "/whirlwind/goes16/ir/conus/latest_conus_29.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_27.jpg", "/whirlwind/goes16/ir/conus/latest_conus_28.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_26.jpg", "/whirlwind/goes16/ir/conus/latest_conus_27.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_25.jpg", "/whirlwind/goes16/ir/conus/latest_conus_26.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_24.jpg", "/whirlwind/goes16/ir/conus/latest_conus_25.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_23.jpg", "/whirlwind/goes16/ir/conus/latest_conus_24.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_22.jpg", "/whirlwind/goes16/ir/conus/latest_conus_23.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_21.jpg", "/whirlwind/goes16/ir/conus/latest_conus_22.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_20.jpg", "/whirlwind/goes16/ir/conus/latest_conus_21.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_19.jpg", "/whirlwind/goes16/ir/conus/latest_conus_20.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_18.jpg", "/whirlwind/goes16/ir/conus/latest_conus_19.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_17.jpg", "/whirlwind/goes16/ir/conus/latest_conus_18.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_16.jpg", "/whirlwind/goes16/ir/conus/latest_conus_17.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_15.jpg", "/whirlwind/goes16/ir/conus/latest_conus_16.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_14.jpg", "/whirlwind/goes16/ir/conus/latest_conus_15.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_13.jpg", "/whirlwind/goes16/ir/conus/latest_conus_14.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_12.jpg", "/whirlwind/goes16/ir/conus/latest_conus_13.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_11.jpg", "/whirlwind/goes16/ir/conus/latest_conus_12.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_10.jpg", "/whirlwind/goes16/ir/conus/latest_conus_11.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_9.jpg", "/whirlwind/goes16/ir/conus/latest_conus_10.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_8.jpg", "/whirlwind/goes16/ir/conus/latest_conus_9.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_7.jpg", "/whirlwind/goes16/ir/conus/latest_conus_8.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_6.jpg", "/whirlwind/goes16/ir/conus/latest_conus_7.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_5.jpg", "/whirlwind/goes16/ir/conus/latest_conus_6.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_4.jpg", "/whirlwind/goes16/ir/conus/latest_conus_5.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_3.jpg", "/whirlwind/goes16/ir/conus/latest_conus_4.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_2.jpg", "/whirlwind/goes16/ir/conus/latest_conus_3.jpg")
silentrename("/whirlwind/goes16/ir/conus/latest_conus_1.jpg", "/whirlwind/goes16/ir/conus/latest_conus_2.jpg")

shutil.copy(filename3, "/whirlwind/goes16/ir/conus/latest_conus_1.jpg")

print("rename ne")
# Northeast
silentremove("/whirlwind/goes16/ir/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_71.jpg", "/whirlwind/goes16/ir/ne/latest_ne_72.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_70.jpg", "/whirlwind/goes16/ir/ne/latest_ne_71.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_69.jpg", "/whirlwind/goes16/ir/ne/latest_ne_70.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_68.jpg", "/whirlwind/goes16/ir/ne/latest_ne_69.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_67.jpg", "/whirlwind/goes16/ir/ne/latest_ne_68.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_66.jpg", "/whirlwind/goes16/ir/ne/latest_ne_67.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_65.jpg", "/whirlwind/goes16/ir/ne/latest_ne_66.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_64.jpg", "/whirlwind/goes16/ir/ne/latest_ne_65.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_63.jpg", "/whirlwind/goes16/ir/ne/latest_ne_64.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_62.jpg", "/whirlwind/goes16/ir/ne/latest_ne_63.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_61.jpg", "/whirlwind/goes16/ir/ne/latest_ne_62.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_60.jpg", "/whirlwind/goes16/ir/ne/latest_ne_61.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_59.jpg", "/whirlwind/goes16/ir/ne/latest_ne_60.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_58.jpg", "/whirlwind/goes16/ir/ne/latest_ne_59.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_57.jpg", "/whirlwind/goes16/ir/ne/latest_ne_58.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_56.jpg", "/whirlwind/goes16/ir/ne/latest_ne_57.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_55.jpg", "/whirlwind/goes16/ir/ne/latest_ne_56.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_54.jpg", "/whirlwind/goes16/ir/ne/latest_ne_55.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_53.jpg", "/whirlwind/goes16/ir/ne/latest_ne_54.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_52.jpg", "/whirlwind/goes16/ir/ne/latest_ne_53.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_51.jpg", "/whirlwind/goes16/ir/ne/latest_ne_52.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_50.jpg", "/whirlwind/goes16/ir/ne/latest_ne_51.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_49.jpg", "/whirlwind/goes16/ir/ne/latest_ne_50.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_48.jpg", "/whirlwind/goes16/ir/ne/latest_ne_49.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_47.jpg", "/whirlwind/goes16/ir/ne/latest_ne_48.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_46.jpg", "/whirlwind/goes16/ir/ne/latest_ne_47.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_45.jpg", "/whirlwind/goes16/ir/ne/latest_ne_46.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_44.jpg", "/whirlwind/goes16/ir/ne/latest_ne_45.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_43.jpg", "/whirlwind/goes16/ir/ne/latest_ne_44.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_42.jpg", "/whirlwind/goes16/ir/ne/latest_ne_43.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_41.jpg", "/whirlwind/goes16/ir/ne/latest_ne_42.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_40.jpg", "/whirlwind/goes16/ir/ne/latest_ne_41.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_39.jpg", "/whirlwind/goes16/ir/ne/latest_ne_40.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_38.jpg", "/whirlwind/goes16/ir/ne/latest_ne_39.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_37.jpg", "/whirlwind/goes16/ir/ne/latest_ne_38.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_36.jpg", "/whirlwind/goes16/ir/ne/latest_ne_37.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_35.jpg", "/whirlwind/goes16/ir/ne/latest_ne_36.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_34.jpg", "/whirlwind/goes16/ir/ne/latest_ne_35.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_33.jpg", "/whirlwind/goes16/ir/ne/latest_ne_34.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_32.jpg", "/whirlwind/goes16/ir/ne/latest_ne_33.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_31.jpg", "/whirlwind/goes16/ir/ne/latest_ne_32.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_30.jpg", "/whirlwind/goes16/ir/ne/latest_ne_31.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_29.jpg", "/whirlwind/goes16/ir/ne/latest_ne_30.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_28.jpg", "/whirlwind/goes16/ir/ne/latest_ne_29.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_27.jpg", "/whirlwind/goes16/ir/ne/latest_ne_28.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_26.jpg", "/whirlwind/goes16/ir/ne/latest_ne_27.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_25.jpg", "/whirlwind/goes16/ir/ne/latest_ne_26.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_24.jpg", "/whirlwind/goes16/ir/ne/latest_ne_25.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_23.jpg", "/whirlwind/goes16/ir/ne/latest_ne_24.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_22.jpg", "/whirlwind/goes16/ir/ne/latest_ne_23.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_21.jpg", "/whirlwind/goes16/ir/ne/latest_ne_22.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_20.jpg", "/whirlwind/goes16/ir/ne/latest_ne_21.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_19.jpg", "/whirlwind/goes16/ir/ne/latest_ne_20.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_18.jpg", "/whirlwind/goes16/ir/ne/latest_ne_19.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_17.jpg", "/whirlwind/goes16/ir/ne/latest_ne_18.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_16.jpg", "/whirlwind/goes16/ir/ne/latest_ne_17.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_15.jpg", "/whirlwind/goes16/ir/ne/latest_ne_16.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_14.jpg", "/whirlwind/goes16/ir/ne/latest_ne_15.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_13.jpg", "/whirlwind/goes16/ir/ne/latest_ne_14.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_12.jpg", "/whirlwind/goes16/ir/ne/latest_ne_13.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_11.jpg", "/whirlwind/goes16/ir/ne/latest_ne_12.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_10.jpg", "/whirlwind/goes16/ir/ne/latest_ne_11.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_9.jpg", "/whirlwind/goes16/ir/ne/latest_ne_10.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_8.jpg", "/whirlwind/goes16/ir/ne/latest_ne_9.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_7.jpg", "/whirlwind/goes16/ir/ne/latest_ne_8.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_6.jpg", "/whirlwind/goes16/ir/ne/latest_ne_7.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_5.jpg", "/whirlwind/goes16/ir/ne/latest_ne_6.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_4.jpg", "/whirlwind/goes16/ir/ne/latest_ne_5.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_3.jpg", "/whirlwind/goes16/ir/ne/latest_ne_4.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_2.jpg", "/whirlwind/goes16/ir/ne/latest_ne_3.jpg")
silentrename("/whirlwind/goes16/ir/ne/latest_ne_1.jpg", "/whirlwind/goes16/ir/ne/latest_ne_2.jpg")

shutil.copy(filename4, "/whirlwind/goes16/ir/ne/latest_ne_1.jpg")

print("rename swi")
# Madison close-up
silentremove("/whirlwind/goes16/ir/swi/latest_swi_72.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_71.jpg", "/whirlwind/goes16/ir/swi/latest_swi_72.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_70.jpg", "/whirlwind/goes16/ir/swi/latest_swi_71.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_69.jpg", "/whirlwind/goes16/ir/swi/latest_swi_70.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_68.jpg", "/whirlwind/goes16/ir/swi/latest_swi_69.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_67.jpg", "/whirlwind/goes16/ir/swi/latest_swi_68.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_66.jpg", "/whirlwind/goes16/ir/swi/latest_swi_67.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_65.jpg", "/whirlwind/goes16/ir/swi/latest_swi_66.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_64.jpg", "/whirlwind/goes16/ir/swi/latest_swi_65.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_63.jpg", "/whirlwind/goes16/ir/swi/latest_swi_64.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_62.jpg", "/whirlwind/goes16/ir/swi/latest_swi_63.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_61.jpg", "/whirlwind/goes16/ir/swi/latest_swi_62.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_60.jpg", "/whirlwind/goes16/ir/swi/latest_swi_61.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_59.jpg", "/whirlwind/goes16/ir/swi/latest_swi_60.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_58.jpg", "/whirlwind/goes16/ir/swi/latest_swi_59.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_57.jpg", "/whirlwind/goes16/ir/swi/latest_swi_58.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_56.jpg", "/whirlwind/goes16/ir/swi/latest_swi_57.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_55.jpg", "/whirlwind/goes16/ir/swi/latest_swi_56.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_54.jpg", "/whirlwind/goes16/ir/swi/latest_swi_55.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_53.jpg", "/whirlwind/goes16/ir/swi/latest_swi_54.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_52.jpg", "/whirlwind/goes16/ir/swi/latest_swi_53.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_51.jpg", "/whirlwind/goes16/ir/swi/latest_swi_52.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_50.jpg", "/whirlwind/goes16/ir/swi/latest_swi_51.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_49.jpg", "/whirlwind/goes16/ir/swi/latest_swi_50.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_48.jpg", "/whirlwind/goes16/ir/swi/latest_swi_49.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_47.jpg", "/whirlwind/goes16/ir/swi/latest_swi_48.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_46.jpg", "/whirlwind/goes16/ir/swi/latest_swi_47.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_45.jpg", "/whirlwind/goes16/ir/swi/latest_swi_46.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_44.jpg", "/whirlwind/goes16/ir/swi/latest_swi_45.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_43.jpg", "/whirlwind/goes16/ir/swi/latest_swi_44.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_42.jpg", "/whirlwind/goes16/ir/swi/latest_swi_43.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_41.jpg", "/whirlwind/goes16/ir/swi/latest_swi_42.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_40.jpg", "/whirlwind/goes16/ir/swi/latest_swi_41.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_39.jpg", "/whirlwind/goes16/ir/swi/latest_swi_40.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_38.jpg", "/whirlwind/goes16/ir/swi/latest_swi_39.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_37.jpg", "/whirlwind/goes16/ir/swi/latest_swi_38.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_36.jpg", "/whirlwind/goes16/ir/swi/latest_swi_37.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_35.jpg", "/whirlwind/goes16/ir/swi/latest_swi_36.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_34.jpg", "/whirlwind/goes16/ir/swi/latest_swi_35.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_33.jpg", "/whirlwind/goes16/ir/swi/latest_swi_34.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_32.jpg", "/whirlwind/goes16/ir/swi/latest_swi_33.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_31.jpg", "/whirlwind/goes16/ir/swi/latest_swi_32.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_30.jpg", "/whirlwind/goes16/ir/swi/latest_swi_31.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_29.jpg", "/whirlwind/goes16/ir/swi/latest_swi_30.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_28.jpg", "/whirlwind/goes16/ir/swi/latest_swi_29.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_27.jpg", "/whirlwind/goes16/ir/swi/latest_swi_28.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_26.jpg", "/whirlwind/goes16/ir/swi/latest_swi_27.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_25.jpg", "/whirlwind/goes16/ir/swi/latest_swi_26.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_24.jpg", "/whirlwind/goes16/ir/swi/latest_swi_25.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_23.jpg", "/whirlwind/goes16/ir/swi/latest_swi_24.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_22.jpg", "/whirlwind/goes16/ir/swi/latest_swi_23.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_21.jpg", "/whirlwind/goes16/ir/swi/latest_swi_22.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_20.jpg", "/whirlwind/goes16/ir/swi/latest_swi_21.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_19.jpg", "/whirlwind/goes16/ir/swi/latest_swi_20.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_18.jpg", "/whirlwind/goes16/ir/swi/latest_swi_19.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_17.jpg", "/whirlwind/goes16/ir/swi/latest_swi_18.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_16.jpg", "/whirlwind/goes16/ir/swi/latest_swi_17.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_15.jpg", "/whirlwind/goes16/ir/swi/latest_swi_16.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_14.jpg", "/whirlwind/goes16/ir/swi/latest_swi_15.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_13.jpg", "/whirlwind/goes16/ir/swi/latest_swi_14.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_12.jpg", "/whirlwind/goes16/ir/swi/latest_swi_13.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_11.jpg", "/whirlwind/goes16/ir/swi/latest_swi_12.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_10.jpg", "/whirlwind/goes16/ir/swi/latest_swi_11.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_9.jpg", "/whirlwind/goes16/ir/swi/latest_swi_10.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_8.jpg", "/whirlwind/goes16/ir/swi/latest_swi_9.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_7.jpg", "/whirlwind/goes16/ir/swi/latest_swi_8.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_6.jpg", "/whirlwind/goes16/ir/swi/latest_swi_7.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_5.jpg", "/whirlwind/goes16/ir/swi/latest_swi_6.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_4.jpg", "/whirlwind/goes16/ir/swi/latest_swi_5.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_3.jpg", "/whirlwind/goes16/ir/swi/latest_swi_4.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_2.jpg", "/whirlwind/goes16/ir/swi/latest_swi_3.jpg")
silentrename("/whirlwind/goes16/ir/swi/latest_swi_1.jpg", "/whirlwind/goes16/ir/swi/latest_swi_2.jpg")

shutil.copy(filename5, "/whirlwind/goes16/ir/swi/latest_swi_1.jpg")

print("rename co")
# Colorado
silentremove("/whirlwind/goes16/ir/co/latest_co_72.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_71.jpg", "/whirlwind/goes16/ir/co/latest_co_72.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_70.jpg", "/whirlwind/goes16/ir/co/latest_co_71.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_69.jpg", "/whirlwind/goes16/ir/co/latest_co_70.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_68.jpg", "/whirlwind/goes16/ir/co/latest_co_69.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_67.jpg", "/whirlwind/goes16/ir/co/latest_co_68.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_66.jpg", "/whirlwind/goes16/ir/co/latest_co_67.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_65.jpg", "/whirlwind/goes16/ir/co/latest_co_66.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_64.jpg", "/whirlwind/goes16/ir/co/latest_co_65.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_63.jpg", "/whirlwind/goes16/ir/co/latest_co_64.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_62.jpg", "/whirlwind/goes16/ir/co/latest_co_63.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_61.jpg", "/whirlwind/goes16/ir/co/latest_co_62.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_60.jpg", "/whirlwind/goes16/ir/co/latest_co_61.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_59.jpg", "/whirlwind/goes16/ir/co/latest_co_60.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_58.jpg", "/whirlwind/goes16/ir/co/latest_co_59.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_57.jpg", "/whirlwind/goes16/ir/co/latest_co_58.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_56.jpg", "/whirlwind/goes16/ir/co/latest_co_57.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_55.jpg", "/whirlwind/goes16/ir/co/latest_co_56.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_54.jpg", "/whirlwind/goes16/ir/co/latest_co_55.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_53.jpg", "/whirlwind/goes16/ir/co/latest_co_54.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_52.jpg", "/whirlwind/goes16/ir/co/latest_co_53.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_51.jpg", "/whirlwind/goes16/ir/co/latest_co_52.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_50.jpg", "/whirlwind/goes16/ir/co/latest_co_51.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_49.jpg", "/whirlwind/goes16/ir/co/latest_co_50.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_48.jpg", "/whirlwind/goes16/ir/co/latest_co_49.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_47.jpg", "/whirlwind/goes16/ir/co/latest_co_48.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_46.jpg", "/whirlwind/goes16/ir/co/latest_co_47.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_45.jpg", "/whirlwind/goes16/ir/co/latest_co_46.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_44.jpg", "/whirlwind/goes16/ir/co/latest_co_45.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_43.jpg", "/whirlwind/goes16/ir/co/latest_co_44.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_42.jpg", "/whirlwind/goes16/ir/co/latest_co_43.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_41.jpg", "/whirlwind/goes16/ir/co/latest_co_42.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_40.jpg", "/whirlwind/goes16/ir/co/latest_co_41.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_39.jpg", "/whirlwind/goes16/ir/co/latest_co_40.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_38.jpg", "/whirlwind/goes16/ir/co/latest_co_39.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_37.jpg", "/whirlwind/goes16/ir/co/latest_co_38.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_36.jpg", "/whirlwind/goes16/ir/co/latest_co_37.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_35.jpg", "/whirlwind/goes16/ir/co/latest_co_36.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_34.jpg", "/whirlwind/goes16/ir/co/latest_co_35.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_33.jpg", "/whirlwind/goes16/ir/co/latest_co_34.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_32.jpg", "/whirlwind/goes16/ir/co/latest_co_33.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_31.jpg", "/whirlwind/goes16/ir/co/latest_co_32.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_30.jpg", "/whirlwind/goes16/ir/co/latest_co_31.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_29.jpg", "/whirlwind/goes16/ir/co/latest_co_30.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_28.jpg", "/whirlwind/goes16/ir/co/latest_co_29.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_27.jpg", "/whirlwind/goes16/ir/co/latest_co_28.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_26.jpg", "/whirlwind/goes16/ir/co/latest_co_27.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_25.jpg", "/whirlwind/goes16/ir/co/latest_co_26.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_24.jpg", "/whirlwind/goes16/ir/co/latest_co_25.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_23.jpg", "/whirlwind/goes16/ir/co/latest_co_24.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_22.jpg", "/whirlwind/goes16/ir/co/latest_co_23.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_21.jpg", "/whirlwind/goes16/ir/co/latest_co_22.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_20.jpg", "/whirlwind/goes16/ir/co/latest_co_21.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_19.jpg", "/whirlwind/goes16/ir/co/latest_co_20.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_18.jpg", "/whirlwind/goes16/ir/co/latest_co_19.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_17.jpg", "/whirlwind/goes16/ir/co/latest_co_18.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_16.jpg", "/whirlwind/goes16/ir/co/latest_co_17.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_15.jpg", "/whirlwind/goes16/ir/co/latest_co_16.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_14.jpg", "/whirlwind/goes16/ir/co/latest_co_15.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_13.jpg", "/whirlwind/goes16/ir/co/latest_co_14.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_12.jpg", "/whirlwind/goes16/ir/co/latest_co_13.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_11.jpg", "/whirlwind/goes16/ir/co/latest_co_12.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_10.jpg", "/whirlwind/goes16/ir/co/latest_co_11.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_9.jpg", "/whirlwind/goes16/ir/co/latest_co_10.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_8.jpg", "/whirlwind/goes16/ir/co/latest_co_9.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_7.jpg", "/whirlwind/goes16/ir/co/latest_co_8.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_6.jpg", "/whirlwind/goes16/ir/co/latest_co_7.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_5.jpg", "/whirlwind/goes16/ir/co/latest_co_6.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_4.jpg", "/whirlwind/goes16/ir/co/latest_co_5.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_3.jpg", "/whirlwind/goes16/ir/co/latest_co_4.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_2.jpg", "/whirlwind/goes16/ir/co/latest_co_3.jpg")
silentrename("/whirlwind/goes16/ir/co/latest_co_1.jpg", "/whirlwind/goes16/ir/co/latest_co_2.jpg")

shutil.copy(filename6, "/whirlwind/goes16/ir/co/latest_co_1.jpg")

print("rename fl")
# Florida
silentremove("/whirlwind/goes16/ir/fl/latest_fl_72.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_71.jpg", "/whirlwind/goes16/ir/fl/latest_fl_72.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_70.jpg", "/whirlwind/goes16/ir/fl/latest_fl_71.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_69.jpg", "/whirlwind/goes16/ir/fl/latest_fl_70.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_68.jpg", "/whirlwind/goes16/ir/fl/latest_fl_69.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_67.jpg", "/whirlwind/goes16/ir/fl/latest_fl_68.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_66.jpg", "/whirlwind/goes16/ir/fl/latest_fl_67.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_65.jpg", "/whirlwind/goes16/ir/fl/latest_fl_66.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_64.jpg", "/whirlwind/goes16/ir/fl/latest_fl_65.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_63.jpg", "/whirlwind/goes16/ir/fl/latest_fl_64.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_62.jpg", "/whirlwind/goes16/ir/fl/latest_fl_63.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_61.jpg", "/whirlwind/goes16/ir/fl/latest_fl_62.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_60.jpg", "/whirlwind/goes16/ir/fl/latest_fl_61.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_59.jpg", "/whirlwind/goes16/ir/fl/latest_fl_60.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_58.jpg", "/whirlwind/goes16/ir/fl/latest_fl_59.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_57.jpg", "/whirlwind/goes16/ir/fl/latest_fl_58.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_56.jpg", "/whirlwind/goes16/ir/fl/latest_fl_57.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_55.jpg", "/whirlwind/goes16/ir/fl/latest_fl_56.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_54.jpg", "/whirlwind/goes16/ir/fl/latest_fl_55.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_53.jpg", "/whirlwind/goes16/ir/fl/latest_fl_54.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_52.jpg", "/whirlwind/goes16/ir/fl/latest_fl_53.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_51.jpg", "/whirlwind/goes16/ir/fl/latest_fl_52.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_50.jpg", "/whirlwind/goes16/ir/fl/latest_fl_51.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_49.jpg", "/whirlwind/goes16/ir/fl/latest_fl_50.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_48.jpg", "/whirlwind/goes16/ir/fl/latest_fl_49.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_47.jpg", "/whirlwind/goes16/ir/fl/latest_fl_48.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_46.jpg", "/whirlwind/goes16/ir/fl/latest_fl_47.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_45.jpg", "/whirlwind/goes16/ir/fl/latest_fl_46.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_44.jpg", "/whirlwind/goes16/ir/fl/latest_fl_45.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_43.jpg", "/whirlwind/goes16/ir/fl/latest_fl_44.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_42.jpg", "/whirlwind/goes16/ir/fl/latest_fl_43.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_41.jpg", "/whirlwind/goes16/ir/fl/latest_fl_42.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_40.jpg", "/whirlwind/goes16/ir/fl/latest_fl_41.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_39.jpg", "/whirlwind/goes16/ir/fl/latest_fl_40.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_38.jpg", "/whirlwind/goes16/ir/fl/latest_fl_39.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_37.jpg", "/whirlwind/goes16/ir/fl/latest_fl_38.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_36.jpg", "/whirlwind/goes16/ir/fl/latest_fl_37.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_35.jpg", "/whirlwind/goes16/ir/fl/latest_fl_36.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_34.jpg", "/whirlwind/goes16/ir/fl/latest_fl_35.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_33.jpg", "/whirlwind/goes16/ir/fl/latest_fl_34.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_32.jpg", "/whirlwind/goes16/ir/fl/latest_fl_33.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_31.jpg", "/whirlwind/goes16/ir/fl/latest_fl_32.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_30.jpg", "/whirlwind/goes16/ir/fl/latest_fl_31.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_29.jpg", "/whirlwind/goes16/ir/fl/latest_fl_30.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_28.jpg", "/whirlwind/goes16/ir/fl/latest_fl_29.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_27.jpg", "/whirlwind/goes16/ir/fl/latest_fl_28.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_26.jpg", "/whirlwind/goes16/ir/fl/latest_fl_27.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_25.jpg", "/whirlwind/goes16/ir/fl/latest_fl_26.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_24.jpg", "/whirlwind/goes16/ir/fl/latest_fl_25.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_23.jpg", "/whirlwind/goes16/ir/fl/latest_fl_24.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_22.jpg", "/whirlwind/goes16/ir/fl/latest_fl_23.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_21.jpg", "/whirlwind/goes16/ir/fl/latest_fl_22.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_20.jpg", "/whirlwind/goes16/ir/fl/latest_fl_21.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_19.jpg", "/whirlwind/goes16/ir/fl/latest_fl_20.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_18.jpg", "/whirlwind/goes16/ir/fl/latest_fl_19.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_17.jpg", "/whirlwind/goes16/ir/fl/latest_fl_18.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_16.jpg", "/whirlwind/goes16/ir/fl/latest_fl_17.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_15.jpg", "/whirlwind/goes16/ir/fl/latest_fl_16.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_14.jpg", "/whirlwind/goes16/ir/fl/latest_fl_15.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_13.jpg", "/whirlwind/goes16/ir/fl/latest_fl_14.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_12.jpg", "/whirlwind/goes16/ir/fl/latest_fl_13.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_11.jpg", "/whirlwind/goes16/ir/fl/latest_fl_12.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_10.jpg", "/whirlwind/goes16/ir/fl/latest_fl_11.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_9.jpg", "/whirlwind/goes16/ir/fl/latest_fl_10.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_8.jpg", "/whirlwind/goes16/ir/fl/latest_fl_9.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_7.jpg", "/whirlwind/goes16/ir/fl/latest_fl_8.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_6.jpg", "/whirlwind/goes16/ir/fl/latest_fl_7.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_5.jpg", "/whirlwind/goes16/ir/fl/latest_fl_6.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_4.jpg", "/whirlwind/goes16/ir/fl/latest_fl_5.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_3.jpg", "/whirlwind/goes16/ir/fl/latest_fl_4.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_2.jpg", "/whirlwind/goes16/ir/fl/latest_fl_3.jpg")
silentrename("/whirlwind/goes16/ir/fl/latest_fl_1.jpg", "/whirlwind/goes16/ir/fl/latest_fl_2.jpg")

shutil.copy(filename7, "/whirlwind/goes16/ir/fl/latest_fl_1.jpg")

print("rename gulf")
# Gulf of Mexico
silentremove("/whirlwind/goes16/ir/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_71.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_72.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_70.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_71.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_69.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_70.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_68.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_69.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_67.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_68.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_66.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_67.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_65.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_66.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_64.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_65.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_63.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_64.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_62.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_63.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_61.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_62.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_60.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_61.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_59.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_60.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_58.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_59.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_57.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_58.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_56.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_57.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_55.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_56.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_54.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_55.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_53.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_54.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_52.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_53.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_51.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_52.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_50.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_51.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_49.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_50.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_48.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_49.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_47.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_48.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_46.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_47.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_45.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_46.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_44.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_45.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_43.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_44.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_42.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_43.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_41.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_42.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_40.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_41.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_39.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_40.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_38.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_39.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_37.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_38.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_36.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_37.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_35.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_36.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_34.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_35.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_33.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_34.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_32.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_33.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_31.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_32.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_30.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_31.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_29.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_30.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_28.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_29.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_27.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_28.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_26.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_27.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_25.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_26.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_24.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_25.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_23.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_24.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_22.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_23.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_21.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_22.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_20.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_21.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_19.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_20.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_18.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_19.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_17.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_18.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_16.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_17.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_15.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_16.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_14.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_15.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_13.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_14.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_12.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_13.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_11.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_12.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_10.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_11.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_9.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_10.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_8.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_9.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_7.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_8.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_6.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_7.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_5.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_6.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_4.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_5.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_3.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_4.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_2.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_3.jpg")
silentrename("/whirlwind/goes16/ir/gulf/latest_gulf_1.jpg", "/whirlwind/goes16/ir/gulf/latest_gulf_2.jpg")

shutil.copy(filename8, "/whirlwind/goes16/ir/gulf/latest_gulf_1.jpg")

print("rename alex")
# Alex Stormchase
silentremove("/whirlwind/goes16/ir/alex/latest_alex_72.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_71.jpg", "/whirlwind/goes16/ir/alex/latest_alex_72.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_70.jpg", "/whirlwind/goes16/ir/alex/latest_alex_71.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_69.jpg", "/whirlwind/goes16/ir/alex/latest_alex_70.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_68.jpg", "/whirlwind/goes16/ir/alex/latest_alex_69.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_67.jpg", "/whirlwind/goes16/ir/alex/latest_alex_68.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_66.jpg", "/whirlwind/goes16/ir/alex/latest_alex_67.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_65.jpg", "/whirlwind/goes16/ir/alex/latest_alex_66.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_64.jpg", "/whirlwind/goes16/ir/alex/latest_alex_65.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_63.jpg", "/whirlwind/goes16/ir/alex/latest_alex_64.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_62.jpg", "/whirlwind/goes16/ir/alex/latest_alex_63.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_61.jpg", "/whirlwind/goes16/ir/alex/latest_alex_62.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_60.jpg", "/whirlwind/goes16/ir/alex/latest_alex_61.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_59.jpg", "/whirlwind/goes16/ir/alex/latest_alex_60.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_58.jpg", "/whirlwind/goes16/ir/alex/latest_alex_59.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_57.jpg", "/whirlwind/goes16/ir/alex/latest_alex_58.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_56.jpg", "/whirlwind/goes16/ir/alex/latest_alex_57.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_55.jpg", "/whirlwind/goes16/ir/alex/latest_alex_56.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_54.jpg", "/whirlwind/goes16/ir/alex/latest_alex_55.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_53.jpg", "/whirlwind/goes16/ir/alex/latest_alex_54.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_52.jpg", "/whirlwind/goes16/ir/alex/latest_alex_53.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_51.jpg", "/whirlwind/goes16/ir/alex/latest_alex_52.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_50.jpg", "/whirlwind/goes16/ir/alex/latest_alex_51.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_49.jpg", "/whirlwind/goes16/ir/alex/latest_alex_50.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_48.jpg", "/whirlwind/goes16/ir/alex/latest_alex_49.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_47.jpg", "/whirlwind/goes16/ir/alex/latest_alex_48.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_46.jpg", "/whirlwind/goes16/ir/alex/latest_alex_47.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_45.jpg", "/whirlwind/goes16/ir/alex/latest_alex_46.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_44.jpg", "/whirlwind/goes16/ir/alex/latest_alex_45.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_43.jpg", "/whirlwind/goes16/ir/alex/latest_alex_44.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_42.jpg", "/whirlwind/goes16/ir/alex/latest_alex_43.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_41.jpg", "/whirlwind/goes16/ir/alex/latest_alex_42.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_40.jpg", "/whirlwind/goes16/ir/alex/latest_alex_41.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_39.jpg", "/whirlwind/goes16/ir/alex/latest_alex_40.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_38.jpg", "/whirlwind/goes16/ir/alex/latest_alex_39.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_37.jpg", "/whirlwind/goes16/ir/alex/latest_alex_38.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_36.jpg", "/whirlwind/goes16/ir/alex/latest_alex_37.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_35.jpg", "/whirlwind/goes16/ir/alex/latest_alex_36.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_34.jpg", "/whirlwind/goes16/ir/alex/latest_alex_35.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_33.jpg", "/whirlwind/goes16/ir/alex/latest_alex_34.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_32.jpg", "/whirlwind/goes16/ir/alex/latest_alex_33.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_31.jpg", "/whirlwind/goes16/ir/alex/latest_alex_32.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_30.jpg", "/whirlwind/goes16/ir/alex/latest_alex_31.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_29.jpg", "/whirlwind/goes16/ir/alex/latest_alex_30.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_28.jpg", "/whirlwind/goes16/ir/alex/latest_alex_29.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_27.jpg", "/whirlwind/goes16/ir/alex/latest_alex_28.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_26.jpg", "/whirlwind/goes16/ir/alex/latest_alex_27.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_25.jpg", "/whirlwind/goes16/ir/alex/latest_alex_26.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_24.jpg", "/whirlwind/goes16/ir/alex/latest_alex_25.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_23.jpg", "/whirlwind/goes16/ir/alex/latest_alex_24.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_22.jpg", "/whirlwind/goes16/ir/alex/latest_alex_23.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_21.jpg", "/whirlwind/goes16/ir/alex/latest_alex_22.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_20.jpg", "/whirlwind/goes16/ir/alex/latest_alex_21.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_19.jpg", "/whirlwind/goes16/ir/alex/latest_alex_20.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_18.jpg", "/whirlwind/goes16/ir/alex/latest_alex_19.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_17.jpg", "/whirlwind/goes16/ir/alex/latest_alex_18.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_16.jpg", "/whirlwind/goes16/ir/alex/latest_alex_17.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_15.jpg", "/whirlwind/goes16/ir/alex/latest_alex_16.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_14.jpg", "/whirlwind/goes16/ir/alex/latest_alex_15.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_13.jpg", "/whirlwind/goes16/ir/alex/latest_alex_14.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_12.jpg", "/whirlwind/goes16/ir/alex/latest_alex_13.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_11.jpg", "/whirlwind/goes16/ir/alex/latest_alex_12.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_10.jpg", "/whirlwind/goes16/ir/alex/latest_alex_11.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_9.jpg", "/whirlwind/goes16/ir/alex/latest_alex_10.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_8.jpg", "/whirlwind/goes16/ir/alex/latest_alex_9.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_7.jpg", "/whirlwind/goes16/ir/alex/latest_alex_8.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_6.jpg", "/whirlwind/goes16/ir/alex/latest_alex_7.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_5.jpg", "/whirlwind/goes16/ir/alex/latest_alex_6.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_4.jpg", "/whirlwind/goes16/ir/alex/latest_alex_5.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_3.jpg", "/whirlwind/goes16/ir/alex/latest_alex_4.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_2.jpg", "/whirlwind/goes16/ir/alex/latest_alex_3.jpg")
silentrename("/whirlwind/goes16/ir/alex/latest_alex_1.jpg", "/whirlwind/goes16/ir/alex/latest_alex_2.jpg")

shutil.copy(filename10, "/whirlwind/goes16/ir/alex/latest_alex_1.jpg")

print("rename sw")
silentremove("/whirlwind/goes16/ir/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_71.jpg", "/whirlwind/goes16/ir/sw/latest_sw_72.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_70.jpg", "/whirlwind/goes16/ir/sw/latest_sw_71.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_69.jpg", "/whirlwind/goes16/ir/sw/latest_sw_70.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_68.jpg", "/whirlwind/goes16/ir/sw/latest_sw_69.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_67.jpg", "/whirlwind/goes16/ir/sw/latest_sw_68.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_66.jpg", "/whirlwind/goes16/ir/sw/latest_sw_67.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_65.jpg", "/whirlwind/goes16/ir/sw/latest_sw_66.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_64.jpg", "/whirlwind/goes16/ir/sw/latest_sw_65.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_63.jpg", "/whirlwind/goes16/ir/sw/latest_sw_64.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_62.jpg", "/whirlwind/goes16/ir/sw/latest_sw_63.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_61.jpg", "/whirlwind/goes16/ir/sw/latest_sw_62.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_60.jpg", "/whirlwind/goes16/ir/sw/latest_sw_61.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_59.jpg", "/whirlwind/goes16/ir/sw/latest_sw_60.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_58.jpg", "/whirlwind/goes16/ir/sw/latest_sw_59.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_57.jpg", "/whirlwind/goes16/ir/sw/latest_sw_58.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_56.jpg", "/whirlwind/goes16/ir/sw/latest_sw_57.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_55.jpg", "/whirlwind/goes16/ir/sw/latest_sw_56.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_54.jpg", "/whirlwind/goes16/ir/sw/latest_sw_55.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_53.jpg", "/whirlwind/goes16/ir/sw/latest_sw_54.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_52.jpg", "/whirlwind/goes16/ir/sw/latest_sw_53.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_51.jpg", "/whirlwind/goes16/ir/sw/latest_sw_52.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_50.jpg", "/whirlwind/goes16/ir/sw/latest_sw_51.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_49.jpg", "/whirlwind/goes16/ir/sw/latest_sw_50.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_48.jpg", "/whirlwind/goes16/ir/sw/latest_sw_49.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_47.jpg", "/whirlwind/goes16/ir/sw/latest_sw_48.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_46.jpg", "/whirlwind/goes16/ir/sw/latest_sw_47.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_45.jpg", "/whirlwind/goes16/ir/sw/latest_sw_46.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_44.jpg", "/whirlwind/goes16/ir/sw/latest_sw_45.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_43.jpg", "/whirlwind/goes16/ir/sw/latest_sw_44.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_42.jpg", "/whirlwind/goes16/ir/sw/latest_sw_43.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_41.jpg", "/whirlwind/goes16/ir/sw/latest_sw_42.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_40.jpg", "/whirlwind/goes16/ir/sw/latest_sw_41.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_39.jpg", "/whirlwind/goes16/ir/sw/latest_sw_40.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_38.jpg", "/whirlwind/goes16/ir/sw/latest_sw_39.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_37.jpg", "/whirlwind/goes16/ir/sw/latest_sw_38.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_36.jpg", "/whirlwind/goes16/ir/sw/latest_sw_37.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_35.jpg", "/whirlwind/goes16/ir/sw/latest_sw_36.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_34.jpg", "/whirlwind/goes16/ir/sw/latest_sw_35.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_33.jpg", "/whirlwind/goes16/ir/sw/latest_sw_34.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_32.jpg", "/whirlwind/goes16/ir/sw/latest_sw_33.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_31.jpg", "/whirlwind/goes16/ir/sw/latest_sw_32.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_30.jpg", "/whirlwind/goes16/ir/sw/latest_sw_31.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_29.jpg", "/whirlwind/goes16/ir/sw/latest_sw_30.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_28.jpg", "/whirlwind/goes16/ir/sw/latest_sw_29.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_27.jpg", "/whirlwind/goes16/ir/sw/latest_sw_28.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_26.jpg", "/whirlwind/goes16/ir/sw/latest_sw_27.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_25.jpg", "/whirlwind/goes16/ir/sw/latest_sw_26.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_24.jpg", "/whirlwind/goes16/ir/sw/latest_sw_25.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_23.jpg", "/whirlwind/goes16/ir/sw/latest_sw_24.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_22.jpg", "/whirlwind/goes16/ir/sw/latest_sw_23.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_21.jpg", "/whirlwind/goes16/ir/sw/latest_sw_22.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_20.jpg", "/whirlwind/goes16/ir/sw/latest_sw_21.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_19.jpg", "/whirlwind/goes16/ir/sw/latest_sw_20.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_18.jpg", "/whirlwind/goes16/ir/sw/latest_sw_19.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_17.jpg", "/whirlwind/goes16/ir/sw/latest_sw_18.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_16.jpg", "/whirlwind/goes16/ir/sw/latest_sw_17.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_15.jpg", "/whirlwind/goes16/ir/sw/latest_sw_16.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_14.jpg", "/whirlwind/goes16/ir/sw/latest_sw_15.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_13.jpg", "/whirlwind/goes16/ir/sw/latest_sw_14.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_12.jpg", "/whirlwind/goes16/ir/sw/latest_sw_13.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_11.jpg", "/whirlwind/goes16/ir/sw/latest_sw_12.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_10.jpg", "/whirlwind/goes16/ir/sw/latest_sw_11.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_9.jpg", "/whirlwind/goes16/ir/sw/latest_sw_10.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_8.jpg", "/whirlwind/goes16/ir/sw/latest_sw_9.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_7.jpg", "/whirlwind/goes16/ir/sw/latest_sw_8.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_6.jpg", "/whirlwind/goes16/ir/sw/latest_sw_7.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_5.jpg", "/whirlwind/goes16/ir/sw/latest_sw_6.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_4.jpg", "/whirlwind/goes16/ir/sw/latest_sw_5.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_3.jpg", "/whirlwind/goes16/ir/sw/latest_sw_4.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_2.jpg", "/whirlwind/goes16/ir/sw/latest_sw_3.jpg")
silentrename("/whirlwind/goes16/ir/sw/latest_sw_1.jpg", "/whirlwind/goes16/ir/sw/latest_sw_2.jpg")

shutil.copy(filename13, "/whirlwind/goes16/ir/sw/latest_sw_1.jpg")

print("rename nw")
silentremove("/whirlwind/goes16/ir/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_71.jpg", "/whirlwind/goes16/ir/nw/latest_nw_72.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_70.jpg", "/whirlwind/goes16/ir/nw/latest_nw_71.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_69.jpg", "/whirlwind/goes16/ir/nw/latest_nw_70.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_68.jpg", "/whirlwind/goes16/ir/nw/latest_nw_69.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_67.jpg", "/whirlwind/goes16/ir/nw/latest_nw_68.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_66.jpg", "/whirlwind/goes16/ir/nw/latest_nw_67.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_65.jpg", "/whirlwind/goes16/ir/nw/latest_nw_66.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_64.jpg", "/whirlwind/goes16/ir/nw/latest_nw_65.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_63.jpg", "/whirlwind/goes16/ir/nw/latest_nw_64.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_62.jpg", "/whirlwind/goes16/ir/nw/latest_nw_63.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_61.jpg", "/whirlwind/goes16/ir/nw/latest_nw_62.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_60.jpg", "/whirlwind/goes16/ir/nw/latest_nw_61.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_59.jpg", "/whirlwind/goes16/ir/nw/latest_nw_60.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_58.jpg", "/whirlwind/goes16/ir/nw/latest_nw_59.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_57.jpg", "/whirlwind/goes16/ir/nw/latest_nw_58.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_56.jpg", "/whirlwind/goes16/ir/nw/latest_nw_57.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_55.jpg", "/whirlwind/goes16/ir/nw/latest_nw_56.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_54.jpg", "/whirlwind/goes16/ir/nw/latest_nw_55.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_53.jpg", "/whirlwind/goes16/ir/nw/latest_nw_54.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_52.jpg", "/whirlwind/goes16/ir/nw/latest_nw_53.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_51.jpg", "/whirlwind/goes16/ir/nw/latest_nw_52.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_50.jpg", "/whirlwind/goes16/ir/nw/latest_nw_51.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_49.jpg", "/whirlwind/goes16/ir/nw/latest_nw_50.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_48.jpg", "/whirlwind/goes16/ir/nw/latest_nw_49.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_47.jpg", "/whirlwind/goes16/ir/nw/latest_nw_48.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_46.jpg", "/whirlwind/goes16/ir/nw/latest_nw_47.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_45.jpg", "/whirlwind/goes16/ir/nw/latest_nw_46.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_44.jpg", "/whirlwind/goes16/ir/nw/latest_nw_45.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_43.jpg", "/whirlwind/goes16/ir/nw/latest_nw_44.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_42.jpg", "/whirlwind/goes16/ir/nw/latest_nw_43.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_41.jpg", "/whirlwind/goes16/ir/nw/latest_nw_42.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_40.jpg", "/whirlwind/goes16/ir/nw/latest_nw_41.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_39.jpg", "/whirlwind/goes16/ir/nw/latest_nw_40.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_38.jpg", "/whirlwind/goes16/ir/nw/latest_nw_39.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_37.jpg", "/whirlwind/goes16/ir/nw/latest_nw_38.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_36.jpg", "/whirlwind/goes16/ir/nw/latest_nw_37.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_35.jpg", "/whirlwind/goes16/ir/nw/latest_nw_36.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_34.jpg", "/whirlwind/goes16/ir/nw/latest_nw_35.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_33.jpg", "/whirlwind/goes16/ir/nw/latest_nw_34.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_32.jpg", "/whirlwind/goes16/ir/nw/latest_nw_33.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_31.jpg", "/whirlwind/goes16/ir/nw/latest_nw_32.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_30.jpg", "/whirlwind/goes16/ir/nw/latest_nw_31.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_29.jpg", "/whirlwind/goes16/ir/nw/latest_nw_30.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_28.jpg", "/whirlwind/goes16/ir/nw/latest_nw_29.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_27.jpg", "/whirlwind/goes16/ir/nw/latest_nw_28.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_26.jpg", "/whirlwind/goes16/ir/nw/latest_nw_27.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_25.jpg", "/whirlwind/goes16/ir/nw/latest_nw_26.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_24.jpg", "/whirlwind/goes16/ir/nw/latest_nw_25.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_23.jpg", "/whirlwind/goes16/ir/nw/latest_nw_24.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_22.jpg", "/whirlwind/goes16/ir/nw/latest_nw_23.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_21.jpg", "/whirlwind/goes16/ir/nw/latest_nw_22.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_20.jpg", "/whirlwind/goes16/ir/nw/latest_nw_21.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_19.jpg", "/whirlwind/goes16/ir/nw/latest_nw_20.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_18.jpg", "/whirlwind/goes16/ir/nw/latest_nw_19.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_17.jpg", "/whirlwind/goes16/ir/nw/latest_nw_18.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_16.jpg", "/whirlwind/goes16/ir/nw/latest_nw_17.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_15.jpg", "/whirlwind/goes16/ir/nw/latest_nw_16.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_14.jpg", "/whirlwind/goes16/ir/nw/latest_nw_15.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_13.jpg", "/whirlwind/goes16/ir/nw/latest_nw_14.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_12.jpg", "/whirlwind/goes16/ir/nw/latest_nw_13.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_11.jpg", "/whirlwind/goes16/ir/nw/latest_nw_12.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_10.jpg", "/whirlwind/goes16/ir/nw/latest_nw_11.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_9.jpg", "/whirlwind/goes16/ir/nw/latest_nw_10.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_8.jpg", "/whirlwind/goes16/ir/nw/latest_nw_9.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_7.jpg", "/whirlwind/goes16/ir/nw/latest_nw_8.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_6.jpg", "/whirlwind/goes16/ir/nw/latest_nw_7.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_5.jpg", "/whirlwind/goes16/ir/nw/latest_nw_6.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_4.jpg", "/whirlwind/goes16/ir/nw/latest_nw_5.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_3.jpg", "/whirlwind/goes16/ir/nw/latest_nw_4.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_2.jpg", "/whirlwind/goes16/ir/nw/latest_nw_3.jpg")
silentrename("/whirlwind/goes16/ir/nw/latest_nw_1.jpg", "/whirlwind/goes16/ir/nw/latest_nw_2.jpg")

shutil.copy(filename14, "/whirlwind/goes16/ir/nw/latest_nw_1.jpg")

print("rename gtlakes")
silentremove("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_72.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_71.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_72.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_70.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_71.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_69.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_70.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_68.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_69.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_67.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_68.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_66.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_67.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_65.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_66.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_64.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_65.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_63.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_64.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_62.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_63.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_61.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_62.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_60.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_61.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_59.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_60.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_58.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_59.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_57.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_58.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_56.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_57.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_55.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_56.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_54.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_55.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_53.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_54.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_52.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_53.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_51.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_52.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_50.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_51.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_49.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_50.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_48.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_49.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_47.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_48.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_46.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_47.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_45.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_46.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_44.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_45.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_43.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_44.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_42.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_43.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_41.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_42.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_40.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_41.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_39.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_40.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_38.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_39.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_37.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_38.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_36.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_37.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_35.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_36.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_34.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_35.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_33.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_34.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_32.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_33.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_31.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_32.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_30.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_31.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_29.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_30.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_28.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_29.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_27.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_28.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_26.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_27.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_25.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_26.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_24.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_25.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_23.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_24.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_22.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_23.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_21.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_22.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_20.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_21.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_19.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_20.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_18.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_19.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_17.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_18.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_16.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_17.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_15.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_16.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_14.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_15.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_13.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_14.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_12.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_13.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_11.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_12.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_10.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_11.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_9.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_10.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_8.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_9.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_7.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_8.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_6.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_7.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_5.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_6.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_4.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_5.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_3.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_4.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_2.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_3.jpg")
silentrename("/whirlwind/goes16/ir/gtlakes/latest_gtlakes_1.jpg", "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_2.jpg")

shutil.copy(filename15, "/whirlwind/goes16/ir/gtlakes/latest_gtlakes_1.jpg")


shutil.copy(filename9, "/whirlwind/goes16/ir/full/latest_full_1.jpg")
