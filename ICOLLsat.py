#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings

#load modules
import os
import numpy as np
#import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download, SDS_preprocess, SDS_entrance, SDS_tools #SDS_transects  #SDS_download,
import pandas as pd
#from osgeo import gdal, gdalconst

# load additional machine learning modules #VH check if still needed
from sklearn.externals import joblib
#import matplotlib.cm as cm
#import csv
import geopandas as gpd
#from skimage.filters import sobel
#from skimage.morphology import watershed
from matplotlib import colors
#import scipy



#from glob import glob

#img bands are B, G, R, NIR, SWIR starting from img[0]

#Analysis version/name
Analysis_version = 'V1' #this is a short user defined identifier that is added to end of directories to allow for multiple analysis to be done for each site with different parameters. 

# name of the site
sitename = 'CONJOLA'

# date range
dates = ['1985-01-01', '2020-08-01']

# satellite missions
sat_list = ['L5','L7','L8','S2']
sat_list = ['S2']

# filepath where data will be stored
filepath_data = os.path.join('H:/WRL_Projects/Estuary_sat_data/', 'data')

#load shapefile that conta0ins specific shapes for each ICOLL site as per readme file
location_shp_fp = os.path.join(os.getcwd(), 'sites', 'All_sites.shp')
Allsites = gpd.read_file(location_shp_fp)
Site_shps = Allsites.loc[(Allsites.Sitename==sitename)]
layers = Site_shps['layer'].values
Site_shps.plot(color='None', edgecolor='black')
BBX_coords = []
for b in Site_shps.loc[(Site_shps.layer=='full_bounding_box')].geometry.boundary:
    coords = np.dstack(b.coords.xy).tolist()
    BBX_coords.append(*coords) 
   
# put all the inputs into a dictionnary
inputs = {
    'polygon': BBX_coords,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data,
    'location_shps': Site_shps,
    'analysis_vrs' : Analysis_version
        }

# retrieve satellite images from GEE
#metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs) 
    

# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.05,        # threshold on maximum cloud cover
    'output_epsg': 3577,       # epsg code of spatial reference system desired for the output  
    'manual_seed': False,
    'shapefile_EPSG' : 4326,     #epsg of shapefiles that define sub regions for entrance detection 
    # quality control:
    'check_detection': True,        # if True, shows each entrance state detection to the user for validation #####!!!!!##### Intermediate - change variable to manual_input
    'shuffle_training_imgs':True,   # if True, images durin manual/visual detection of entrance states are shuffled in time to provide a more independent sample
    'shuffle_entrance_paths_imgs':True, 
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'dark_sand': False,         # only switch to True if your site has dark sand (e.g. black sand beach)
    'color_sand': False,         # set to true in case of black, grey, white, orange beaches 
}



# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
#SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
#settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
#settings['max_dist_ref'] = 20        


##### Original Code Above


#%%  Step 1: create training data
"""
Extracts ICOLL entrance characteristics from satellite images. Method development
Either do a 100% manual analysis or create a training and validation dataset for automated classification
"""

#run the training data creator function
settings_training =  { # set parameters for automated entrance detection
                    'check_detection': True,        # if True, shows each entrance state detection to the user for validation #####!!!!!##### Intermediate - change variable to manual_input
                    'shuffle_training_imgs':True,   # if True, images durin manual/visual detection of entrance states are shuffled in time to provide a more independent sample
                    'shuffle_entrance_paths_imgs':True, 
                    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image      
                      }

Training_data_df = SDS_entrance.create_training_data(metadata, settings)
layers['entrance_bounding_box'] 

#%%  Step 2: digitize transects
"""
# Currently research only: Run a function that enables the user to manually draw the seed to receiver 
connection for every image to generate a validation dataset
# After the dataset is created for a number of open and closed images, a series of interactive plots are available to illustrate the method
"""
#import re
#import glob
#import matplotlib
#import matplotlib.pyplot as plt
#from datetime import datetime 
#
#
##First create 20 closed and 20 open transects based on the yellow points (from ocean to ICOLL)               
#Experiment_code = 'Exp_1_S2_Yellow'
#csv_out_path = os.path.join(settings['inputs']['filepath'], sitename,  'results_' + settings['inputs']['analysis_vrs'], Experiment_code)
#if not os.path.exists(csv_out_path):
#        os.makedirs(csv_out_path)  
#       
##User to digitize transects for closed and then open conditions via the following two lines
#XS_c_df, XS_c_gdf, geoms = SDS_entrance.user_defined_entrance_paths(metadata, settings, Experiment_code, Experiment_code  + '_closed') 
#XS_o_df, XS_o_gdf, geoms  = SDS_entrance.user_defined_entrance_paths(metadata, settings, Experiment_code, Experiment_code + '_open')
# 
##Then create 20 closed and 20 open transects based on the green points (from north to south, always starting from the same point)               
#Experiment_code = 'Exp_1_S2_Green'
#csv_out_path = os.path.join(settings['inputs']['filepath'], sitename,  'results_' + settings['inputs']['analysis_vrs'], Experiment_code)
#if not os.path.exists(csv_out_path):
#        os.makedirs(csv_out_path)  
#        
##User to digitize transects for closed and then open conditions via the following two lines
#XS_c_df, XS_c_gdf, geoms = SDS_entrance.user_defined_entrance_paths(metadata, settings, Experiment_code, Experiment_code  + '_closed') 
#XS_o_df, XS_o_gdf, geoms  = SDS_entrance.user_defined_entrance_paths(metadata, settings, Experiment_code, Experiment_code + '_open')
#
#
#Experiment_code = 'Exp_5_S2'
#csv_out_path = os.path.join(settings['inputs']['filepath'], sitename,  'results_' + settings['inputs']['analysis_vrs'], Experiment_code)
###if XS are already digitized for a given Experiment_code, just load them here via csv. 
#XS_c_df =  pd.read_csv(glob.glob(csv_out_path + '/*' +  '*closed*' '*.csv' )[0], index_col=0)  
#XS_o_df =  pd.read_csv(glob.glob(csv_out_path + '/*' +  '*open*' '*.csv' )[0], index_col=0)  
#XS_c_gdf =  gpd.read_file(glob.glob(csv_out_path + '/*' +  '*closed*' '*.shp')[0])  
#XS_o_gdf = gpd.read_file(glob.glob(csv_out_path + '/*' +  '*open*' '*.shp')[0]) 


 
#%%  Step 2B: find transects automatically and write results to dataframe

#to do list
#extract XS from both NDWI and mNDWI
#for NDWI, apply NN classifier first and then replace white water with -.6 or sth to enable the least cost path finder 
#write least coast path to pd dataframe original infrastructure
#loop through satellites and all images within
#do along and across berm
#bring in truth data for open vs. closed vs. unclear either via user input or externally and incorporate into plots
#run 4-5 case studies

from coastsat import SDS_entrance

settings_entrance =  { # set parameters for automated entrance detection
                      'path_index': 'ndwi',
                      'Experiment_code': 'Exp_14_S2',       #unique identifier for the experiment. Outputs will be stored in this sub folder
                      'sand_percentile': 40 ,               #percentile of sand to plot in addition to 10th, 20th and 30th (which are always done)
                      'ndwi_whitewhater_delta': -0.2,       #where the NN classifier detects whitewater, NDWI will be adjusted by this value to facilitate least cost path finding
                      'ndwi_sand_delta': 0.6 ,              #where the NN classifier detects sand, NDWI will be adjusted by this value to facilitate least cost path finding
                      'vhline_transparancy': 0.8 ,          #transparancy of v and h lines in the output plots
                      'hist_bw': 0.1,                       #parameter for histogram smoothing in the output plots
                      'tide_bool': True ,                   #include FES-based analysis of tides?
                      'plot_bool': True  ,                  #create the output plots in addition to csvs?
                      'img_crop_adjsut': 20 ,               #nr of pixels to add on each side (along x axis) to the cropped image to fill out plt space. needs to be adjusted for each site
                      'fontsize' : 15 ,                      #size of fonts in plot
                      'axlabelsize': 20 ,
                      'number_of_images': 10,               #nr of images to process - if it exceeds len(images) all images will be processed
                      'plot percentiles': True,             #calculate the percentiles of the sand pixels in each scene and overlay on mNDWI and NDWI transects?          
                      }

XS_c_df, XS_c_gdf,geoms, sat_tide_df = SDS_entrance.automated_entrance_paths(metadata, settings, settings_entrance) 

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import skimage.filters as filters
from coastsat import SDS_entrance 
import skimage.transform as transform
from skimage.graph import route_through_array
import seaborn
from shapely import geometry
import scipy
from coastsat import SDS_slope
import matplotlib

#plot font size and type
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : settings_entrance['fontsize']}
matplotlib.rc('font', **font)  
labelsize = 26

sitename = settings['inputs']['sitename']
filepath_data = settings['inputs']['filepath']

print('Auto generating entrance paths at ' + sitename)

# create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
csv_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'], settings_entrance['Experiment_code'])
if not os.path.exists(csv_out_path):
        os.makedirs(csv_out_path)  
image_out_path = os.path.join(csv_out_path, 'auto_transects')
if not os.path.exists(image_out_path):
        os.makedirs(image_out_path) 

# close all open figures
plt.close('all')   

# initialise output data structure
iterator=0 #additional incremental integer to control the addition of user input lines to the gpd dataframe
gdf_all = gpd.GeoDataFrame()
XS={} 

# loop through the user selected satellites 
#for satname in settings['inputs']['sat_list']:
satname = 'S2'
# load classifiers
if satname in ['L5','L7','L8']:
    pixel_size = 15            
    if settings['dark_sand']:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat_dark.pkl'))
    elif settings['color_sand']:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat_diff_col_beaches.pkl'))
    else:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat_SEA.pkl'))      
elif satname == 'S2':
    pixel_size = 10
    clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_S2_SEA.pkl'))

# convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels         #####!!!!!##### Intermediate - probably drop    
#buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
  
#dates = metadata[satname]['dates']
# get images
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = metadata[satname]['filenames']

#randomize the time step to create a more independent training data set
epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))
dates_dict = dict(zip(filenames, metadata[satname]['dates']))

if settings_entrance['tide_bool']:
    
    #load required packages
    import pyfes
    import pytz
    from datetime import datetime

    #general base setting for tide analysis
    if satname in ['L5']:
        date_range = [1987,2011] 
    if satname in ['L7']:
        date_range = [2000,2013] 
    if satname in ['L7L8']:
        date_range = [2012,2021] 
    if satname == 'S2':
        date_range = [2015,2021] # range of dates over which to perform the analysis
    
    seconds_in_day = 24*3600
    # get tide time-series with 15 minutes intervals
    time_step = 15*60
    n_days =   8  # sampling period [days]
    dates_sat = metadata[satname]['dates']   
        
    #tide computations
    date_range = [pytz.utc.localize(datetime(date_range[0],5,1)), pytz.utc.localize(datetime(date_range[1],1,1))]
    filepath = r"H:\Downloads\fes-2.9.1-Source\data\fes2014"
    config_ocean = os.path.join(filepath, 'ocean_tide_extrapolated.ini')
    #config_ocean = os.path.join(filepath, 'ocean_tide_Kilian.ini')
    #config_ocean_extrap =  os.path.join(filepath, 'ocean_tide_extrapolated_Kilian.ini')
    config_load =  os.path.join(filepath, 'load_tide.ini')  
    ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
    load_tide = pyfes.Handler("radial", "io", config_load)
    
    # coordinates of the location (always select a point 1-2km offshore from the beach)
    Oceanseed_coords = []
    for b in settings['inputs']['location_shps'].loc[(settings['inputs']['location_shps'].layer=='ocean_seed')].geometry.boundary:
        coords = np.dstack(b.coords.xy).tolist()
        Oceanseed_coords.append(*coords)    
    coords = Oceanseed_coords[0][0]
    
    #obtain full tide time series for date range
    dates_fes, tide_fes = SDS_slope.compute_tide(coords,date_range, time_step, ocean_tide, load_tide)
       
    # get tide level at times of image acquisition
    tide_sat = SDS_slope.compute_tide_dates(coords, dates_sat, ocean_tide, load_tide)
    
    #create dataframe of tides
    sat_tides_df = pd.DataFrame(tide_sat,index=dates_sat)
    sat_tides_df.columns = ['tide_level']
    sat_tides_df['fn'] = metadata[satname]['filenames']
    
    #sat_tides_df = pd.DataFrame(tide_sat,index=dates_sat)
    #sat_tides_df.columns = ['tide_level']
    #sat_tides_df['fn'] = metadata[satname]['filenames']

#reassign the filepath
filepath = SDS_tools.get_filepath(settings['inputs'],satname)

for i in range(len(filenames)):
#for i in range(20):
    
    # read image
    fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
    im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])

    # calculate cloud cover
    cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                            (cloud_mask.shape[0]*cloud_mask.shape[1]))

    # skip image if cloud cover is above threshold
    if cloud_cover > settings['cloud_thresh']:
        continue

    print(i)
            
    # classify image in 4 classes (sand, vegetation, water, other) with NN classifier
    im_classif, im_labels = SDS_entrance.classify_image_NN(im_ms, im_extra, cloud_mask,
                            min_beach_area_pixels, clf)

    
    # rescale image intensity for display purposes
    #im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    
    if settings_entrance['path_index'] == 'mndwi':
        costSurfaceArray  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
        #adust the mndwi over NN-classified sanad with sand_delta adjustment factor  to assist the path finder
        mask = np.zeros(costSurfaceArray.shape, dtype=float)
        mask[im_classif == 1] = settings_entrance['ndwi_sand_delta']
        costSurfaceArray = costSurfaceArray + mask
        # define the costsurfaceArray for across berm (X) and along berm (AB) analysis direction
        # AB: the routthrougharray algorithm did not work well with negative values so spectral indices are shifted into the positive here via +1
        costSurfaceArray_A = costSurfaceArray + 1
        # XB: for along_berm analysis, we want the 'driest' path over the beach berm so we need to invert the costsurface via division
        costSurfaceArray_B = 1/(costSurfaceArray + 1)
    
    elif settings_entrance['path_index'] == 'ndwi':
        costSurfaceArray  = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
        #adjust ndwi over whitewater with ndwi_whitewhater_delta and over sand with sand_delta to assist the path finder
        mask = np.zeros(costSurfaceArray.shape, dtype=float)
        mask[im_classif == 2] = settings_entrance['ndwi_whitewhater_delta']
        mask[im_classif == 1] = settings_entrance['ndwi_sand_delta']
        costSurfaceArray = costSurfaceArray + mask
        # define the costsurfaceArray for across berm (X) and along berm (AB) analysis direction
        # AB: the routthrougharray algorithm did not work well with negative values so spectral indices are shifted into the positive here via +1
        costSurfaceArray_A = costSurfaceArray + 1
        # XB: for along_berm analysis, we want the 'driest' path over the beach berm so we need to invert the costsurface via division
        costSurfaceArray_B = 1/(SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)+ 1)    #we don't want the sand and whitewash coorection for the along berm analysis     
        
    else:
        costSurfaceArray  = SDS_tools.bathy_index(im_ms[:,:,0], im_ms[:,:,1], cloud_mask) 

    #load all shape and area polygons in pixel coordinates to set up the configuration for the spatial analysis of entrances
    shapes = SDS_tools.load_shapes_as_ndarrays_2(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], satname, sitename, settings['shapefile_EPSG'],
                                       georef, metadata, epsg_dict[filenames[i]] )   
    
    #get the min and max corner (in pixel coordinates) of the entrance area that will be used for plotting the data for visual inspection
    Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box']) 
    
    # define seed and receiver points
    startIndexX, startIndexY = shapes['ocean_seed'][1,:]
    stopIndexX, stopIndexY = shapes['entrance_seed'][1,:]
    startIndexX_B, startIndexY_B = shapes['berm_point_A'][1,:]
    stopIndexX_B, stopIndexY_B = shapes['berm_point_B'][1,:]

    #execute the least cost path search algorithm
    indices, weight = route_through_array(costSurfaceArray_A, (int(startIndexY),int(startIndexX)),
                                          (int(stopIndexY),int(stopIndexX)),geometric=True,fully_connected=True)
    indices_B, weight_B = route_through_array(costSurfaceArray_B, (int(startIndexY_B),int(startIndexX_B)),
                                              (int(stopIndexY_B),int(stopIndexX_B)),geometric=True,fully_connected=True)
             
    #invert the x y values to be in line with the np image array conventions used in coastsat
    indices = list(map(lambda sub: (sub[1], sub[0]), indices))
    indices = np.array(indices)
    indices_B = list(map(lambda sub: (sub[1], sub[0]), indices_B))
    indices_B = np.array(indices_B)
    
    #create indexed raster from indices
    path = np.zeros_like(costSurfaceArray)
    path[indices.T[1], indices.T[0]] = 1
    path_B = np.zeros_like(costSurfaceArray)
    path_B[indices_B.T[1], indices_B.T[0]] = 1
    
    # geospatial processing of the least cost path including coordinate transformations and splitting the path into intervals of 1m
    geoms = []
    # convert pixel coordinates to world coordinates
    pts_world = SDS_tools.convert_pix2world(indices[:,[1,0]], georef)
    #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
    for k in range(len(pts_world)-1):
        #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
        pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
        xvals = np.arange(0,pt_dist)
        yvals = np.zeros(len(xvals))
        pt_coords = np.zeros((len(xvals),2))
        pt_coords[:,0] = xvals
        pt_coords[:,1] = yvals
        phi = 0
        deltax = pts_world[k+1,0] - pts_world[k,0]
        deltay = pts_world[k+1,1] - pts_world[k,1]
        phi = np.pi/2 - np.math.atan2(deltax, deltay)
        tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
        pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
    pts_world_interp = np.delete(pts_world_interp,0,axis=0)
    # convert world image coordinates to user-defined coordinate system
    pts_world_interp_reproj = SDS_tools.convert_epsg(pts_world_interp,  epsg_dict[filenames[i]], settings['output_epsg'])
    pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
    #save as geometry (to create .geojson file later)
    geoms.append(geometry.LineString(pts_world_interp_reproj))
    
    # geospatial processing of the least cost path including coordinate transformations and splitting the path into intervals of 1m
    geoms_B = []
    # convert pixel coordinates to world coordinates
    pts_world_B = SDS_tools.convert_pix2world(indices_B[:,[1,0]], georef)
    #interpolate between array incices to account for different distances across diagonal vs orthogonal pixel paths
    pts_world_interp_B = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
    for k in range(len(pts_world_B)-1):
        #create a path between subsequent pairs of pixel centrepoints into intervals of 1m
        pt_dist = np.linalg.norm(pts_world_B[k,:]-pts_world_B[k+1,:])
        xvals = np.arange(0,pt_dist)
        yvals = np.zeros(len(xvals))
        pt_coords = np.zeros((len(xvals),2))
        pt_coords[:,0] = xvals
        pt_coords[:,1] = yvals
        phi = 0
        deltax = pts_world_B[k+1,0] - pts_world_B[k,0]
        deltay = pts_world_B[k+1,1] - pts_world_B[k,1]
        phi = np.pi/2 - np.math.atan2(deltax, deltay)
        tf = transform.EuclideanTransform(rotation=phi, translation=pts_world_B[k,:])
        pts_world_interp_B = np.append(pts_world_interp_B,tf(pt_coords), axis=0)
    pts_world_interp_B = np.delete(pts_world_interp_B,0,axis=0)
    # convert world image coordinates to user-defined coordinate system
    pts_world_interp_reproj_B = SDS_tools.convert_epsg(pts_world_interp_B,  epsg_dict[filenames[i]], settings['output_epsg'])
    pts_pix_interp_B = SDS_tools.convert_world2pix(pts_world_interp_B, georef)
    #save as geometry (to create .geojson file later)
    geoms_B.append(geometry.LineString(pts_world_interp_reproj_B))
    
    #extract NDWI alon the digitized line.
    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)           
    im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
    #replace the ndwi over whitewater with ndwi minus ndwi_whitewhater_delta
    im_ndwi_adj = np.copy(im_ndwi)
    mask = np.zeros(im_ndwi_adj .shape, dtype=float)
    mask[im_classif == 2] = settings_entrance['ndwi_whitewhater_delta']
    #mask[im_classif == 1] = settings_entrance['ndwi_sand_delta'] # we're not interested in the 'sand adjusted' ndwi values. They are only important for least cost path finding
    im_ndwi_adj = im_ndwi_adj + mask        
    im_bathy_sdb = SDS_tools.bathy_index(im_ms[:,:,0], im_ms[:,:,1], cloud_mask)                              
    
    z_mndwi = scipy.ndimage.map_coordinates(im_mndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
    z_ndwi = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
    z_ndwi_adj = scipy.ndimage.map_coordinates(im_ndwi_adj, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
    z_bathy = scipy.ndimage.map_coordinates(im_bathy_sdb, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
    XS[str(dates_dict[filenames[i]].date())+ '_' + satname + '_mndwi_XB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_mndwi
    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_ndwi_XB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_ndwi
    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_ndwiadj_XB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_ndwi_adj
    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_bathy_XB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_bathy
    
    z_mndwi_B = scipy.ndimage.map_coordinates(im_mndwi, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
    z_ndwi_B = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
    #z_ndwi_B_adj = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
    z_bathy_B = scipy.ndimage.map_coordinates(im_bathy_sdb, np.vstack((pts_pix_interp_B[:,1], pts_pix_interp_B[:,0])),order=1)
    XS[str(dates_dict[filenames[i]].date())+ '_' + satname + '_mndwi_AB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_mndwi_B
    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_ndwi_AB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_ndwi_B
    #XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_ndwiadj_AB'] = z_ndwi_B_adj
    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_bathy_AB_' + str(np.round(sat_tides_df['tide_level'][i],2))] = z_bathy_B

    # also store as .geojson in case user wants to drag-and-drop on GIS for verification
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms))
    gdf.index = [k]
    gdf.loc[k,'name'] = 'entrance_line_XB_' + str(k+1)
    gdf.loc[k,'date'] = filenames[i][:19]
    gdf.loc[k,'satname'] = satname
    gdf.loc[k,'direction'] = 'XB'
    
    gdf_B = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms_B))
    gdf_B.index = [k]
    gdf_B.loc[k,'name'] = 'entrance_line_AB_' + str(k+1)
    gdf_B.loc[k,'date'] = filenames[i][:19]
    gdf_B.loc[k,'satname'] = satname
    gdf_B.loc[k,'direction'] = 'AB'
    
    # store into geodataframe
    if iterator == 0:
        gdf_all = gdf
        gdf_all = gdf_all.append(gdf_B)
    else:
        gdf_all = gdf_all.append(gdf)
        gdf_all = gdf_all.append(gdf_B)
    iterator = iterator + 1         
    
    # generate a distribution of the NDWI for the pixels that are classified as sand by NN
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    
    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_ndwi, im_mndwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)

    # get vector of pixel intensities for each class (first column NDWI, second MNDWI)
    int_water_df = pd.DataFrame(vec_ind[vec_water,:])
    int_sand_df = pd.DataFrame(vec_ind[vec_sand,:])
    int_water_df.columns = ['_ndwi', '_mndwi']
    int_sand_df.columns = ['_ndwi', '_mndwi']
          
    if settings_entrance['plot_bool']:
        linestyle = ['-', '--', '-.']
        # compute classified image
        im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
        im_class = np.copy(im_RGB)
        cmap = cm.get_cmap('tab20c')
        colorpalette = cmap(np.arange(0,13,1))
        colours = np.zeros((3,4))
        colours[0,:] = colorpalette[5]
        colours[1,:] = np.array([204/255,1,1,1])
        colours[2,:] = np.array([0,91/255,1,1])
        for k in range(0,im_labels.shape[2]):
            im_class[im_labels[:,:,k],0] = colours[k,0]
            im_class[im_labels[:,:,k],1] = colours[k,1]
            im_class[im_labels[:,:,k],2] = colours[k,2]  
        im_class = np.where(np.isnan(im_class), 1.0, im_class)
        
        """
        Classifies every pixel in the image in one of 4 classes:
            - sand                                          --> label = 1
            - whitewater (breaking waves and swash)         --> label = 2
            - water                                         --> label = 3
            - other (vegetation, buildings, rocks...)       --> label = 0
        """
        
        #plot the path and save as figure
        fig = plt.figure(figsize=(40,30)) 
        ax=plt.subplot(4,3,1)
        plt.imshow(im_RGB, interpolation="bicubic") 
        plt.rcParams["axes.grid"] = False
        plt.title(str(dates_dict[filenames[i]].date()))
        ax.axis('off')
        plt.xlim(Xmin, Xmax)
        plt.ylim(Ymax,Ymin) 
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp[0,0]+2, pts_pix_interp[0,1]+2,'A',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp[-1,0]+2, pts_pix_interp[-1,1]+2,'B',horizontalalignment='left', color='yellow', fontsize=16)               
        ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
        
        ax=plt.subplot(4,3,2) 
        im_class_b = np.copy(im_class)
        im_class_b[path == 1] = 0
        im_class_b[path_B == 1] = 0   
        #surrogate_img = im_ndwi
        #surrogate_img[path == 1] = 1
        plt.imshow(im_class_b) 
        orange_patch = mpatches.Patch(color=colours[0,:], label='sand', alpha=0.5)
        white_patch = mpatches.Patch(color=colours[1,:], label='whitewater', alpha=0.5)
        blue_patch = mpatches.Patch(color=colours[2,:], label='water', alpha=0.5)
        ax.legend(handles=[orange_patch,white_patch,blue_patch],
                   bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.title('NN classified + path')
        ax.axis('off')
        plt.xlim(Xmin, Xmax)
        plt.ylim(Ymax,Ymin)
        plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
        plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
        
        ax=plt.subplot(4,3,3) 
        if settings_entrance['tide_bool']:
            # plot time-step distribution
            seaborn.kdeplot(tide_fes, shade=True,vertical=False, color='blue',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
            seaborn.kdeplot(sat_tides_df['tide_level'], shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
            plt.xlim(-1,1)
            plt.ylabel('Probability density')
            plt.xlabel('Tides over full period (darkblue) and during images only (lightblue)')
            plt.axvline(x=sat_tides_df['tide_level'][i], color='red', linestyle='dotted', lw=2, alpha=0.9) 
            #plt.text(sat_tides_df['tide_level'][i] , 0.5 ,  'tide @image', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                                   
#                    t = np.array([_.timestamp() for _ in dates_sat]).astype('float64')
#                    delta_t = np.diff(t)
#                    #fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
#                    ax.grid(which='major', linestyle=':', color='0.5')
#                    bins = np.arange(np.min(delta_t)/seconds_in_day, np.max(delta_t)/seconds_in_day+1,1)-0.5
#                    ax.hist(delta_t/seconds_in_day, bins=bins, ec='k', width=1);
#                    ax.set(xlabel='timestep [days]', ylabel='counts',
#                           xticks=n_days*np.arange(0,20),
#                           xlim=[0,50], title='Timestep distribution');
        else:     
            plt.imshow(im_class) 
            orange_patch = mpatches.Patch(color=colours[0,:], label='sand', alpha=0.5)
            white_patch = mpatches.Patch(color=colours[1,:], label='whitewater', alpha=0.5)
            blue_patch = mpatches.Patch(color=colours[2,:], label='water', alpha=0.5)
            ax.legend(handles=[orange_patch,white_patch,blue_patch],
                       bbox_to_anchor=(1, 0.5), fontsize=10)
            plt.title('NN classified')
            ax.axis('off')
            plt.xlim(Xmin, Xmax)
            plt.ylim(Ymax,Ymin)
        
        ax=plt.subplot(4,3,4) 
        plt.imshow(im_mndwi, cmap='seismic', vmin=-1, vmax=1) 
        ax.axis('off')
        plt.rcParams["axes.grid"] = False
        plt.title('modified NDWI')
        #plt.colorbar()
        plt.xlim(Xmin, Xmax)
        plt.ylim(Ymax,Ymin) 
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
        ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
        
        ax=plt.subplot(4,3,5) 
        plt.imshow(im_ndwi, cmap='seismic', vmin=-1, vmax=1) 
        ax.axis('off')
        plt.title('NDWI')
        #plt.colorbar()
        plt.rcParams["axes.grid"] = False
        plt.xlim(Xmin, Xmax)
        plt.ylim(Ymax,Ymin) 
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
        ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
        ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
        ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
        plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
        plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
        
        ax=plt.subplot(4,3,6) 
        if settings_entrance['tide_bool']:               
            # plot tide time-series
            #fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
            ax.set_title('Tide level at img aquisition = ' + str(np.round(sat_tides_df['tide_level'][i],2)) +  ' [m aMSL]')
            ax.grid(which='major', linestyle=':', color='0.5')
            ax.plot(dates_fes, tide_fes, '-', color='0.6')
            ax.plot(dates_sat, tide_sat, '-o', color='k', ms=4, mfc='w',lw=1)
            plt.axhline(y=sat_tides_df['tide_level'][i], color='red', linestyle='dotted', lw=2, alpha=0.9) 
            ax.plot(sat_tides_df.index[i], sat_tides_df['tide_level'][i], '-o', color='red', ms=12, mfc='w',lw=7)
            ax.set_ylabel('tide level [m]')
            ax.set_ylim(SDS_slope.get_min_max(tide_fes))
            #plt.text(sat_tides_df['tide_level'][i] , 0.5 ,  'tide @image', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])     
            
        else:
            plt.imshow(im_bathy_sdb, cmap='seismic', vmin=0.9, vmax=1.1) 
            ax.axis('off')
            plt.title('blue/green')
            #plt.colorbar()
            plt.rcParams["axes.grid"] = False
            plt.xlim(Xmin, Xmax)
            plt.ylim(Ymax,Ymin) 
            ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--', color='yellow')
            ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko', color='yellow')
            ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='yellow')
            plt.text(pts_pix_interp[0,0]+1, pts_pix_interp[0,1]+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
            plt.text(pts_pix_interp[-1,0]+1, pts_pix_interp[-1,1]+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
            ax.plot(pts_pix_interp_B[:,0], pts_pix_interp_B[:,1], 'r--', color='yellow')
            ax.plot(pts_pix_interp_B[0,0], pts_pix_interp_B[0,1],'ko', color='yellow')
            ax.plot(pts_pix_interp_B[-1,0], pts_pix_interp_B[-1,1],'ko', color='yellow')
            plt.text(pts_pix_interp_B[0,0]+2, pts_pix_interp_B[0,1]+2,'C',horizontalalignment='left', color='yellow' , fontsize=16)
            plt.text(pts_pix_interp_B[-1,0]+2, pts_pix_interp_B[-1,1]+2,'D',horizontalalignment='left', color='yellow', fontsize=16)
        
        ax=plt.subplot(4,3,9)
        seaborn.kdeplot(int_water_df.filter(regex='_mndwi').iloc[:,0], shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
        #plt.xlim(int_water_df.filter(regex='_mndwi').min(),int_sand_df.filter(regex='_mndwi').max())
        plt.xlim(-1,1)
        seaborn.kdeplot(int_sand_df.filter(regex='_mndwi').iloc[:,0], shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
        plt.ylabel('Probability density')
        plt.xlabel('mNDWI over sand (orange) and water (blue) classes')
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10:
            img_mndwi_otsu = filters.threshold_otsu(pd.DataFrame(int_water_df.filter(regex='_mndwi').iloc[:,0]).append(pd.DataFrame(int_sand_df.filter(regex='_mndwi').iloc[:,0])).iloc[:,0].values)    
            img_mndwi_perc = np.nanpercentile(int_sand_df.filter(regex='_mndwi').iloc[:,0].values, settings_entrance['sand_percentile'] )
            img_mndwi_perc10 = np.nanpercentile(int_sand_df.filter(regex='_mndwi').iloc[:,0].values, 10 )
            img_mndwi_perc20 = np.nanpercentile(int_sand_df.filter(regex='_mndwi').iloc[:,0].values, 20 ) 
            img_mndwi_perc30 = np.nanpercentile(int_sand_df.filter(regex='_mndwi').iloc[:,0].values, 30 )  
            plt.axvline(x=img_mndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_mndwi_perc , 0.5 ,  str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                  
            plt.axvline(x=img_mndwi_perc10, color='red', linestyle=linestyle[0], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_mndwi_perc10 , 0.5 , '10p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])
            plt.axvline(x=img_mndwi_perc20, color='red', linestyle=linestyle[1], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_mndwi_perc20 , 0.5 , '20p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])    
            plt.axvline(x=img_mndwi_perc30, color='red', linestyle=linestyle[2], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_mndwi_perc30 , 0.5 , '30p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])   
            plt.axvline(x=img_mndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_mndwi_otsu , 0.5 , 'OTSU', rotation=90, ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])
        
        ax=plt.subplot(4,3,7)   
        pd.DataFrame(z_mndwi).plot(color='blue', linestyle='--', ax=ax)     
        plt.ylim(-0.9,0.9)
        #plt.title('modified NDWI along transect') 
        #plt.legend()
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10:
            plt.axhline(y=img_mndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_mndwi_perc10, color='red', linestyle=linestyle[0], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_mndwi_perc20, color='red', linestyle=linestyle[1], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_mndwi_perc30, color='red', linestyle=linestyle[2], lw=1, alpha=settings_entrance['vhline_transparancy'])                
            plt.axhline(y=img_mndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy'])    
        plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
        plt.text(len(z_mndwi)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
        plt.xlabel('Distance along transect [m]')
        plt.ylabel('mNDWI [-]')
        ax.get_legend().remove()
        
        ax=plt.subplot(4,3,8)
        pd.DataFrame(z_mndwi_B).plot(color='blue', linestyle='--', ax=ax)     
        plt.ylim(-0.9,0.9)
        #plt.title('modified NDWI along transect') 
        #plt.legend()
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10:    
            plt.axhline(y=img_mndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_mndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
        plt.text(len(z_mndwi_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
        plt.xlabel('Distance along transect [m]')
        plt.ylabel('mNDWI [-]')
        ax.get_legend().remove()            
        
        ax=plt.subplot(4,3,12)
        seaborn.kdeplot(int_water_df.filter(regex='_ndwi').iloc[:,0], shade=True,vertical=False, color='lightblue',bw=settings_entrance['hist_bw'], legend=False, lw=2, ax=ax)
        plt.xlim(-1,1)
        seaborn.kdeplot(int_sand_df.filter(regex='_ndwi').iloc[:,0], shade=True,vertical=False, color='orange',bw=settings_entrance['hist_bw'],legend=False, lw=2, ax=ax)
        plt.ylabel('Probability density')
        plt.xlabel('NDWI over sand (orange) and water (blue) classes')
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10:
            img_ndwi_otsu = filters.threshold_otsu(pd.DataFrame(int_water_df.filter(regex='_ndwi').iloc[:,0]).append(pd.DataFrame(int_sand_df.filter(regex='_ndwi').iloc[:,0])).iloc[:,0].values)
            img_ndwi_perc = np.nanpercentile(int_sand_df.filter(regex='_ndwi').iloc[:,0].values, settings_entrance['sand_percentile'])
            img_ndwi_perc10 = np.nanpercentile(int_sand_df.filter(regex='_ndwi').iloc[:,0].values, 10 )
            img_ndwi_perc20 = np.nanpercentile(int_sand_df.filter(regex='_ndwi').iloc[:,0].values, 20 )
            img_ndwi_perc30 = np.nanpercentile(int_sand_df.filter(regex='_ndwi').iloc[:,0].values, 30 ) 
            plt.axvline(x=img_ndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_ndwi_perc , 0.5 ,str(settings_entrance['sand_percentile']) + 'p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                  
            plt.axvline(x=img_ndwi_perc10, color='red', linestyle=linestyle[0], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_ndwi_perc10 , 0.5 , '10p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])
            plt.axvline(x=img_ndwi_perc20, color='red', linestyle=linestyle[1], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_ndwi_perc20 , 0.5 , '20p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy']) 
            plt.axvline(x=img_ndwi_perc30, color='red', linestyle=linestyle[2], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_ndwi_perc30 , 0.5 , '30p', rotation=90 , ha='right', va='bottom', alpha=settings_entrance['vhline_transparancy'])                 
            plt.axvline(x=img_ndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.text(img_ndwi_otsu , 0.5 , 'OTSU sand-water', rotation=90, ha='right', va='bottom', alpha= settings_entrance['vhline_transparancy'])                
        
        ax=plt.subplot(4,3,10)
        pd.DataFrame(z_ndwi).plot(color='lightblue', linestyle='--', ax=ax) 
        pd.DataFrame(z_ndwi_adj).plot(color='blue', linestyle='--', ax=ax) 
        plt.ylim(-0.9,0.9)
        plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10:   
            plt.axhline(y=img_ndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_ndwi_perc10, color='red', linestyle=linestyle[0], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_ndwi_perc20, color='red', linestyle=linestyle[1], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_ndwi_perc30, color='red', linestyle=linestyle[2], lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_ndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
        plt.text(len(z_mndwi)-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
        plt.xlabel('Distance along transect [m]')
        plt.ylabel('NDWI [-]')
        ax.get_legend().remove()
        
        ax=plt.subplot(4,3,11)
        pd.DataFrame(z_ndwi_B).plot(color='blue', linestyle='--', ax=ax) 
        #pd.DataFrame(z_ndwi_B_adj).plot(color='blue', linestyle='--', ax=ax) 
        plt.ylim(-0.9,0.9)
        plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        if len(int_sand_df.filter(regex='_ndwi').iloc[:,0].values) > 10: 
            plt.axhline(y=img_ndwi_perc, color='orangered', linestyle='dotted', lw=1, alpha=settings_entrance['vhline_transparancy']) 
            plt.axhline(y=img_ndwi_otsu, color='limegreen', linestyle='--', lw=1, alpha=settings_entrance['vhline_transparancy']) 
        plt.text(1,0,'C',horizontalalignment='left', color='grey' , fontsize=labelsize)
        plt.text(len(z_mndwi_B)-2,0,'D',horizontalalignment='right', color='grey' , fontsize=labelsize)
        plt.xlabel('Distance along transect [m]')
        plt.ylabel('NDWI [-]')
        ax.get_legend().remove()
 
        fig.tight_layout() 
        fig.savefig(image_out_path + '/' + filenames[i][:-4]  + '_' + settings_entrance['path_index'] +'_based.png') 
        plt.close('all') 
  
#gdf_all.crs = {'init':'epsg:'+str(image_epsg)} # looks like mistake. geoms as input to the dataframe should already have the output epsg. 
gdf_all.crs = {'init': 'epsg:'+ str(settings['output_epsg'])}
# convert from image_epsg to user-defined coordinate system
#gdf_all = gdf_all.to_crs({'init': 'epsg:'+str(settings['output_epsg'])})
# save as shapefile
gdf_all.to_file(os.path.join(csv_out_path, sitename + '_entrance_lines_auto.shp'), driver='ESRI Shapefile') 
print('Entrance lines have been saved as a shapefile and in csv format for NDWI and mNDWI')

#save the data to csv            
XS_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in XS.items() ])) 
XS_df.to_csv(os.path.join(csv_out_path, sitename + '_XS' + '_entrance_lines_auto.csv')) 
 
    



#%%  Step 3 plot figures:


#%%Figure A
"""
########################################################################
Figure A - show only images & different spectral transects.
The idea behind this figure is to just illustrate the concept of the 
methodology by showing 3 open and 3 closed images and the corresponding
spectral transects for three spectral indices
This plot is interactive and require a number of user inputs
########################################################################
"""
#Set up the plot for the key method figure of the paper
satname = 'S2'
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = metadata[satname]['filenames']
epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))    

figure_A_out_path = os.path.join(csv_out_path, 'figure_A')
if not os.path.exists(figure_A_out_path ):
        os.makedirs(figure_A_out_path) 
        
#plot font size and type
ALPHA_figs = 1
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)     

#plot settings
nrows = 5 
xaxisadjust = 20
Interpolation_method = "bicubic" #"None" #"bicubic")
labelsize = 26
spectral_index = 'mndwi'

linestyle = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.','-', '--', '-.', '-', '--', '-.',
             '-', '--', '-.','-', '--', '-.', '-', '--', '-.', '-', '--', '-.']

for j in range(0, len(XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns)-1, 3):
    k = 1
    fig = plt.figure(figsize=(21,30))   
    if j> min(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns), len(XS_o_df.filter(regex=satname).filter(regex='_mndwi').columns))-2:
        j = j-2
    print('processed ' + str(j))
    for i in range(j,j+3,1):
        plot_identifier = 'single_index'
    #for i in range(0,3,1):
        ####################################
        #Plot the closed entrance states
        ####################################
        Loopimage_c_date = XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns[i][:-9]
        r = re.compile(".*" + Loopimage_c_date)    
        #combined_df.filter(regex=Gauge[3:]+'_')
        fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, satname)
        #print(fn)
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
        # rescale image intensity for display purposes

        if spectral_index == 'mndwi':
            im_plot  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask) 
        elif spectral_index == 'ndwi':
            im_plot  = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
        else:
            im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            
        image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
        shapes = SDS_tools.load_shapes_as_ndarrays_2(layers,Site_shps, satname, sitename, settings['shapefile_EPSG'],
                                           georef, metadata, image_epsg)        
        x0, y0 = shapes['ocean_seed'][1,:]
        x1, y1 = shapes['entrance_seed'][1,:]
        Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box'])
    
        ax=plt.subplot(nrows,2,k)
        plt.title(satname + ' ' + Loopimage_c_date + ' '  + spectral_index + ' closed') 
    
        if spectral_index in ['mndwi', 'ndwi']:
            plt.imshow(im_plot, cmap='seismic', vmin=-1, vmax=1) 
            plt.colorbar()
        else:
            plt.imshow(im_plot, interpolation=Interpolation_method) 
        
        plt.rcParams["axes.grid"] = False
        plt.xlim(Xmin-xaxisadjust , Xmax+xaxisadjust)
        plt.ylim(Ymax,Ymin) 
        #ax.grid(None)
        ax.axis('off')
        #plt.plot(x0, y0, 'ro', color='yellow', marker="X")
        #plt.plot(x1, y1, 'ro', color='yellow', marker="D")

        #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
        line = list(XS_c_gdf[XS_c_gdf['date'].str.contains(Loopimage_c_date)].geometry.iloc[0].coords)
        #df = pd.DataFrame(line).drop(columns=2).values 
        pts_world_interp_reproj = SDS_tools.convert_epsg(pd.DataFrame(line).drop(columns=2).values , settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=linestyle[i], color='lightblue')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color='lightblue')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='lightblue')  
        plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color='lightblue' , fontsize=labelsize)
        plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color='lightblue', fontsize=labelsize)
        
        ####################################
        #Plot the open entrance states
        ####################################
        Loopimage_o_date = XS_o_df.filter(regex=satname).filter(regex='_mndwi').columns[i][:-9]
        r = re.compile(".*" + Loopimage_o_date)    
        #combined_df.filter(regex=Gauge[3:]+'_')
        fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, satname)
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
        
        # rescale image intensity for display purposes
        if spectral_index == 'mndwi':
            im_plot  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask) 
        elif spectral_index == 'ndwi':
            im_plot  = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
        else:
            im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
        
        image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
        shapes = SDS_tools.load_shapes_as_ndarrays_2(layers,Site_shps, satname, sitename, settings['shapefile_EPSG'],
                                           georef, metadata, image_epsg)        
        x0, y0 = shapes['ocean_seed'][1,:]
        x1, y1 = shapes['entrance_seed'][1,:]
        Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box'])
        
        ax=plt.subplot(nrows,2,k+1) 
        k=k+2
        ax.axis('off')
        plt.title(satname + ' ' + Loopimage_o_date + ' '  + spectral_index + ' open') 
        
        if spectral_index in ['mndwi', 'ndwi']:
            plt.imshow(im_plot, cmap='seismic', vmin=-1, vmax=1) 
            plt.colorbar()
        else:
            plt.imshow(im_plot, interpolation=Interpolation_method) 
        
        plt.rcParams["axes.grid"] = False
        plt.grid(None)
        plt.xlim(Xmin-xaxisadjust , Xmax+xaxisadjust)
        plt.ylim(Ymax,Ymin) 
        #plt.plot(x0, y0, 'ro', color='yellow', marker="X")
        #plt.plot(x1, y1, 'ro', color='yellow', marker="D")
        
        #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
        line = list(XS_o_gdf[XS_o_gdf['date'].str.contains(Loopimage_o_date)].geometry.iloc[0].coords)
        df = pd.DataFrame(line)
        df = df.drop(columns=2)      
        pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
        df2 = pd.DataFrame(pts_world_interp_reproj)
        df2 = df2.drop(columns=2)  
        pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=linestyle[i], color='orange')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko' , linestyle=linestyle[i] , color='orange')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color='orange' )
        plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color='orange' , fontsize=labelsize)
        plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color='orange' , fontsize=labelsize)
    
        #plt.xlim(400,500)
        #plt.ylim( 860,760)
        #plt.plot([XS_array_pix[0,0], XS_array_pix[len(XS_array_pix)-1,0]], [XS_array_pix[0,1], XS_array_pix[len(XS_array_pix)-1,1]], 'ro-', color='g', lw=1)
        ax=plt.subplot(nrows,1,4)
        #Path_mndwi_c_sat_df_1date = Path_mndwi_c_sat_df.filter(regex=Loopimage_o_date)
        #XS_c_df.filter(regex=satname).filter(regex='_'+spectral_index).filter(regex=Loopimage_c_date).plot(color='lightblue', linestyle=linestyle[i], ax=ax)
        #XS_o_df.filter(regex=satname).filter(regex='_'+spectral_index).filter(regex=Loopimage_o_date).plot(color='orange',  linestyle=linestyle[i],ax=ax)
        
        XS_c_df.filter(regex=satname).filter(regex='_mndwi').filter(regex=Loopimage_c_date).plot(color='lightblue', linestyle=linestyle[i], ax=ax)
        XS_o_df.filter(regex=satname).filter(regex='_mndwi').filter(regex=Loopimage_o_date).plot(color='orange',  linestyle=linestyle[i],ax=ax)        
        plt.ylim(-0.9,0.9)
        plt.title('modified NDWI extracted along each transect shown above') 
        #plt.legend()
        plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
        plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
        plt.text(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi'))-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
        plt.xlabel('Distance along transect [m]')
        plt.ylabel('mNDWI [-]')
        ax.get_legend().remove()
    
        if nrows > 4:
            plot_identifier = 'two_indices'
            ax=plt.subplot(nrows,1,5)
            XS_c_df.filter(regex=satname).filter(regex='_ndwi').filter(regex=Loopimage_c_date).plot(color='lightblue', linestyle=linestyle[i], ax=ax)
            XS_o_df.filter(regex=satname).filter(regex='_ndwi').filter(regex=Loopimage_o_date).plot(color='orange',  linestyle=linestyle[i],ax=ax)
            plt.ylim(-0.9,0.9)
            plt.title('NDWI') 
            #plt.legend()
            plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5)   
            plt.xlabel('Distance along transect')
            plt.ylabel('NDWI')
            ax.get_legend().remove()
        
            if nrows >5:
                plot_identifier = 'three_indices'
                ax=plt.subplot(nrows,1,6)
                XS_c_df.filter(regex=satname).filter(regex='_bathy').filter(regex=Loopimage_c_date).plot(color='lightblue', linestyle=linestyle[i], ax=ax)
                XS_o_df.filter(regex=satname).filter(regex='_bathy').filter(regex=Loopimage_o_date).plot(color='orange',  linestyle=linestyle[i],ax=ax)
                plt.ylim(0.8,1.2)
                plt.title('blue gren ratio') 
                #plt.legend()
                plt.axhline(y=1, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
                plt.xlabel('Distance along transect')
                plt.ylabel('Blue/Green')
                ax.get_legend().remove()
                
    #save the figure            
    fig.tight_layout() 
    fig.savefig(os.path.join(figure_A_out_path, satname + '_Figure_A_' +  plot_identifier + '_' + str(j) + '_'+ spectral_index +'_' + datetime.now().strftime("%d-%m-%Y") +'.png'))     
    plt.close('all')    


#%% Figure B
"""
########################################################################
Figure B - This is currently still in development and might be dropped later

We need a figure where we show the time series of open vs. closed alongside the summary stats obatained for each image
########################################################################
"""
#Set up the plot for the key method figure of the paper
 



#%% Figure C
"""
########################################################################
Figure C - show only 2 images & mNDWI all transects & scatterplots
The idea here is to show a lot more transects, possibly hundrets as found
by the least cost path marching algorithm, spatially on one open and one closed
image as well as in line and scatterplots. I'd like to show absolute mNDWI and 
slope + the corresponding total cost along each transect in scatterplot 
This plot is interactive and require a number of user inputs
########################################################################
"""

from matplotlib.pyplot import cm
import numpy as np
import seaborn
import skimage.filters as filters

#Set up the plot for the key method figure of the paper
satname = 'S2'
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = metadata[satname]['filenames']
epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))     

figure_B_out_path = os.path.join(csv_out_path, 'figure_C')
if not os.path.exists(figure_B_out_path ):
        os.makedirs(figure_B_out_path) 
        
#plot font size and type
ALPHA_figs = 1
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

plt.close('all')      

#plot settings
fig = plt.figure(figsize=(35,30))   
nrows = 3 
xaxisadjust = 0
Interpolation_method = "bicubic" #"None" #"bicubic")
labelsize = 26
spectral_index = 'ndwi'
k = 1
linestyle = ['-', '--', '-.']
linestyletype = 1
bandwidth = 0.1
slope_cumulator = 'sum' #max, sum, deltaminmax


####################################
#Plot the closed entrance states
####################################

# row 1 pos 2   ################################################################################  
Loopimage_c_date = XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns[0][:-9]
r = re.compile(".*" + Loopimage_c_date)    
#combined_df.filter(regex=Gauge[3:]+'_')
fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, satname)
print('plotting Figure C')
im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
# rescale image intensity for display purposes  

#if spectral_index == 'mndwi':
#    im_plot  = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
#    plt.colorbar()
#elif spectral_index == 'ndwi':
#    im_plot  = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
#    plt.colorbar()
#else:
im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
    
image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
shapes = SDS_tools.load_shapes_as_ndarrays_2(layers,Site_shps, satname, sitename, settings['shapefile_EPSG'],
                                   georef, metadata, image_epsg)        
x0, y0 = shapes['ocean_seed'][1,:]
x1, y1 = shapes['entrance_seed'][1,:]
Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box'])

ax=plt.subplot(nrows,2,1)
plt.title(satname + ' ' + Loopimage_c_date + ' closed') 

plt.imshow(im_plot, interpolation=Interpolation_method) 

plt.rcParams["axes.grid"] = False
plt.xlim(Xmin-xaxisadjust , Xmax+xaxisadjust)
plt.ylim(Ymax,Ymin) 
#ax.grid(None)
ax.axis('off')

n=len(XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns)
color=iter(cm.Blues(np.linspace(0,1,n)))
for i in range(0,n,1):    
    #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
    line = list(XS_c_gdf[XS_c_gdf['date'].str.contains(XS_c_df.filter(regex=satname).filter(regex='_mndwi').columns[i][:-9])].geometry.iloc[0].coords)
    df = pd.DataFrame(line)
    df = df.drop(columns=2)      
    pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
    df2 = pd.DataFrame(pts_world_interp_reproj)
    df2 = df2.drop(columns=2)  
    pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
    c= next(color)
    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=linestyle[0], color=c)
    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
    if i==0:
        plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color='lightblue' , fontsize=labelsize)
        plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color='lightblue', fontsize=labelsize)


####################################
#Plot the open entrance states
#################################### 
        
# row 1 pos 2   ################################################################################         
Loopimage_c_date = XS_o_df.filter(regex=satname).filter(regex='_mndwi').columns[0][:-9]
r = re.compile(".*" + Loopimage_c_date)    
#combined_df.filter(regex=Gauge[3:]+'_')
fn = SDS_tools.get_filenames(list(filter(r.match, filenames))[0],filepath, satname)
#print(fn)
im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
# rescale image intensity for display purposes

im_plot = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)  
    
image_epsg = epsg_dict[list(filter(r.match, filenames))[0]]
shapes = SDS_tools.load_shapes_as_ndarrays_2(layers,Site_shps, satname, sitename, settings['shapefile_EPSG'],
                                   georef, metadata, image_epsg)        
x0, y0 = shapes['ocean_seed'][1,:]
x1, y1 = shapes['entrance_seed'][1,:]
Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box'])

ax=plt.subplot(nrows,2,2)
plt.title(satname + ' ' + Loopimage_c_date + ' open') 

plt.imshow(im_plot, interpolation=Interpolation_method) 

plt.rcParams["axes.grid"] = False
plt.xlim(Xmin-xaxisadjust , Xmax+xaxisadjust)
plt.ylim(Ymax,Ymin) 
#ax.grid(None)
ax.axis('off')

n=len(XS_o_df.filter(regex=satname).filter(regex='_mndwi').columns)
color=iter(cm.Oranges(np.linspace(0,1,n)))
for i in range(0,n,1):
    #plot the digitized entrance paths on top of images - failing due to mixed up epsgs
    line = list(XS_o_gdf[XS_o_gdf['date'].str.contains(XS_o_df.filter(regex=satname).filter(regex='_mndwi').columns[i][:-9])].geometry.iloc[0].coords)
    df = pd.DataFrame(line)
    df = df.drop(columns=2)      
    pts_world_interp_reproj = SDS_tools.convert_epsg(df.values, settings['output_epsg'], image_epsg)
    df2 = pd.DataFrame(pts_world_interp_reproj)
    df2 = df2.drop(columns=2)  
    pts_pix_interp = SDS_tools.convert_world2pix(df2.values, georef)
    c= next(color)
    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], linestyle=linestyle[0], color=c)
    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko',   color=c)
    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko', color=c)  
    if i==0:
        plt.text(pts_pix_interp[0,0]+3, pts_pix_interp[0,1]+3,'A',horizontalalignment='left', color='orange' , fontsize=labelsize)
        plt.text(pts_pix_interp[-1,0]+3, pts_pix_interp[-1,1]+3,'B',horizontalalignment='left', color='orange', fontsize=labelsize)
        

# row 2 pos 1  ################################################################################          
#plot the mNDWI transects
ax=plt.subplot(nrows,3,4)
XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).plot(color='lightblue', linestyle=linestyle[linestyletype], ax=ax)
XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).plot(color='orange',  linestyle=linestyle[linestyletype],ax=ax)
plt.ylim(-0.9,0.9)
#plt.title('spectral_index + ' along transects') 
#plt.legend()
plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
plt.text(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi'))-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
plt.xlabel('Distance along transects [m]')
plt.ylabel(spectral_index)
ax.get_legend().remove()  


# row 2 pos 2  ################################################################################   
#plot the mNDWI sums along the transects over time
ax=plt.subplot(nrows,3,5)

if slope_cumulator == 'sum':
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).sum()
elif slope_cumulator == 'deltaminmax':
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).max() - XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).min()
else:
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).max()
    
newindex = {}
for index in XS_c_sums_df.index:
    if spectral_index == 'mndwi':
        newindex[index]= pd.to_datetime(index[:-9], format = '%Y-%m-%d')
    if spectral_index == 'ndwi':
        newindex[index]= pd.to_datetime(index[:-8], format = '%Y-%m-%d')
XS_c_sums_df.index = newindex.values()
XS_c_sums_df = pd.DataFrame(XS_c_sums_df)
XS_c_sums_df.plot(color='lightblue',style='.--',  ax=ax)

if slope_cumulator == 'sum':
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).sum()
elif slope_cumulator == 'deltaminmax':
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).max() - XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).min()
else:
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).max()
    
newindex = {}
for index in XS_o_sums_df.index:
    if spectral_index == 'mndwi':
        newindex[index]= pd.to_datetime(index[:-9], format = '%Y-%m-%d')
    if spectral_index == 'ndwi':
        newindex[index]= pd.to_datetime(index[:-8], format = '%Y-%m-%d')
XS_o_sums_df.index = newindex.values()
XS_o_sums_df = pd.DataFrame(XS_o_sums_df)
XS_o_sums_df.plot(color='orange', style='.--',ax=ax)

#df = XS_o_sums_df
#df.reset_index(inplace=True)
#df.columns = ['time','value']
#df.plot(kind='scatter',x='time',y='value')
        
plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
#plt.title('mNDWI along transects') 
#plt.legend()
#plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
#plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
#plt.text(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi'))-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
#plt.xlabel('Time')
plt.ylabel(slope_cumulator +' of ' +spectral_index + ' along transects')
ax.get_legend().remove()  


# row 2 pos 3  ################################################################################   
#plot the densities of the sums along each transect
ax=plt.subplot(nrows,3,6)
#seaborn.kdeplot(XS_c_df.filter(regex=satname).filter(regex='_mndwi').dropna().iloc[:,0], shade=True,vertical=True, color='orange',bw=bandwidth, lw=2, ax=ax)
seaborn.kdeplot(XS_o_sums_df.iloc[:,0], shade=True,vertical=True, color='orange',bw=bandwidth, lw=2, ax=ax)
plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
seaborn.kdeplot(XS_c_sums_df.iloc[:,0], shade=True,vertical=True, color='lightblue',bw=bandwidth, lw=2, ax=ax)
plt.axhline(y=filters.threshold_otsu(pd.DataFrame(XS_o_sums_df.iloc[:,0]).append(pd.DataFrame(XS_c_sums_df.iloc[:,0])).iloc[:,0].values),
            color='grey', linestyle='--', lw=1, alpha=0.5) 
#plt.text( xlocation , (np.nanpercentile(TS_1,87)), '13% exceedance', ha='left', va='bottom')
#plt.axhline(y=np.nanpercentile(TS_1,87), color=color_1, linestyle='solid', lw=width, alpha=1) 
plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
plt.xlabel('Probability density')
plt.ylabel(slope_cumulator +' of ' +spectral_index + ' along transects')
    

# row 3 pos 1  ################################################################################          
#plot the mNDWI slope transects
ax=plt.subplot(nrows,3,7)
XS_c_df.diff().filter(regex=satname).filter(regex='_' + spectral_index).plot(color='lightblue', linestyle=linestyle[linestyletype], ax=ax)
XS_o_df.diff().filter(regex=satname).filter(regex='_' + spectral_index).plot(color='orange',  linestyle=linestyle[linestyletype],ax=ax)
plt.ylim(XS_c_df.diff().min().min(),XS_c_df.diff().max().max())
#plt.title('Gradient of modified NDWI along transects') 
#plt.legend()
plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
plt.text(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi'))-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
plt.xlabel('Distance along transects [m]')
plt.ylabel('Slope of ' + spectral_index)
ax.get_legend().remove()     
 

# row 3 pos 2  ################################################################################      
#plot the mNDWI sums along the transects over time
ax=plt.subplot(nrows,3,8)

if slope_cumulator == 'sum':
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().abs().sum()
elif slope_cumulator == 'deltaminmax':
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().max() - XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().min()
else:
    XS_c_sums_df = XS_c_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().max()

newindex = {}
for index in XS_c_sums_df.index:
    if spectral_index == 'mndwi':
        newindex[index]= pd.to_datetime(index[:-9], format = '%Y-%m-%d')
    if spectral_index == 'ndwi':
        newindex[index]= pd.to_datetime(index[:-8], format = '%Y-%m-%d')
XS_c_sums_df.index = newindex.values()
XS_c_sums_df = pd.DataFrame(XS_c_sums_df)
XS_c_sums_df.plot(color='lightblue',style='.--',  ax=ax)




if slope_cumulator == 'sum':
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().abs().sum() 
elif slope_cumulator == 'deltaminmax':
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().max() - XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().min()
else:
    XS_o_sums_df = XS_o_df.filter(regex=satname).filter(regex='_' + spectral_index).diff().abs().max()
    
newindex = {}
for index in XS_o_sums_df.index:
    if spectral_index == 'mndwi':
        newindex[index]= pd.to_datetime(index[:-9], format = '%Y-%m-%d')
    if spectral_index == 'ndwi':
        newindex[index]= pd.to_datetime(index[:-8], format = '%Y-%m-%d')
XS_o_sums_df.index = newindex.values()
XS_o_sums_df = pd.DataFrame(XS_o_sums_df)
XS_o_sums_df.plot(color='orange', style='.--', ax=ax)


plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
#plt.title('max slope along mNDWI transects') 
#plt.legend()
#plt.axhline(y=0, xmin=-1, xmax=1, color='grey', linestyle='--', lw=1, alpha=0.5) 
#plt.text(1,0,'A',horizontalalignment='left', color='grey' , fontsize=labelsize)
#plt.text(len(XS_c_df.filter(regex=satname).filter(regex='_mndwi'))-2,0,'B',horizontalalignment='right', color='grey' , fontsize=labelsize)
#plt.xlabel('Time')
plt.ylabel(slope_cumulator +' of abs(slope) along ' + spectral_index + ' transects')
ax.get_legend().remove()   


# row 3 pos 3  ################################################################################  
#plot the densities of the sums along each transect
ax=plt.subplot(nrows,3,9)
#seaborn.kdeplot(XS_c_df.filter(regex=satname).filter(regex='_mndwi').dropna().iloc[:,0], shade=True,vertical=True, color='orange',bw=bandwidth, lw=2, ax=ax)
seaborn.kdeplot(XS_o_sums_df.iloc[:,0], shade=True,vertical=True, color='orange',bw=0.05, legend=False, lw=2, ax=ax)
plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
seaborn.kdeplot(XS_c_sums_df.iloc[:,0], shade=True,vertical=True, color='lightblue',bw=0.05,legend=False, lw=2, ax=ax)
plt.xlabel('Probability density')
plt.ylabel(slope_cumulator +' of abs(slope) along ' + spectral_index + ' transects')
plt.ylim(XS_o_sums_df.min().min(),XS_c_sums_df.max().max())
plt.axhline(y=filters.threshold_otsu(pd.DataFrame(XS_o_sums_df.iloc[:,0]).append(pd.DataFrame(XS_c_sums_df.iloc[:,0])).iloc[:,0].values),
            color='grey', linestyle='--', lw=1, alpha=0.5) 
#plt.text( xlocation , (np.nanpercentile(TS_1,87)), '13% exceedance', ha='left', va='bottom')
    
fig.tight_layout()        
fig.savefig(os.path.join(figure_B_out_path, satname + '_Figure_C_'  + spectral_index +'_' + slope_cumulator + '_based_'+ datetime.now().strftime("%d-%m-%Y") +'.png'))          

"""
########################################################################
End of figure C
########################################################################
"""      
        




#%% Step 4: Run validation of automated entrance state detection vs training data and provide user outputs               
            
 


        











