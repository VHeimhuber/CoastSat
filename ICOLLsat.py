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
Analysis_version = 'V3' #this is a short user defined identifier that is added to end of directories to allow for multiple analysis to be done for each site with different parameters. 

# name of the site
sitename = 'DURRAS'

# date range
dates = ['1985-01-01', '2020-08-01']

# satellite missions
sat_list = ['L5','L7','L8','S2']
sat_list = ['S2']

# filepath where data will be stored
filepath_data = os.path.join('H:/WRL_Projects/Estuary_sat_data', 'data')

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
    'check_detection': True,    # if True, shows each entrance state detection to the user for validation #####!!!!!##### Intermediate - change variable to manual_input
    'shuffle_training_imgs':True,  # if True, images durin manual/visual detection of entrance states are shuffled in time to provide a more independent sample
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
Training_data_df = SDS_entrance.create_training_data(metadata, settings)
    

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
                      'Experiment_code': 'Exp_11_S2',  #unique identifier for the experiment. Outputs will be stored in this sub folder
                      'sand_percentile': 40 ,                 #percentile of sand to plot in addition to 10th, 20th and 30th (which are always done)
                      'ndwi_whitewhater_delta': -0.2,       #where the NN classifier detects whitewater, NDWI will be adjusted by this value to facilitate least cost path finding
                      'ndwi_sand_delta': 0.5 ,                   #where the NN classifier detects sand, NDWI will be adjusted by this value to facilitate least cost path finding
                      'vhline_transparancy': 0.8 ,          #transparancy of v and h lines in the output plots
                      'hist_bw': 0.1,                       #parameter for histogram smoothing in the output plots
                      'tide_bool': True ,                   #include FES-based analysis of tides?
                      'plot_bool': True  ,                  #create the output plots in addition to csvs?
                      }

XS_c_df, XS_c_gdf,geoms, sat_tide_df = SDS_entrance.automated_entrance_paths(metadata, settings, settings_entrance) 


    



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
            
 


        











