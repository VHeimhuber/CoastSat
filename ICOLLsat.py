#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings

#load modules
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download2, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects  #SDS_download,
import fiona
import pandas as pd
from osgeo import gdal, gdalconst
# load additional machine learning modules #VH check if still needed
from sklearn.externals import joblib
import matplotlib.cm as cm
from skimage.segmentation import flood, flood_fill
import csv

# name of the site
sitename = 'DURRAS'

# region of interest (longitude, latitude in WGS84)

# can also be loaded from a .kml polygon
shp_polygon = os.path.join(os.getcwd(), 'Sites', sitename + '_full_bounding_box.shp')
with fiona.open(shp_polygon, "r") as shapefile:
    polygon = [feature["geometry"] for feature in shapefile] 
polygon = [[list(elem) for elem in polygon[0]['coordinates'][0]]] #to get coordinates in the right format

# date range
dates = ['1990-01-01', '2000-01-01']

# satellite missions
sat_list = ['L5'] #,'L7','L8','S2']

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), 'data')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data
        }

#%% 2. Retrieve images

# Load site polygons from shapefile database

# retrieve satellite images from GEE
metadata = SDS_download2.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download2.get_metadata(inputs) 

#%% 3. Batch shoreline detection
    
# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.05,        # threshold on maximum cloud cover
    'output_epsg': 3577,       # epsg code of spatial reference system desired for the output  
    'manual_seed': False,
    'shapefile_EPSG' : 4326,     #epsg of shapefiles that define sub regions for entrance detection 
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 4500,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 150,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 200,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'dark_sand': False,         # only switch to True if your site has dark sand (e.g. black sand beach)
}



# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
#SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
#settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
#settings['max_dist_ref'] = 20        


##### Original Code Above









"""
Extracts ICOLL entrance characteristics from satellite images.
"""

sitename = settings['inputs']['sitename']
filepath_data = settings['inputs']['filepath']
# initialise output structure
output = dict([]) 
   
# create a subfolder to store the .jpg images showing the detection
csv_out_path = os.path.join(os.getcwd(), 'data',sitename,  'results')
if not os.path.exists(csv_out_path):
        os.makedirs(csv_out_path)   
jpg_out_path =  os.path.join(filepath_data, sitename, 'jpg_files', 'classified')     
if not os.path.exists(jpg_out_path):      
    os.makedirs(jpg_out_path)
    
# close all open figures
plt.close('all')

print('Analyzing ICOLL entrance at: ' + sitename)

# loop through satellite list
#initialize summary dictionary
Summary={} 
#for satname in metadata.keys():                 #####!!!!!##### Intermediate
satname =  'L5'
# get images
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = metadata[satname]['filenames']
#filenames = filenames[46:52]                                                 #####!!!!!##### Intermediate

# load classifiers and
if satname in ['L5','L7','L8']:
    pixel_size = 15            
    if settings['dark_sand']:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat_dark.pkl'))
    else:
        clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_Landsat.pkl'))      
elif satname == 'S2':
    pixel_size = 10
    clf = joblib.load(os.path.join(os.getcwd(), 'classifiers', 'NN_4classes_S2.pkl'))

# convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels            
buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
  



##########################################
#load spatial configuration files from QGIS shapefiles
##########################################
fn1 = SDS_tools.get_filenames(filenames[1],filepath, satname)
# preprocess image (cloud mask + pansharpening/downsampling)
im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn1, satname,
                                                                             settings['cloud_mask_issue'])
    
shapes = SDS_tools.load_shapes_as_ndarrays(satname, sitename, settings['shapefile_EPSG'],
                                           georef, metadata)
x0, y0 = shapes['seedandreceivingpoint'][0,:]
x1, y1 = shapes['seedandreceivingpoint'][1,:]
Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bbx'])
##########################################
# end of load spatial configuration files 
##########################################  
    
   

##########################################
#loop through all images and store results in pd DataFrame
##########################################                
p=1
for i in range(len(filenames)): #####!!!!!##### Intermediate
    #i=2        #####!!!!!##### Intermediate
    print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
    # get image filename
    fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
    date = filenames[i][:19]
    # preprocess image (cloud mask + pansharpening/downsampling)
    im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname,
                                                                                 settings['cloud_mask_issue'])

    # get image spatial reference system (epsg code) from metadata dict
    image_epsg = metadata[satname]['epsg'][i]
    # calculate cloud cover
    cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                            (cloud_mask.shape[0]*cloud_mask.shape[1]))
    
    #skip image if cloud cover is above threshold                              #####!!!!!##### move up later on
    if cloud_cover > settings['cloud_thresh']:     #####!!!!!##### Intermediate
        if i==0:
            p=0
        continue

    # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
    im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, im_extra, cloud_mask,
                            min_beach_area_pixels, clf)
    
    # calculate a buffer around the reference shoreline (if any has been digitised)
    im_ref_buffer = SDS_shoreline.create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                            pixel_size, settings)
    
    # compute NDWI image (NIR)
    im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #NDWI
    #im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #ndwi
    #im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #Automatic Water Extraction Index

    #create an NDWI image where only entrance area is shown
    im_ndwi_masked = SDS_tools.maskimage_frompolygon(im_ndwi, shapes['entrance_bbx'])
    plt.imshow(im_ndwi_masked, cmap='seismic')
        
    #calculate water colour indices in line with Bugnot et al. 2018
    #0=blue, 1=green, 2=red, 4=NIR, 5=SWIR1
    #green: green/red, blue: blue/red) 
    im_GdR =  im_ms[:,:,1]/ im_ms[:,:,2]
    im_BdR =  im_ms[:,:,0]/ im_ms[:,:,2]
    #plt.imshow(im_RdR, cmap='seismic')
    
    #mask the water quality ration images and calculate the mean over the WQ area 
    im_GdR_masked = SDS_tools.maskimage_frompolygon(im_GdR, shapes['water_quality_area'])
    Green_by_red_mean = np.nanmean(im_GdR_masked)
    im_BdR_masked = SDS_tools.maskimage_frompolygon(im_BdR, shapes['water_quality_area'])
    Blue_by_red_mean = np.nanmean(im_BdR_masked)
    
    #Manually do the otsu threshold based classification for entire image area 0 = water, 1 = dryland
    im_class_ndwi, t_otsu_fullscene = SDS_tools.classify_binary_otsu(im_ndwi, cloud_mask)
    im_class_ndwi_masked, t_otsu_masked = SDS_tools.classify_binary_otsu(im_ndwi_masked, cloud_mask)

#    if p==0 or i==0:
#        im_class_ndwi_sum = np.copy(im_class_ndwi)
#        im_class_ndwi_sum.fill(0)
#        p=p+1
#    else:
#        im_class_ndwi_sum = im_class_ndwi_sum + im_class_ndwi
        
    #Use region growing on NDWI to test wether ICOLL entrance is open or closed.
    NDWI_open = 'closed'
    for tol in np.round(np.linspace(0.05, 1, num=20),2):
        if NDWI_open == 'closed':
            im_ndwi_grow = flood_fill(im_ndwi_masked, (int(y0), int(x0)), 8888, tolerance=tol)
            im_ndwi_grow[im_ndwi_grow != 8888] = 1
            im_ndwi_grow[im_ndwi_grow == 8888] = 0
            im_ndwi_grow_masked = SDS_tools.maskimage_frompolygon(im_ndwi_grow, shapes['entrance_receiving_area'])  
            if np.isin(im_ndwi_grow_masked, 0).any():  
                tol2 = tol
                NDWI_open = 'open'
                #print('ICOLL entrance was open for NDWI with a minimum region growing tolerance of ' + str(tol2))
    
    #Use region growing on NIR to test wether ICOLL entrance is open or closed.
    NIR_open = 'closed'
    for toln in np.linspace(0.005, 0.3, num=60):
        if NIR_open == 'closed':
            im_NIR_grow = flood_fill(im_ms[:,:,3], (int(y0), int(x0)), 8888, tolerance=toln)
            im_NIR_grow[im_NIR_grow != 8888] = 1
            im_NIR_grow[im_NIR_grow == 8888] = 0
            im_NIR_grow_masked = SDS_tools.maskimage_frompolygon(im_NIR_grow, shapes['entrance_receiving_area'])   
            if np.isin(im_NIR_grow_masked, 0).any(): 
                toln2 = toln
                NIR_open = 'open'
                #print('ICOLL entrance was open for NIR with a minimum region growing tolerance of ' + str(toln))
    
    #Use region growing on SWIR to test wether ICOLL entrance is open or closed.
    SWIR_open = 'closed'
    for tols in np.linspace(0.005, 0.3, num=60):
        if SWIR_open == 'closed':
            im_SWIR_grow = flood_fill(im_ms[:,:,4], (int(y0), int(x0)), 8888, tolerance=tols)
            im_SWIR_grow[im_SWIR_grow != 8888] = 1
            im_SWIR_grow[im_SWIR_grow == 8888] = 0
            im_SWIR_grow_masked = SDS_tools.maskimage_frompolygon(im_SWIR_grow, shapes['entrance_receiving_area'])  
            if np.isin(im_SWIR_grow_masked, 0).any(): 
                SWIR_open = 'open'
                tols2 = tols
                #print('ICOLL entrance was open for SWIR with a minimum region growing tolerance of ' + str(tols))
    
    #Use simple OTSU threshold NDWI based approach to test if ICOLL entrance is open or closed. 
    OTSU_ndwi_open = 'closed'
    im_class_fill = flood_fill(im_class_ndwi, (int(y0), int(x0)), 8888, tolerance=0.1)
    im_class_fill_masked = SDS_tools.maskimage_frompolygon(im_class_fill, shapes['entrance_receiving_area']) 
    if np.isin(im_class_fill_masked, 8888).any():
        OTSU_ndwi_open = 'open'
        #print('ICOLL entrance was open according to NDWI classification with OTSU = ' + str(round(t_otsu_masked, 3)))
       
    #Use simple OTSU threshold NDWI based approach to test if ICOLL entrance is open or closed. 
    OTSU_ndwi_ent_open = 'closed'
    im_class_fill1 = flood_fill(im_class_ndwi_masked, (int(y0), int(x0)), 8888, tolerance=0.1)
    im_class_fill_masked1 = SDS_tools.maskimage_frompolygon(im_class_fill1, shapes['entrance_receiving_area']) 
    if np.isin(im_class_fill_masked1, 8888).any():
        OTSU_ndwi_ent_open = 'open'
        print('ICOLL was open according to NDWI classification of entrance with OTSU = ' + str(round(t_otsu_masked, 3)))   
         
    #count pixels within estuary area for classified NDWI
    im_ndwi_estuary = SDS_tools.maskimage_frompolygon(im_class_ndwi, shapes['estuary_outline'])
    SWE = np.round(np.sum(np.isin(im_ndwi_estuary, 1)) *15*15/1000000, 4)#convert nr. of pixels (15x15m) to square km
    
    #store results in summary dictionary 
    Summary[date] =  satname, SWE, Green_by_red_mean, Blue_by_red_mean, OTSU_ndwi_open, OTSU_ndwi_ent_open, tol2, toln2,tols2 
    
    ##########################################
    #create plot and save to png
    ##########################################
    #bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])
    
    fig = plt.figure(figsize=(25,15))
    from matplotlib import colors
    cmap = colors.ListedColormap(['blue','red'])
    
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_NIR = SDS_preprocess.rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
    im_SWIR = SDS_preprocess.rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)
        
    #plot RGB
    ax=plt.subplot(2,6,1) 
    plt.title(sitename + ' RGB') 
    plt.imshow(im_RGB)
    
    #plot NDWI
    ax=plt.subplot(2,6,2) 
    plt.title('NDWI') 
    plt.imshow(im_ndwi, cmap='seismic')
    #plt.xlim(Xmin, Xmax)
    #plt.ylim(Ymax,Ymin) 
    
    #plot NIR
    ax=plt.subplot(2,6,3) 
    plt.title('NIR') 
    plt.imshow(im_NIR, cmap='seismic')
    
    #plot SWIR
    ax=plt.subplot(2,6,4) 
    plt.title('SWIR') 
    plt.imshow(im_SWIR, cmap='seismic')
    
    #plot RGB zoomed in
    ax=plt.subplot(2,6,5) 
    plt.title('Entrance RGB') 
    plt.imshow(im_RGB)
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend()
    
    #plot entrance onlz NDWI
    ax=plt.subplot(2,6,6) 
    plt.title('Entrance NDWI') 
    plt.imshow(im_ndwi_masked, cmap='seismic')  
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    
    #Kilian Neural netword
    ax=plt.subplot(2,6,7) 
    plt.title('Neural network clfd') 
    plt.imshow(im_classif) 
    
        #plot OTSU classified NDWI
    ax=plt.subplot(2,6,8) 
    plt.title('NDWI clfd ' + OTSU_ndwi_open) 
    plt.imshow(im_class_ndwi, cmap=cmap)
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend()
    
    
    #plot NIR region grower 
    ax=plt.subplot(2,6,9) 
    plt.title(' NIR grower tol= ' + str(np.round(toln2,3)) + ' ' + NIR_open) 
    plt.imshow(im_NIR_grow, cmap=cmap) 
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend()
    
    #plot SWIR region grower 
    ax=plt.subplot(2,6,10) 
    plt.title('SWIR grower tol= ' + str(np.round(tols2,3)) + ' ' + SWIR_open) 
    plt.imshow(im_SWIR_grow, cmap=cmap) 
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend() 
    
    #NDWI classified for entrance only
    ax=plt.subplot(2,6,11) 
    plt.title('NDWI clfd entrance area '+ OTSU_ndwi_ent_open) 
    plt.imshow(im_class_ndwi_masked, cmap=cmap) 
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend()
    
    #plot NDWI region grower 
    ax=plt.subplot(2,6,12) 
    plt.title('NDWI grower tol= ' + str(tol2) + ' ' + NDWI_open) 
    plt.imshow(im_ndwi_grow, cmap=cmap) 
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymax,Ymin) 
    plt.plot(x0, y0, 'ro', color='yellow', marker="X", label='Reg. growing seed')
    plt.plot(x1, y1, 'ro', color='yellow', marker="D", label='Reg. growing receiver')
    plt.legend()
    
    fig.tight_layout()
    plt.rcParams['savefig.jpeg_quality'] = 100
    fig.savefig(os.path.join(jpg_out_path, filenames[i][:19] + '_' + satname + '_cfd2.jpg') , dpi=150)
    plt.close()
    ##########################################

#plt.imshow(im_ndwi_masked)
pdf1=pd.DataFrame(Summary).transpose()
pdf1.columns = [satname, 'SWE', 'Green_by_red','Blue_by_red', 'OTSU_ndwi_ful',  'OTSU_ndwi_ent', 'NDWI_RG_tol', 'NIR_RG_tol', 'SWIR_RG_tol' ]
pdf1.to_csv(os.path.join(csv_out_path, sitename + '_entrance_stats.csv'))
       

















# visualise the mapped shorelines, there are two options:
# if settings['check_detection'] = True, shows the detection to the user for accept/reject
# if settings['save_figure'] = True, saves a figure for each mapped shoreline
if settings['check_detection'] or settings['save_figure']:
    date = filenames[i][:19]
    skip_image = SDS_shoreline.show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                image_epsg, georef, settings, date, satname)
    # if the user decides to skip the image, continue and do not save the mapped shoreline
    if skip_image:
        continue
    


            
            
            
            









        # there are two options to extract to map the contours:
        # if there are pixels in the 'sand' class --> use find_wl_contours2 (enhanced)
        # otherwise use find_wl_contours2 (traditional)
        try: # use try/except structure for long runs
            if sum(sum(im_labels[:,:,0])) == 0 :
                # compute MNDWI image (SWIR-G)
                im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                # find water contours on MNDWI grayscale image
                contours_mwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
            else:
                # use classification to refine threshold and extract the sand/water interface
                contours_wi, contours_mwi = find_wl_contours2(im_ms, im_labels,
                                            cloud_mask, buffer_size_pixels, im_ref_buffer)
        except:
            print('Could not map shoreline for this image: ' + filenames[i])
            continue

        # process water contours into shorelines
        shoreline = process_shoreline(contours_mwi, georef, image_epsg, settings)

        # visualise the mapped shorelines, there are two options:
        # if settings['check_detection'] = True, shows the detection to the user for accept/reject
        # if settings['save_figure'] = True, saves a figure for each mapped shoreline
        if settings['check_detection'] or settings['save_figure']:
            date = filenames[i][:19]
            skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                        image_epsg, georef, settings, date, satname)
            # if the user decides to skip the image, continue and do not save the mapped shoreline
            if skip_image:
                continue

        # append to output variables
        output_timestamp.append(metadata[satname]['dates'][i])
        output_shoreline.append(shoreline)
        output_filename.append(filenames[i])
        output_cloudcover.append(cloud_cover)
        output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
        output_idxkeep.append(i)

    # create dictionnary of output
    output[satname] = {
            'dates': output_timestamp,
            'shorelines': output_shoreline,
            'filename': output_filename,
            'cloud_cover': output_cloudcover,
            'geoaccuracy': output_geoaccuracy,
            'idx': output_idxkeep
            }
    print('')

# Close figure window if still open
if plt.get_fignums():
    plt.close()

# change the format to have one list sorted by date with all the shorelines (easier to use)
output = SDS_tools.merge_output(output)

# save outputput structure as output.pkl
filepath = os.path.join(filepath_data, sitename)
with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
    pickle.dump(output, f)

# save output into a gdb.GeoDataFrame
gdf = SDS_tools.output_to_gdf(output)
# set projection
gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])}
# save as geojson    
gdf.to_file(os.path.join(filepath, sitename + '_output.geojson'), driver='GeoJSON', encoding='utf-8')

return output















##### Original Code Below 

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# plot the mapped shorelines
fig = plt.figure()
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])

#%% 4. Shoreline analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output' + '.pkl'), 'rb') as f:
    output = pickle.load(f) 

# now we have to define cross-shore transects over which to quantify the shoreline changes
# each transect is defined by two points, its origin and a second point that defines its orientation

# there are 3 options to create the transects:
# - option 1: draw the shore-normal transects along the beach
# - option 2: load the transect coordinates from a .kml file
# - option 3: create the transects manually by providing the coordinates

# option 1: draw origin of transect first and then a second point to define the orientation
transects = SDS_transects.draw_transects(output, settings)
    
# option 2: load the transects from a .geojson file
#geojson_file = os.path.join(os.getcwd(), 'examples', 'NARRA_transects.geojson')
#transects = SDS_tools.transects_from_geojson(geojson_file)

# option 3: create the transects by manually providing the coordinates of two points 
#transects = dict([])
#transects['Transect 1'] = np.array([[342836, 6269215], [343315, 6269071]])
#transects['Transect 2'] = np.array([[342482, 6268466], [342958, 6268310]])
#transects['Transect 3'] = np.array([[342185, 6267650], [342685, 6267641]])
   
# intersect the transects with the 2D shorelines to obtain time-series of cross-shore distance
settings['along_dist'] = 25
cross_distance = SDS_transects.compute_intersection(output, transects, settings) 

# plot the time-series
from matplotlib import gridspec
fig = plt.figure()
gs = gridspec.GridSpec(len(cross_distance),1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-50,50])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-^', markersize=6)
    ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95,'Transect ' + key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
            va='top', transform=ax.transAxes, fontsize=14)
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])