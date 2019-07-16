#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# Kilian Vos WRL 2018

#%% 1. Initial settings

# load modules
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
import fiona

# name of the site
sitename = 'CATHIE'

# region of interest (longitude, latitude in WGS84)

# can also be loaded from a .kml polygon
shp_polygon = os.path.join(os.getcwd(), 'Sites', sitename + '_full_bounding_box.shp')
with fiona.open(shp_polygon, "r") as shapefile:
    polygon = [feature["geometry"] for feature in shapefile] 
polygon = [[list(elem) for elem in polygon[0]['coordinates'][0]]] #to get coordinates in the right format
    
# date range
dates = ['1995-01-01', '1995-06-02']

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
metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs) 


#%% 3. Batch shoreline detection
    
# settings for the shoreline extraction
settings = { 
    # general parameters:
    'cloud_thresh': 0.1,        # threshold on maximum cloud cover
    'output_epsg': 3577,       # epsg code of spatial reference system desired for the output   
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
SDS_preprocess.save_jpg(metadata, settings)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings['max_dist_ref'] = 20        


##### Original Code Above









"""
Extracts ICOLL entrance characteristics from satellite images.

"""
# load machine learning modules
from sklearn.externals import joblib
import skimage.filters as filters
import matplotlib.cm as cm
from skimage.segmentation import flood, flood_fill
import rasterio.mask



sitename = settings['inputs']['sitename']
filepath_data = settings['inputs']['filepath']
# initialise output structure
output = dict([]) 
   
# create a subfolder to store the .jpg images showing the detection
filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection2')
if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
# close all open figures
plt.close('all')

print('Mapping shorelines:')

# loop through satellite list
#for satname in metadata.keys():                 #####!!!!!##### Intermediate
satname =  'L5'
# get images
filepath = SDS_tools.get_filepath(settings['inputs'],satname)
filenames = metadata[satname]['filenames']
filenames = filenames[1:20]                                                 #####!!!!!##### Intermediate

# initialise the output variables
output_timestamp = []  # datetime at which the image was acquired (UTC time)
output_shoreline = []  # vector of shoreline points
output_filename = []   # filename of the images from which the shorelines where derived
output_cloudcover = [] # cloud cover of the images
output_geoaccuracy = []# georeferencing accuracy of the images
output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)

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
#Create reference entrance berm and seed + end point by clicking on image
##########################################
#manual selection of reference points
# get image filename
fn = SDS_tools.get_filenames(filenames[0],filepath, satname)
# preprocess image (cloud mask + pansharpening/downsampling)
im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
coords=[]
ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(im_ndwi, cmap='viridis')
def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        coords.append((event.xdata, event.ydata))
    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
################################################ run until here first and input the cross section start and end point by clicking
x0, y0 = coords[0] # These are in _pixel_ coordinates and I just defined them by looking at the image
x1, y1 = coords[1]
################################################


#automated usage of reference points from predefined shapefiles
#load spatial configuration files 
#seed point for region growing
shp_seed = os.path.join(os.getcwd(), 'Sites', sitename + '_ocean_seed.shp')    
with fiona.open(shp_seed, "r") as shapefile:
    features = [feature["geometry"] for feature in shapefile]
[x0, y0] = features[0]['coordinates'][0],features[0]['coordinates'][1]
[x1, y1] = features[1]['coordinates'][0],features[1]['coordinates'][1]
seedpoint_array = np.array([[x0, y0], [x1, y1]])

#convert input spatial layers to image coordinates
#if lat lon use 4326 (WGS 84)
image_epsg = metadata[satname]['epsg'][1]
seedpoint_array_conv = SDS_tools.convert_epsg(seedpoint_array, 4326, image_epsg)
[x0, y0] = SDS_tools.convert_world2pix(seedpoint_array_conv[:,:-1], georef)[0]
[x1, y1] = SDS_tools.convert_world2pix(seedpoint_array_conv[:,:-1], georef)[1]
 
#ICOLL entrance area bounding box for limiting spectral variability of scene
shp_entrance_bbx = os.path.join(os.getcwd(), 'Sites', sitename + '_entrance_bounding_box.shp')    
with fiona.open(shp_entrance_bbx, "r") as shapefile:
    entrance_bbx = [feature["geometry"] for feature in shapefile]  
entrance_bbx = [[list(elem) for elem in entrance_bbx[0]['coordinates'][0]]]
entrance_bbx_conv = SDS_tools.convert_epsg(entrance_bbx, 4326, image_epsg)
entrance_bbx_pix = SDS_tools.convert_world2pix(entrance_bbx_conv[0][:,:-1], georef)

#ICOLL entrance area that's clearly within the lagoon - it's used as a receiving area for the region grower
shp_entrance_receiver = os.path.join(os.getcwd(), 'Sites', sitename + '_entrance_area.shp')    
with fiona.open(shp_entrance_receiver, "r") as shapefile:
    entrance_receiver = [feature["geometry"] for feature in shapefile] 
entrance_rec = [[list(elem) for elem in entrance_receiver[0]['coordinates'][0]]]
entrance_rec_conv = SDS_tools.convert_epsg(entrance_rec, 4326, image_epsg)
entrance_rec_pix = SDS_tools.convert_world2pix(entrance_rec_conv [0][:,:-1], georef)  

#mask the image based on the input entrance bounding box
#out_image, out_transform = rasterio.mask.mask(im_mndwi, features, crop=True, nodata=np.nan)
#out_meta = src.meta.copy()
    
    
                 
# loop through the images
#for i in range(len(filenames)): #####!!!!!##### Intermediate
i=2        #####!!!!!##### Intermediate
print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

# get image filename
fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
# preprocess image (cloud mask + pansharpening/downsampling)
im_ms, georef, cloud_mask, im_extra, imQA = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
# get image spatial reference system (epsg code) from metadata dict
image_epsg = metadata[satname]['epsg'][i]
# calculate cloud cover
cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                        (cloud_mask.shape[0]*cloud_mask.shape[1]))

cloud_cover > settings['cloud_thresh']
# skip image if cloud cover is above threshold
#if cloud_cover > settings['cloud_thresh']:     #####!!!!!##### Intermediate
#   continue

# classify image in 4 classes (sand, whitewater, water, other) with NN classifier
im_classif, im_labels = SDS_shoreline.classify_image_NN(im_ms, im_extra, cloud_mask,
                        min_beach_area_pixels, clf)

# calculate a buffer around the reference shoreline (if any has been digitised)
im_ref_buffer = SDS_shoreline.create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                        pixel_size, settings)

# compute MNDWI image (SWIR-G)
im_mndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #NDWI
#im_mndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #mNDWI
#im_mndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  #Automatic Water Extraction Index

#Manually do the otsu threshold based classificaiton
# reshape image to vector
vec_ndwi = im_mndwi.reshape(im_mndwi.shape[0] * im_mndwi.shape[1])
vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
vec = vec_ndwi[~vec_mask]
# apply otsu's threshold
vec = vec[~np.isnan(vec)]
t_otsu = filters.threshold_otsu(vec)

# compute classified image
im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
im_class = np.copy(im_RGB[:,:,1])
im_class[im_mndwi < t_otsu] = 0
im_class[im_mndwi >= t_otsu] = 1


#Use region growing on NDWI to test wether ICOLL entrance is open or closed.
placeholder=0
for tol in [0.2,0.3,0.4,0.5, 0.6, 0.9]:
    if placeholder == 0:
        im_mndwi_grow = flood_fill(im_mndwi, (int(y0), int(x0)), 100, tolerance=tol)
        if im_mndwi_grow[int(y1), int(x1)] == 100:
            placeholder=1
            print('ICOLL entrance was open for NDWI with a minimum region growing tolerance of ' + str(tol))

#Use region growing on NIR to test wether ICOLL entrance is open or closed.
placeholder=0
for toln in [0.05, 0.08, 0.1,0.13,0.18, 0.2]:
    if placeholder == 0:
        im_NIR_grow = flood_fill(im_ms[:,:,3], (int(y0), int(x0)), 100, tolerance=toln)
        if im_NIR_grow[int(y1), int(x1)] == 100:
            placeholder=1
            toln2 = toln
            print('ICOLL entrance was open for NIR with a minimum region growing tolerance of ' + str(toln))

#Use region growing on SWIR to test wether ICOLL entrance is open or closed.
placeholder=0
for tols in [0.05, 0.08, 0.1,0.13,0.18, 0.2]:
    if placeholder == 0:
        im_SWIR_grow = flood_fill(im_ms[:,:,4], (int(y0), int(x0)), 100, tolerance=tols)
        if im_SWIR_grow[int(y1), int(x1)] == 100:
            placeholder=1
            tols2 = tols
            print('ICOLL entrance was open for SWIR with a minimum region growing tolerance of ' + str(tols))

#Use simple OTSU threshold NDWI based approach to test if ICOL entrance is open or closed. 
im_class_fill = flood_fill(im_class, (int(y0), int(x0)), 100, tolerance=tol)
if im_class_fill[int(y1), int(x1)] == 100:
    placeholder=1
    print('ICOLL entrance was open according to NDWI classification with OTSU = ' + str(t_otsu))
            

# create a plot of the results
fig = plt.figure(figsize=(25,10))
from matplotlib import colors
cmap = colors.ListedColormap(['red','blue'])

#plot RGB
ax=plt.subplot(2,4,1) 
plt.title(sitename + ' RGB') 
plt.imshow(im_RGB)

#plot NIR
ax=plt.subplot(2,4,2) 
plt.title(sitename + ' NIR') 
plt.imshow(im_ms[:,:,3], cmap='seismic')

#plot SWIR
ax=plt.subplot(2,4,3) 
plt.title(sitename + ' SWIR') 
plt.imshow(im_ms[:,:,4], cmap='seismic')

#plot NDWI
ax=plt.subplot(2,4,4) 
plt.title(sitename + ' NDWI') 
plt.imshow(im_mndwi, cmap='seismic')

#plot OTSU classified NDWI
ax=plt.subplot(2,4,5) 
plt.title(sitename + ' NDWI classfd') 
plt.imshow(im_class)   

#plot NIR region grower 
ax=plt.subplot(2,4,6) 
plt.title(sitename + ' NIR Grown Seed toler= ' + str(toln2)) 
plt.imshow(im_NIR_grow, cmap=cmap) 

#plot SWIR region grower 
ax=plt.subplot(2,4,7) 
plt.title(sitename + ' SWIR Grown Seed toler= ' + str(tols2)) 
plt.imshow(im_SWIR_grow, cmap=cmap) 

#plot NDWI region grower 
ax=plt.subplot(2,4,8) 
plt.title(sitename + ' NDWI Grown Seed toler= ' + str(tol)) 
plt.imshow(im_mndwi_grow, cmap=cmap) 






























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