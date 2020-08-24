"""This module contains all the functions needed for extracting satellite-derived shorelines (SDS)

   Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
#import pdb

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

# machine learning modules
from sklearn.externals import joblib
from shapely.geometry import LineString

# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import gridspec
import matplotlib
import pickle
import geopandas as gpd
import pandas as pd
import random

#ICOLLsat requirements
from pylab import ginput
from shapely import geometry
import scipy
import skimage.transform as transform
from skimage.graph import route_through_array
import seaborn

# own modules
from coastsat import SDS_tools, SDS_preprocess, SDS_floodfill, SDS_slope


np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# IMAGE CLASSIFICATION FUNCTIONS
###################################################################################################

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates a range of features on the image that are used for the supervised classification.
    The features include spectral normalized-difference indices and standard deviation of the image.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_bool: np.array
            2D array of boolean indicating where on the image to calculate the features

    Returns:    -----------
        features: np.array
            matrix containing each feature (columns) calculated for all
            the pixels (rows) indicated in im_bool
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = SDS_tools.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  SDS_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features

def classify_image_NN(im_ms, im_extra, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network previously trained.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            Pansharpened RGB + downsampled NIR and SWIR
        im_extra:
            only used for Landsat 7 and 8 where im_extra is the panchromatic band
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        min_beach_area: int
            minimum number of pixels that have to be connected to belong to the SAND class
        clf: classifier

    Returns:    -----------
        im_classif: np.array
            2D image containing labels
        im_labels: np.array of booleans
            3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)

    return im_classif, im_labels

###################################################################################################
# CONTOUR MAPPING FUNCTIONS
###################################################################################################
    
def find_wl_contours1(im_ndwi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shorelien detection.
    Finds the water line by thresholding the Normalized Difference Water Index and applying
    the Marching Squares Algorithm to contour the iso-value corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
        im_ndwi: np.ndarray
            Image (2D) with the NDWI (water index)
        cloud_mask: np.ndarray
            2D cloud mask with True where cloud pixels are
        im_ref_buffer: np.array
            Binary image containing a buffer around the reference shoreline

    Returns:    -----------
        contours_wl: list of np.arrays
            contains the (row,column) coordinates of the contour lines

    """

    # reshape image to vector
    vec_ndwi = im_ndwi.reshape(im_ndwi.shape[0] * im_ndwi.shape[1])
    vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec = vec_ndwi[~vec_mask]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndwi_buffer = np.copy(im_ndwi)
    im_ndwi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndwi_buffer, t_otsu)

    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours = contours_nonans

    return contours

def find_wl_contours2(im_ms, im_labels, cloud_mask, buffer_size, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification component to
    refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        buffer_size: int
            size of the buffer around the sandy beach over which the pixels are considered in the
            thresholding algorithm.
        im_ref_buffer: np.array
            Binary image containing a buffer around the reference shoreline

    Returns:    -----------
        contours_wi: list of np.arrays
            contains the (row,column) coordinates of the contour lines extracted from the
            NDWI (Normalized Difference Water Index) image
        contours_mwi: list of np.arrays
            contains the (row,column) coordinates of the contour lines extracted from the
            MNDWI (Modified Normalized Difference Water Index) image

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)

    # create a buffer around the sandy beach
    se = morphology.disk(buffer_size)
    im_buffer = morphology.binary_dilation(im_labels[:,:,0], se)
    vec_buffer = im_buffer.reshape(nrows*ncols)

    # select water/sand/swash pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
    int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
            int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_water,int_sand, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])

    # find contour with MS algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)

    # remove contour points that are NaNs (around clouds)
    contours = contours_wi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_wi = contours_nonans
    # repeat for MNDWI contours
    contours = contours_mwi
    contours_nonans = []
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    contours_mwi = contours_nonans

    return contours_wi, contours_mwi

###################################################################################################
# SHORELINE PROCESSING FUNCTIONS
###################################################################################################
    
def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is given by
    settings['max_dist_ref'].

    KV WRL 2018

    Arguments:
    -----------
        im_shape: np.array
            size of the image (rows,columns)
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        pixel_size: int
            size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
        settings: dict
            contains the following fields:
        output_epsg: int
            output spatial reference system
        reference_shoreline: np.array
            coordinates of the reference shoreline
        max_dist_ref: int 
            maximum distance from the reference shoreline in metres

    Returns:    -----------
        im_buffer: np.array
            binary image, True where the buffer is, False otherwise

    """    
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)

    if 'reference_shoreline' in settings.keys():
        
        # convert reference shoreline to pixel coordinates
        ref_sl = settings['reference_shoreline']
        ref_sl_conv = SDS_tools.convert_epsg(ref_sl, settings['output_epsg'],image_epsg)[:,:-1]
        ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
        ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
        
        # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
        im_binary = np.zeros(im_shape)
        for j in range(len(ref_sl_pix_rounded)):
            im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
        im_binary = im_binary.astype(bool)
        
        # dilate the binary image to create a buffer around the reference shoreline
        max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
        se = morphology.disk(max_dist_ref_pixels)
        im_buffer = morphology.binary_dilation(im_binary, se)
        
    return im_buffer

def process_shoreline(contours, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. This function also removes
    the contours that are too small to be a shoreline (based on the parameter
    settings['min_length_sl'])

    KV WRL 2018

    Arguments:
    -----------
        contours: np.array or list of np.array
            image contours as detected by the function find_contours
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        settings: dict
            contains the following fields:
        output_epsg: int
            output spatial reference system
        min_length_sl: float
            minimum length of shoreline perimeter to be kept (in meters)

    Returns:    
    -----------
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline

    """

    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    shoreline = contours_array

    return shoreline

def show_detection(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control. The user can select "keep"
    if the shoreline detection is correct or "skip" if it is incorrect.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        settings: dict
            contains the following fields:
        date: string
            date at which the image was taken
        satname: string
            indicates the satname (L5,L7,L8 or S2)

    Returns:    
    -----------
        skip_image: boolean
            True if the user wants to skip the image, False otherwise.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
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

    # compute MNDWI grayscale image
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:,[0,1]], georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])

    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([12.53, 9.3])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images 
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[2,0])
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title(date, fontweight='bold', fontsize=16)

    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    ax3.set_title(satname, fontweight='bold', fontsize=16)

# additional options
#    ax1.set_anchor('W')
#    ax2.set_anchor('W')
#    cb = plt.colorbar()
#    cb.ax.tick_params(labelsize=10)
#    cb.set_label('MNDWI values')
#    ax3.set_anchor('W')

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection']:
            
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=200)

    # Don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image


def set_openvsclosed(im_ms, inputs,jpg_out_path, cloud_mask, im_labels, georef,
                   settings, date, satname, Xmin, Xmax, Ymin, Ymax, image_nr, filenames_itm):
    """
    Shows the detected shoreline to the user for visual quality control. The user can select "keep"
    if the shoreline detection is correct or "skip" if it is incorrect.

    KV WRL 2018

    Arguments:
    -----------
        im_ms: np.array
            RGB + downsampled NIR and SWIR
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_labels: np.array
            3D image containing a boolean image for each class in the order (sand, swash, water)
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        settings: dict
            contains the following fields:
        date: string
            date at which the image was taken
        satname: string
            indicates the satname (L5,L7,L8 or S2)

    Returns:    
    -----------
        skip_image: boolean
            True if the user wants to skip the image, False otherwise.

    """
    keep_checking_inloop = 'True'
    im_mNDWI = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    im_NDWI = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    if plt.get_fignums():
        # get open figure if it exists
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([12.53, 9.3])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images 
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[2,0])
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)  
    
    # compute classified image
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

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    #ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_xlim(Xmin-30, Xmax+30)
    ax1.set_ylim( Ymax+30, Ymin-30)  
    ax1.set_title(inputs['sitename'], fontweight='bold', fontsize=16)

    # create image 2 (classification)
    #ax2.imshow(im_class)
    ax2.imshow(im_NDWI, cmap='bwr', vmin=-1, vmax=1)
    #ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
#    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
#    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
#    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
#    #black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
#    ax2.legend(handles=[orange_patch,white_patch,blue_patch],
#               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_xlim(Xmin-30, Xmax+30)
    ax2.set_ylim( Ymax+30, Ymin-30)  
    ax2.set_title( 'NDWI ' + date, fontweight='bold', fontsize=16)

    # create image 3 (MNDWI)
    ax3.imshow(im_mNDWI, cmap='bwr', vmin=-1, vmax=1)
    #ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    #plt.colorbar()
    ax3.set_xlim(Xmin-30, Xmax+30)
    ax3.set_ylim( Ymax+30, Ymin-30)  
    ax3.set_title('mNDWI ' +  satname + ' ' + str(int(((image_nr+1)/len(filenames_itm))*100)) + '%', fontweight='bold', fontsize=16)

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    vis_open_vs_closed = 'NA'
    if settings['check_detection']:
            
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_open = plt.text(1.1, 0.95, 'open ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_closed = plt.text(-0.1, 0.95, '⇦ closed', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(0.5, 0.95, '⇧ skip', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w')) 
            btn_esc = plt.text(0.5, 0.07, '⇓ unclear', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, -0.03, 'esc', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_open.remove()
            btn_closed.remove()
            btn_skip.remove()
            btn_esc.remove()
            
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                vis_open_vs_closed = 'open'
                break
            elif key_event.get('pressed') == 'left':
                skip_image = False
                vis_open_vs_closed = 'closed'
                break
            elif key_event.get('pressed') == 'down':
                vis_open_vs_closed = 'unclear'
                skip_image = False
                break
            elif key_event.get('pressed') == 'up':
                vis_open_vs_closed = 'poorquality'
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                skip_image = True
                vis_open_vs_closed = 'exit on image'
                keep_checking_inloop = 'False'
                break
                #raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(jpg_out_path, date + '_' + satname + '.jpg'), dpi=200)

    # Don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()
    
    return vis_open_vs_closed, skip_image, keep_checking_inloop
 



def create_training_data(metadata, settings):
    """
    Function that lets user visually inspect satellite images and decide if 
    entrance is open or closed.
    
    This can be done for the entire dataset or to a limited number of images, which will then be used to train the machine learning classifier for open vs. closed

    VH WRL 2020

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded

        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        check_detection: boolean
            True to show each invidual satellite image and let the user decide if the entrance was open or closed
    Returns:
    -----------
        output: dict
            contains the training data set for all inspected images

    """      
        
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    
    print('Generating traning data for entrance state at: ' + sitename)
    print('Manually inspect each image to create training data. Press esc once a satisfactory number of images was inspected') 
    
    # create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
    csv_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'])
    if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path) 
            
    #load shapefile that conta0ins specific shapes for each ICOLL site as per readme file
    Allsites = gpd.read_file(os.path.join(os.getcwd(), 'Sites', 'All_sites9.shp')) #.iloc[:,-4:]
    Site_shps = Allsites.loc[(Allsites.Sitename==sitename)]
    layers = Site_shps['layer'].values

    # initialise output data structure
    Training={}
       
    # create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
    csv_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'])
    if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path)   
    jpg_out_path =  os.path.join(filepath_data, sitename, 'jpg_files', 'classified_' + settings['inputs']['analysis_vrs'])     
    if not os.path.exists(jpg_out_path):      
        os.makedirs(jpg_out_path)
    
    # close all open figures
    plt.close('all')
    
    
    # loop through the user selecte satellites 
    for satname in settings['inputs']['sat_list']:
      
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']      

        #randomize the time step to create a more independent training data set
        epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))
        if settings['shuffle_training_imgs']==True:
            filenames = random.sample(filenames, len(filenames))

        
        # load classifiers and
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
        
        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)                 
        
        ##########################################
        #loop through all images and store results in pd DataFrame
        ##########################################   
        plt.close()
        keep_checking = 'True'
        for i in range(len(filenames)):
            if keep_checking == 'True':
                print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
                
                # get image filename
                fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
                date = filenames[i][:19]
        
                # preprocess image (cloud mask + pansharpening/downsampling)
                im_ms, georef, cloud_mask, im_extra, imQA, im_nodata  = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            
                # calculate cloud cover
                cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                        (cloud_mask.shape[0]*cloud_mask.shape[1]))
                
                #skip image if cloud cover is above threshold
                if cloud_cover > settings['cloud_thresh']:     #####!!!!!##### Intermediate
                    continue
                
                #load boundary shapefiles for each scene and reproject according to satellite image epsg  
                shapes = SDS_tools.load_shapes_as_ndarrays_2(layers, Site_shps, satname, sitename, settings['shapefile_EPSG'],
                                                   georef, metadata, epsg_dict[filenames[i]])
                #get the min and max corner (in pixel coordinates) of the entrance area that will be used for plotting the data for visual inspection
                Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box'])      
                
                # classify image in 4 classes (sand, vegetation, water, other) with NN classifier
                im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask,
                                        min_beach_area_pixels, clf)
                    
                #Manually check entrance state to generate training data
                if settings['check_detection'] or settings['save_figure']:
                    vis_open_vs_closed, skip_image, keep_checking = set_openvsclosed(im_ms, settings['inputs'],jpg_out_path, cloud_mask, im_labels, georef, settings, date,
                                                                                                   satname, Xmin, Xmax, Ymin, Ymax, i, filenames)     
                #add results to intermediate list
                Training[date] =  satname, vis_open_vs_closed, skip_image
    
    Training_df= pd.DataFrame(Training).transpose()
    Training_df.columns = ['satname', 'Entrance_state','skip image']
    if len(Training_df.index) > 5:
        Training_df.to_csv(os.path.join(csv_out_path, sitename +'_visual_training_data.csv'))
    
    return Training_df  



def user_defined_entrance_paths(metadata, settings, Exp_code, Identifier):
    """
    Function that lets user visually input the connecting path from the ocean seed point
    to the entrance receiver point, regardless weather the entrance is open or not
    
    This can be done for the entire dataset or to a limited number of images
    
    It is mainly for research and algorithm refinement purposes

    VH WRL 2020

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded

        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
    Returns:
    -----------
        output: pandas dataframe withthe NDWI values along the transect for each image

    """      
        
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    
    print('Generating entrance NDWI paths at ' + sitename)
    print('Manually digitize the entrance paths for each image. Press esc once a satisfactory number of images were digitized for each satellite') 
       
    # create a subfolder to store the .jpg images showing the detection + csv file of the generated training dataset
    csv_out_path = os.path.join(filepath_data, sitename,  'results_' + settings['inputs']['analysis_vrs'], Exp_code)
    if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path)  
    image_out_path = os.path.join(csv_out_path, 'user_digitized')
    if not os.path.exists(image_out_path):
            os.makedirs(image_out_path) 
    
    # close all open figures
    plt.close('all')   
    
    # initialise output data structure
    iterator=0 #additional incremental integer to control the addition of user input lines to the gpd dataframe
    gdf_all = gpd.GeoDataFrame()
    XS={} 
    
    # loop through the user selected satellites 
    for satname in settings['inputs']['sat_list']:
        keep_inputting = 'True'
      
        #dates = metadata[satname]['dates']
        print('Digitizing ICOLL entrances at: ' + sitename + ' for ' + satname)
        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        
        #randomize the time step to create a more independent training data set
        epsg_dict = dict(zip(filenames, metadata[satname]['epsg']))
        dates_dict = dict(zip(filenames, metadata[satname]['dates']))
        if settings['shuffle_entrance_paths_imgs']==True:
            filenames = random.sample(filenames, len(filenames))
        
        #initialize figure
        fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    
        nr_of_inputs = 1
        for i in range(len(filenames)):
            if keep_inputting == 'True':
                # create figure
                if plt.get_fignums():
                # get open figure if it exists
                    fig = plt.gcf()
                    ax = fig.axes[0]    
                    #ax2 = fig.axes[1]
                    #ax3 = fig.axes[2]
                
                # read image
                fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
                im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            
                # calculate cloud cover
                cloud_cover = np.divide(sum(sum(cloud_mask.astype(int))),
                                        (cloud_mask.shape[0]*cloud_mask.shape[1]))
            
                # skip image if cloud cover is above threshold
                if cloud_cover > settings['cloud_thresh']:
                    continue
            
                # rescale image intensity for display purposes
                im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            
                #load all shape and area polygons in pixel coordinates to set up the configuration for the spatial analysis of entrances
                shapes = SDS_tools.load_shapes_as_ndarrays_2(settings['inputs']['location_shps']['layer'].values, settings['inputs']['location_shps'], satname, sitename, settings['shapefile_EPSG'],
                                                   georef, metadata, epsg_dict[filenames[i]] )   
                
                #get the min and max corner (in pixel coordinates) of the entrance area that will be used for plotting the data for visual inspection
                Xmin,Xmax,Ymin,Ymax = SDS_tools.get_bounding_box_minmax(shapes['entrance_bounding_box']) 
                
                x0, y0 = shapes['ocean_seed'][1,:]
                x1, y1 = shapes['entrance_seed'][1,:]
                x2, y2 = shapes['berm_point_A'][1,:]
                x3, y3 = shapes['berm_point_B'][1,:]
                
                # plot the image RGB on a figure
                ax.axis('off')
                ax.imshow(im_RGB)
                plt.plot(x0, y0, 'ro', color='yellow', marker="X")
                plt.plot(x1, y1, 'ro', color='yellow', marker="X")   
                plt.plot(x2, y2, 'ro', color='lime', marker="X")
                plt.plot(x3, y3, 'ro', color='lime', marker="X")  
                plt.text(x0+1, y0+1,'A',horizontalalignment='left', color='yellow' , fontsize=16)
                plt.text(x1+1, y1+1,'B',horizontalalignment='left', color='yellow', fontsize=16)
                plt.text(x2+1, y2+1,'A',horizontalalignment='left', color='lime', fontsize=16)
                plt.text(x3+1, y3+1,'B',horizontalalignment='left', color='lime', fontsize=16)
  
                ax.set_xlim(Xmin-30, Xmax+30)
                ax.set_ylim(Ymax+30, Ymin-30) 
                
                # decide if the image if good enough for digitizing the shoreline
                ax.set_title('Press <right arrow> if image is clear enough to digitize the entrance connection .\n ' +
                          'press <left arrow> to get another image.\n' +
                          filenames[i] + ' do ' + Identifier , fontsize=14)
                # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
                # this variable needs to be immuatable so we can access it after the keypress event
                skip_image = False
                key_event = {}
                def press(event):
                    # store what key was pressed in the dictionary
                    key_event['pressed'] = event.key
                # let the user press a key, right arrow to keep the image, left arrow to skip it
                # to break the loop the user can press 'escape'
                while True:
                    btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                        transform=ax.transAxes,
                                        bbox=dict(boxstyle="square", ec='k',fc='w'))
                    btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                        transform=ax.transAxes,
                                        bbox=dict(boxstyle="square", ec='k',fc='w'))
                    btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                        transform=ax.transAxes,
                                        bbox=dict(boxstyle="square", ec='k',fc='w'))
                    plt.draw()
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    # after button is pressed, remove the buttons
                    btn_skip.remove()
                    btn_keep.remove()
                    btn_esc.remove()
                    # keep/skip image according to the pressed key, 'escape' to break the loop
                    if key_event.get('pressed') == 'right':
                        skip_image = False
                        break
                    elif key_event.get('pressed') == 'left':
                        skip_image = True
                        break
                    elif key_event.get('pressed') == 'escape':
                        plt.close()
                        keep_inputting =  'False'
                        skip_image = True
                        #raise StopIteration('User cancelled checking shoreline detection')
                        break
                    else:
                        plt.waitforbuttonpress()
                 
                if keep_inputting == 'False':
                    plt.close('all') 
                if skip_image:
                    ax.clear() 
                    continue
                else:  
                    # create a new continue button
                    add_button = plt.text(0, 0.9, 'continue', size=16, ha="left", va="top",
                                           transform=plt.gca().transAxes,
                                           bbox=dict(boxstyle="square", ec='k',fc='w'))
                    
                    # add multiple reference shorelines (until user clicks on <end> button)
                    pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    geoms = []
                    #while 1:
                    add_button.set_visible(False)
                    #end_button.set_visible(False)
                    # update title (instructions)
                    ax.set_title('Click points along the entrance from the seed point to the receiver point.\n' +
                              'Start at the ocean or northern seed point.\n' + 'When finished digitizing, click <ENTER>; ' + '# of transects= ' + str(nr_of_inputs),
                              fontsize=14)
                    plt.draw()
                    nr_of_inputs = nr_of_inputs + 1
            
                    # let user click on the shoreline
                    pts = ginput(n=50000, timeout=1e9, show_clicks=True)
                    pts_pix = np.array(pts)
                    # convert pixel coordinates to world coordinates
                    pts_world = SDS_tools.convert_pix2world(pts_pix[:,[1,0]], georef)
            
                    # interpolate between points clicked by the user (1m resolution)
                    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(pts_world)-1):
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
                    image_epsg = metadata[satname]['epsg'][i]
                    pts_world_interp_reproj = SDS_tools.convert_epsg(pts_world_interp, image_epsg, settings['output_epsg'])
                
                    #save as geometry (to create .geojson file later)
                    geoms.append(geometry.LineString(pts_world_interp_reproj))
            
                    # convert back to pixel coordinates and plot
                    pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
                    pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')
            
                    ax.set_title('manually digitized entrance path \n   \n' + filenames[i],
                              fontsize=14)
                    fig.savefig(image_out_path + '/' + filenames[i]  + '.png') 
                                        
                    # update title and buttons
                    add_button.set_visible(True)
                    #end_button.set_visible(True)
                    #ax.set_title('click on <continue> .\n to move on to  .\n the next image', fontsize=14)
                    plt.draw()
            
                    # let the user click again (<add> another shoreline or <end>)
                    pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
                    pt_input = np.array(pt_input)
            
                    # if user clicks on <end>, save the points and break the loop
                    #if pt_input[0][0] > im_ms.shape[1]/2:
                    add_button.set_visible(False)
                    plt.draw()
 
                    ginput(n=1, timeout=3, show_clicks=False)
                    
                    #empty the figure axes but don't clear the figure
                    for ax in fig.axes:
                        ax.clear()        
            
                    pts_sl = np.delete(pts_sl,0,axis=0)
                    
                    #extract NDWI alon the digitized line.
                    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                    im_ndwi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask) 
                    im_bathy_sdb = SDS_tools.bathy_index(im_ms[:,:,0], im_ms[:,:,1], cloud_mask)                    
                    
                    z_mndwi = scipy.ndimage.map_coordinates(im_mndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
                    z_ndwi = scipy.ndimage.map_coordinates(im_ndwi, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
                    z_bathy = scipy.ndimage.map_coordinates(im_bathy_sdb, np.vstack((pts_pix_interp[:,1], pts_pix_interp[:,0])),order=1)
                    XS[str(dates_dict[filenames[i]].date())+ '_' + satname + '_mndwi'] = z_mndwi
                    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_ndwi'] = z_ndwi
                    XS[str(dates_dict[filenames[i]].date()) + '_' + satname + '_bathy'] = z_bathy
                
                    # also store as .geojson in case user wants to drag-and-drop on GIS for verification
                    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms))
                    gdf.index = [k]
                    gdf.loc[k,'name'] = 'entrance line ' + str(k+1)
                    gdf.loc[k,'date'] = filenames[i][:19]
                    gdf.loc[k,'satname'] = satname
                    
                    # store into geodataframe
                    if iterator == 0:
                        gdf_all = gdf
                    else:
                        gdf_all = gdf_all.append(gdf)
                    iterator = iterator + 1   
  
    #gdf_all.crs = {'init':'epsg:'+str(image_epsg)} # looks like mistake. geoms as input to the dataframe should already have the output epsg. 
    gdf_all.crs = {'init': 'epsg:'+str(settings['output_epsg'])}
    # convert from image_epsg to user-defined coordinate system
    #gdf_all = gdf_all.to_crs({'init': 'epsg:'+str(settings['output_epsg'])})
    # save as shapefile
    gdf_all.to_file(os.path.join(csv_out_path, sitename + '_entrance_lines_' + Identifier + '.shp'), driver='ESRI Shapefile') 
    print('Entrance lines have been saved as a shapefile and in csv format for NDWI and mNDWI')
    
    #save the data to csv            
    XS_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in XS.items() ])) 
    XS_df.to_csv(os.path.join(csv_out_path, sitename + '_XS' + '_entrance_lines_' + Identifier + '.csv')) 
      
    return XS_df, gdf_all, geoms






def automated_entrance_paths(metadata, settings, settings_entrance):
    """
    Function that automatically finds the connecting path from the ocean seed point
    to the entrance receiver point, regardless weather the entrance is open or not
    
    This can be done for the entire dataset or to a limited number of images
    
    It is mainly for research and algorithm refinement purposes

    VH WRL 2020

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded
        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        ndwi_whitewhater_delta: float
            Float number by which the actual NDWI values are adjusted for pixels that are classified as whitewater by the NN classifier
        ndwi_sand_delta: float
            Float number by which the actual NDWI values are adjusted for pixels that are classified as sand by the NN classifier
        path_index:
            Currently either NDWI or mNDWI: This is the spectral index that will be used for detecting the least cost path. 
        tide_bool:
            Include analysis of the tide via pyfes global tide model - need to discuss whether to leave this included or not due to user difficulties
    Returns:
    -----------
        output: pandas dataframe withthe NDWI values along the transect for each image
        if plotbool = True: each detection will be output as a plot in png as well

    """           
    #plot font size and type
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 20}
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
    for satname in settings['inputs']['sat_list']:
        
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
            im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask,
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
      
    return XS_df, gdf_all,geoms, sat_tides_df



def extract_shorelines(metadata, settings):
    """
    Extracts shorelines from satellite images.

    KV WRL 2018

    Arguments:
    -----------
        metadata: dict
            contains all the information about the satellite images that were downloaded

        settings: dict
            contains the following fields:
        sitename: str
            String containig the name of the site
        cloud_mask_issue: boolean
            True if there is an issue with the cloud mask and sand pixels are being masked on the images
        buffer_size: int
            size of the buffer (m) around the sandy beach over which the pixels are considered in the
            thresholding algorithm
        min_beach_area: int
            minimum allowable object area (in metres^2) for the class 'sand'
        cloud_thresh: float
            value between 0 and 1 defining the maximum percentage of cloud cover allowed in the images
        output_epsg: int
            output spatial reference system as EPSG code
        check_detection: boolean
            True to show each invidual detection and let the user validate the mapped shoreline

    Returns:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # initialise output structure
    output = dict([])    
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping shorelines:')

    # loop through satellite list
    for satname in metadata.keys():

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

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
            
        # loop through the images
        for i in range(len(filenames)):

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
            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask,
                                    min_beach_area_pixels, clf)

            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                                    pixel_size, settings)

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
