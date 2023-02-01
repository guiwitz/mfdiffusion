import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
from skimage.feature import blob_dog, blob_log
import pandas as pd
import skimage

def fit_coords_to_line(coords_x, coords_y, to_pred_min, to_pred_max):
    """
    Fit a line to the coordinates given by coords_x and coords_y.
    The line is fitted using RANSAC.
    
    Parameters
    ----------
    coords_x : array
        x-coordinates of the points to fit
    coords_y : array
        y-coordinates of the points to fit
    to_pred_min : int
        minimum x-value to predict
    to_pred_max : int
        maximum x-value to predict

    """
    X=coords_x
    X = X.reshape(-1, 1)
    y=coords_y

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(to_pred_min,to_pred_max)[:, np.newaxis]
    #line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)
    
    return line_y_ransac, ransac

    
def find_channel_boundary(image, margin=50):
    """
    Image to be analyzed are composed of a horizontal flow channel 
    and an interface region through which cells can migrate. In such images, three
    bright regions are visible: channel and interface boundaries. This function
    detects these boundaries and returns them.

    Parameters
    ----------
    image : array
        image to find the channel boundary in
    margin : int
        margin to add to the channel boundary

    Returns
    -------
    im_masked : array
        image with only interface region unmasked
    line_model : sklearn.linear_model._ransac.RANSACRegressor
        line model of the upper interface boundary
    lims1 : array
        upper boundary of the interface as list of y-coordinates
    lims2 : array
        lower boundary of the intrerface as list of y-coordinates
    """
    inds = []
    im_th = np.median(image)*1.
    
    im_split = np.split(image, image.shape[1]/128, axis=1)
    ind_split = np.arange(64, image.shape[1], 128)
    mean_over_splits =[x.mean(axis=1) for x in im_split]

    for ms in mean_over_splits:
        peak_ind, peak_props = find_peaks(-ms, height=-im_th, distance=200)
        sorted_peaks = np.sort(peak_ind[np.argsort(peak_props['peak_heights'])][-3::])
        inds.append(sorted_peaks)
        
    inds = [x for x in inds if len(x) == 3]
    inds = np.stack(inds)
    
    lims1, line_model = fit_coords_to_line(coords_x=ind_split, coords_y=inds[:,1], to_pred_min=0, to_pred_max=image.shape[1])
    lims2, _ = fit_coords_to_line(coords_x=ind_split, coords_y=inds[:,2], to_pred_min=0, to_pred_max=image.shape[1])

    im_masked = mask_from_boundary(image, lims1, lims2, margin)
        
    return im_masked, line_model, lims1, lims2

def mask_from_boundary(image, lims1, lims2, margin):
    """
    Given an image and the boundaries of the interface region, mask everything
    except for the interface region with the median value of the image.
    
    Parameters
    ----------
    image : array
        image to mask
    lims1 : array
        upper boundary of the interface as list of y-coordinates
    lims2 : array
        lower boundary of the intrerface as list of y-coordinates
    margin : int
        margin to add to the channel boundary

    Returns
    -------
    im_masked : array
        image with only interface region unmasked
    """
    im_median = np.median(image)
    
    im_masked = image.copy()
    for ind, (l1, l2), in enumerate(zip(lims1, lims2)):
        im_masked[0:int(l1)+margin, ind] = im_median
        im_masked[int(l2)-margin::, ind] = im_median
        
    return im_masked

def values_between_boundaries(image, lims1, lims2, margin):
    """
    Given an image and the boundaries of the interface region, return
    a list of all pixel intensities present in the interface.
    
    Parameters
    ----------
    image : array
        image from where to extract the values
    lims1 : array
        upper boundary of the interface as list of y-coordinates
    lims2 : array
        lower boundary of the intrerface as list of y-coordinates
    margin : int
        margin to add to the channel boundary

    Returns
    -------
    values : array
        pixel intensities present in the interface
    """
        
    values = []
    for ind, (l1, l2), in enumerate(zip(lims1, lims2)):
        values.append(image[int(l1)+margin: int(l2)-margin, ind])
    values = np.concatenate(values)
        
    return values

def background_peak(image, lims1, lims2, margin=100):
    """
    Given an image and the boundaries of the interface region, return
    the peak position of the distribution of pixel intensities present
    in the interface in order to estimate the background level.
    
    Parameters
    ----------
    image : array
        image from where to extract the values
    lims1 : array
        upper boundary of the interface as list of y-coordinates
    lims2 : array
        lower boundary of the intrerface as list of y-coordinates
    margin : int
        margin to add to the channel boundary

    Returns
    -------
    peak : float
        position of histogram peak
    threshold : float
        position of the right half-maximum of the histogram peak

    """

    values = values_between_boundaries(image, lims1=lims1, lims2=lims2, margin=margin)
    hist, bin_edges = np.histogram(values, bins=np.arange(0,values.max(),10))
    peak = bin_edges[np.argmax(hist)]

    peak_ind = np.argmax(hist)
    peak_val = hist[peak_ind]
    bin_pos = bin_edges[peak_ind]

    x = peak_ind
    while hist[x] > peak_val / 2:
        x = x+1
        
    threshold = bin_edges[x]

    return peak, threshold

def points_intensity(image_intensity, points, expand_points=5):
    """
    Given an image and a list of points, return properties of the area around
    the points including position and mean intensity. Some points can be close to each
    other, so point expansion can merge them.

    Parameters
    ----------
    image_intensity : array
        image from where to extract the values
    points : array
        list of points as (x,y) coordinates
    expand_points : int
        number of pixels to expand the points

    Returns
    -------
    props : pandas dataframe
        dataframe with properties of the area around the points
    
    """
    im_positions = np.zeros(image_intensity.shape, dtype=np.uint16)
    
    im_positions[points[:,0], points[:,1]] = 1
    im_positions = skimage.morphology.binary_dilation(im_positions, skimage.morphology.disk(expand_points))
    im_labels = skimage.morphology.label(im_positions)

    props = pd.DataFrame(skimage.measure.regionprops_table(im_labels, image_intensity,
                                            properties=('label', 'mean_intensity','centroid')))
    props = props.rename(columns={'centroid-0': 'ypos', 'centroid-1': 'xpos'})

    return props

def stack_processing(image_channel, image_fluo, time, margin=100, threshold=0.1, overlap=0.1, min_sigma=5, max_sigma=7):
    """
    Given a phase and a fluorescence image, find the interface region in the phase image,
    detect cells in that same channel. Use the fluorescence image to remove false positives.
    
    Parameters
    ----------
    image_channel : array
        phase image
    image_fluo : array
        fluorescence image
    time : int
        time point
    margin : int
        margin to add to the channel boundary
    threshold, overlap, min_sigma, max_sigma : float
        unused parameters for old blob detection algorithm

    Returns
    -------
    props : pandas dataframe
        dataframe with properties of the cells detected in the interface region

    """

    im_masked, line_model, lims1, lims2 = find_channel_boundary(image_channel, margin=margin)
    im_fluo_masked = mask_from_boundary(image_fluo, lims1, lims2, margin=margin)
    #seg = blob_log(image=im_fluo_masked, min_sigma=4, max_sigma=5, threshold=0.05, overlap=0.1)
    #seg = blob_log(image=im_masked, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)

    #im_match1 = skimage.feature.match_template(im_masked,skimage.morphology.disk(6), pad_input=True)
    #im_match2 = skimage.feature.match_template(im_masked,skimage.morphology.disk(10), pad_input=True)
    #peaks1 = skimage.feature.peak_local_max(im_match1, min_distance=5, threshold_abs=0.7)
    #peaks2 = skimage.feature.peak_local_max(im_match2, min_distance=5, threshold_abs=0.5)
    #seg = np.concatenate([peaks1, peaks2])

    seg = detect_cells(im_masked)
    seg = np.concatenate(seg)

    _, background_th  = background_peak(image_fluo, lims1, lims2, margin=100)

    props = None    
    if len(seg)>0:
        props = points_intensity(image_intensity=im_fluo_masked, points=seg)
        props['y_lim'] = line_model.predict(props['xpos'].values[:,np.newaxis])
        props['time'] = time
        props['background'] = background_th

    return props

def detect_cells(image):
    """
    Detect cells in the image by using a series of template matching with different
    templates considering all apparent cell shapes caused by out of focus.

    Parameters
    ----------
    image : array
        image in which to detect cells
    
    Returns
    -------
    all_peaks : list of arrays
        list of arrays with the coordinates of the detected cells for different templates
     
    """
    
    all_peaks = []
    donuts = genereate_templates()
    threshold_abs = [0.6, 0.6, 0.6, 0.8, 0.8]
    for d, th in zip(donuts, threshold_abs):
        matched = skimage.feature.match_template(image, d, pad_input=True)
        peaks = skimage.feature.peak_local_max(matched, min_distance=5, threshold_abs=th)
        all_peaks.append(peaks)
    
    return all_peaks

def genereate_templates():
    """
    Function to generate the templates used for cell detection.

    Returns
    -------
    donuts : list of arrays
        list of image templates
    """

    donuts = []
    rls = [7,9,11]
    rss = [5,7,9]

    for r_l, r_s in zip(rls, rss):
        small_disk = skimage.morphology.disk(r_s)
        large_disk = skimage.morphology.disk(r_l)
        small_disk = np.pad(small_disk, (large_disk.shape[0] - small_disk.shape[0])//2)

        donut = large_disk - small_disk
        donut = np.pad(donut,2)
        donut_filt = skimage.filters.gaussian(donut,2)
        
        donuts.append(donut_filt)
        
    rls = [9,11]
    rss = [7,9]

    for r_l, r_s in zip(rls, rss):
        small_disk = skimage.morphology.disk(r_s).astype(float)
        large_disk = skimage.morphology.disk(r_l).astype(float)
        small_disk = np.pad(small_disk, (large_disk.shape[0] - small_disk.shape[0])//2)

        donut = -2*large_disk + 3* small_disk
        donut = np.pad(donut,2)
        donut_filt = skimage.filters.gaussian(donut,2)

        donuts.append(donut_filt)

    return donuts