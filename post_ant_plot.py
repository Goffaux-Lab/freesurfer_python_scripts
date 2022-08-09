#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage.filters as f
import nibabel as nib
from itertools import islice
from copy import deepcopy as dc
import os

def apply_to_axes(axes,  title='', thresh=None):
    """
    Execute some commands on every axis
    Inputs:
        axes: <List> of axis objects

        title: Title for plot (<string>)

        thresh: <int> If passed, threshold will be plotted as a horizontal line
    """
    for ax in axes:
        ax.set_title(title, y=1, pad=-14)
        ax.set_ylim(-3, 0)
        if thresh:
            ax.axhline(thresh)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_linewidth(2)

def our_func(data, average_bin_size=1, thresh=0.35):
    """
    Function to be applied to data. MUST return a single value!

    Inputs:
        data:   1D numpy array

    Returns:
        result: result of the function (if data is empty, returns nan value)
    """
    result = np.nan
    thresh_vals = data[data<thresh]
    if thresh_vals.size:
        result = np.mean(data[data<thresh])
    return result

def label_do(data, labels, func=np.mean, *args, **kwargs):
    """
    Apply any function (default np.mean) to each set of labelled data voxels
    and return the results as a 1D numpy array

    Inputs:
        data:           Numpy array

        labels:         Numpy array of the same size as data, containing numpy
                        array of non-zero int label value for each voxel

        func:           Function to be applied to each set of voxels labelled
                        in data. The function MUST return a single value! (the
                        default fuction is np.mean)

        *args:          Any non-named arguments to pass to func

        **kwargs:       Any named arguments to pass to func

    Returns:
        func_results:   1D numpy array containing the value returned from the
                        function from each set of labelled voxels
    """
    func_results = np.zeros((len(np.unique(labels))-1))
    for idx, label in enumerate(np.unique(labels[labels>0])):
        func_results[idx] = func(data[labels==label], *args, **kwargs)
    return func_results

def split_by_label(data, labels):
    """
    Put <data> into a dictionary with each label number from <labels> as a
    'key' and a 1D numpy array containing the set of labelled data as the
    'value'

    Inputs:
        data:               Numpy array

        labels:             Numpy array of the same size as data, containing
                            numpy array of non-zero int label value for each
                            voxel

    Returns:
        labelled_voxels:    Dictionary for each set of labelled voxels, with
                            each label number as a 'key' and the 1D numpy
                            array containing the set of labelled data as the
                            'value'
    """
    labelled_voxels = {}
    for label in np.unique(labels[labels>0]):
        labelled_voxels[label] = data[labels==label]
    return labelled_voxels

def chunks(data, n_chunks=10):
    """
    Inputs:
        data:       Dictionary with keys being numbers and vaules being 1D
                    numpy arrays

        n_chunks:   Number of keys for resulting dictionary (default 10)

    Returns:
        A dictionary containing one key per chunk, and containing all the
        values from the original keys that fit into each chunk

    """
    size = int(np.ceil((max(data) - min(data))/n_chunks))
    it = iter(data)
    for plot_count in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}

def chunk_summarise(chunk, thresh=-float('inf')):
    """
    Takes in a dictionary and calculates the mean and standard deviation of the
    values associated with each key

    Inputs:
        chunk:          Dictionary containing keys as numbers and values as 1D
                        numpy arrays

        thresh:         Threshold to apply to data before summarising. Any data
                        not passing the threshold won't be inlcuded (default
                        any number that isn't negative infinity i.e. ~ any
                        number)
    Returns:
        overall_mean:   Mean of the data passing the threshold

        overall_std:    Standard deviation of the data passing the threshold
    """
    overall_mean = np.nan
    overall_std = np.nan
    overall_vals = np.zeros(1,)
    for values in chunk.values():
        thresh_vals = values[values<thresh]
        if thresh_vals.size:
            overall_mean =+ np.mean(thresh_vals)
            overall_vals = np.concatenate((overall_vals, thresh_vals), axis=None)
    if overall_vals.size > 1:
        overall_std = np.std(overall_vals[1:])
    return overall_mean, overall_std

def binarise_mask(mask, thresh=0):
    """
    Take a numpy array and return a boolean mask showing where values are above
    some threshold

    Inputs:
        mask: Numpy array of zero and non-zero values

        thresh: <Float> for which any value in <mask> less than <thresh> will
                be set to False (default zero)

    Returns:
        Numpy array of boolean values
    """
    return mask>thresh

def merge_masks(masks, *args, **kwargs):
    """
    Take a list of numpy arrays and sum them to make a single numpy array. Any
    vaules above some threshold (default zero) are set to True and False
    otherwise

    Inputs:
        masks: <List> of numpy arrays all the same shape

        thresh: <Float> for which any value in any <mask> less than <thresh>
                will be set to False (default zero). See help(binarise_mask)

    Returns:
        a single numpy array of boolean values
    """
    overall_mask = dc(masks[0])
    for mask in masks[1:]:
        overall_mask += mask
    return binarise_mask(overall_mask, *args, **kwargs)

def mask_data(data, mask, non_mask_value=0, *args, **kwargs):
    """
    Set any values in <data> falling outside <mask> to <non_mask_value>

    Inputs:
        data:           Numpy array of numbers

        mask:           Numpy array of booleans (or zero and non-zero values)
                        the same shape as data

        non_mask_value: What to set the data values falling outside the mask to
                        (default zero)

        thresh:         If <mask> is not already boolean, a threshold may be
                        for which any value in any <mask> less than <thresh>
                        will be set to False (default zero). See
                        help(binarise_mask)

    Returns:
        data:           The original data, but with any values falling outside
                        the mask set to <non_mask_value>
    """
    data, mask = dc(data), dc(mask)
    mask = binarise_mask(mask, *args, **kwargs)
    data[~mask] = non_mask_value
    return data

whichPC = 2 # 0 = Matt / 1 = Jolien / 2 = pony
if whichPC == 0:
    account = 'joliens'
    seedhemi = 'left'
elif whichPC == 1:
    account = 'jschuurmans'
    seedhemi = 'right'
elif whichPC == 2:
    account = 'joliens'
    seedhemi = 'both'

# paths
base_path = f'/home/{account}/Documents/02_recurrentSF_3T/data-bids/derivatives/'
ppi_path = f'{base_path}ppi/'
plot_path = f'{ppi_path}plots/'
thresh = -1
nbins = 10
name_addon = f'_{seedhemi}_{nbins}bins_thrlt1'

if not os.path.exists(f'{ppi_path}plots/'):
    os.system(f'mkdir {ppi_path}plots/')

roi_list={'14':'InferiorTemporalGyrusAnterior',
         '15':'InferiorTemporalGyrusPosterior',
         '16':'InferiorTemporalGyrusTemporooccipital',
         '23':'LateralOccipitalCortexInferior'}
         #'37':'TemporalFusiformCortexAnterior',
         #'38':'TemporalFusiformCortexPosterior',
         #'39':'TemporalOccipitalFusiformCortex',
         #'40':'OccipitalFusiformGyrus'}

condition_names = {
    '1': 'pos 50 HSF',
    '3': 'pos 83 HSF',
    '5': 'pos 100 HSF',
    '7': 'pos 150 HSF',
    '2': 'pos 50 LSF',
    '4': 'pos 83 LSF',
    '6': 'pos 100 LSF',
    '8': 'pos 150 LSF',
    '9': 'neg 50 HSF',
    '11': 'neg 83 HSF',
    '13': 'neg 100 HSF',
    '15': 'neg 150 HSF',
    '10': 'neg 50 LSF',
    '12': 'neg 83 LSF',
    '14': 'neg 100 LSF',
    '16': 'neg 150 LSF',
}

subject_names = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','18']
# subject_names = ['01','02','03','04','06','07','08','09','10','11','12','13','14','15','18']

# find the RGB values of the jet map for the posterior/anterior axis
colors = []
cmap = plt.cm.get_cmap('jet')
for plot_count, bin_idx in enumerate(range(nbins)):
    rgba = cmap((bin_idx+1)/nbins)
    rgb = list(rgba[0:3])
    # rgb = (0.6, 0.6, 0.6)
    # if plot_count % 2:
    #     rgb = [col/2 for col in rgb]
    colors.append(rgb)

# better_col_order = []
# for splice in zip(range(int(nbins/2)), range(int(nbins/2), nbins)):
#     better_col_order.append(splice[0])
#     better_col_order.append(splice[1])

edgecolors = [[0, 0.8, 0.9],
             [0, 0.2, 0.5],
             [1, 0.5, 0.7],
             [0.9, 0, 0]]


chunk_means_all_subs = []
for sub in subject_names:
    print(f'sub-{sub}')
    anat_path = f'{base_path}fmriprep/sub-{sub}/anat/'

    # load subject's mask
    labels = nib.load(f'{anat_path}sub-{sub}_label-mnistripeFunc_T1w.nii.gz').get_fdata()

    # load masks
    rois = []
    for roi_num,roi_name in roi_list.items():
        roi_name = f"{anat_path}ROIs/sub-{sub}_label-{roi_num}{roi_name}Func_roi"
        rois.append(nib.load(f'{roi_name}.nii.gz').get_fdata())

    mask = merge_masks(rois)

    # # delete the left half of the mask.
    # mask[round(mask.shape[1]/2):,:,:] = 0

    # delete the right half of the mask.
    mask[:,round(mask.shape[1]/2):,round(mask.shape[2]/2):] = 1

    # Right-Left x Posterior-Anterior

    # MAKE THIS A HELPER FUNCTION
    for i in range(1,60):
        plt.subplot(6,10,i)
        plt.imshow(mask[:,:,i])
    plt.show()

    breakpoint()

    # restrict the labels to the mask
    labels = mask_data(labels, mask)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    plt.close('all')
    plot_count = 0
    chunk_means_all_conds = []
    for condition, condition_name in condition_names.items():

        # load functional data
        func_data = nib.load(f'{ppi_path}secondlevel/sub-{sub}/secondlevel_{seedhemi}.gfeat/cope{condition}.feat/stats/pe1.nii.gz').get_fdata()

        # update plot count
        plot_count += 1

        # labels = np.ceil(labels)
        labelled_voxels = split_by_label(func_data, labels)
        # print(min(labelled_voxels), max(labelled_voxels))

        # plot each set of voxels according to label
        matplotlib.rc('axes',edgecolor=edgecolors[int(np.floor((plot_count-1)/4))])

        # plot individual voxels, per subject
        ax1 = fig1.add_subplot(4,4,plot_count)
        ax1.set_xlim(30,150)
        # plot average across voxel bins, per subject
        ax2 = fig2.add_subplot(4,4,plot_count)
        ax2.set_xlim(-1,nbins)

        if (plot_count % 4) != 1:
            ax1.get_yaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

        if plot_count < 13:
            ax1.get_xaxis().set_visible(False)
            ax2.get_xaxis().set_visible(False)

        # put average curve of the individual voxels
        # average = label_do(func_data, labels)
        average = label_do(func_data, labels, our_func, thresh=thresh)

        chunk_means, chunk_stds = [], []
        # plot individual voxels across posterior/anterior
        for col_idx, chunk in enumerate(chunks(labelled_voxels,nbins)):
            for key, vals in chunk.items():
                thresh_vals = vals[vals<thresh]
                x_coords = np.full((1, len(thresh_vals)),key)
                # ax1.scatter(x_coords, thresh_vals, marker='.', s=1, color=colors[better_col_order[col_idx]])
                ax1.scatter(x_coords, thresh_vals, marker='.', s=2, color=colors[col_idx])

            apply_to_axes([ax1, ax2], title=condition_name, thresh=thresh)

            ax1.plot(range(int(min(labelled_voxels)),
                     int(min(labelled_voxels))+len(labelled_voxels)),
                     average, '.k', markersize=3)

            mean, std = chunk_summarise(chunk, thresh)
            chunk_means.append(mean), chunk_stds.append(std)
            ax2.plot([col_idx, col_idx], [mean+std, mean-std], '-k', linewidth=1)

        ax2.plot(chunk_means, '.k', markersize=10, linewidth=2)

        chunk_means_all_conds.append(chunk_means)

    fig1.suptitle(f'subject {sub}', fontsize=24)
    fig1.set_size_inches(12,12)
    fig1.savefig(f'{plot_path}sub-{sub}_sep_cond_sep_vox{name_addon}.png')

    fig2.suptitle(f'subject {sub}', fontsize=24)
    fig2.set_size_inches(12,12)
    fig2.savefig(f'{plot_path}sub-{sub}_chunk_mean{name_addon}.png')

    chunk_means_all_subs.append(chunk_means_all_conds)

plt.close('all')

# plot all subjects (subs x conds x chunks) (15 x 16 x 9)
chunk_means_all_subs = np.array(chunk_means_all_subs)

plot_count = 0
for condition, condition_name in condition_names.items():
    data = chunk_means_all_subs[:,plot_count,:]
    plot_count += 1
    # plot each set of voxels according to label
    matplotlib.rc('axes',edgecolor=edgecolors[int(np.floor((plot_count-1)/4))])
    ax = plt.subplot(4,4,plot_count)
    ax.plot(np.transpose(data), '.', color=[0.5, 0.5, 0.5])
    ax.set_title(f'{condition_name}', y=1, pad=-14)
    ax.set_ylim(-2.5, -0.7)
    if (plot_count % 4) != 1:
        ax.get_yaxis().set_visible(False)
    if plot_count < 13:
        ax.get_xaxis().set_visible(False)
    ax.axhline(thresh)
    ax.plot(np.nanmean(data, axis=0), '-k')
plt.gcf().set_size_inches(12,12)
plt.savefig(f'{plot_path}subs_N{len(subject_names)}_sep_cond{name_addon}.png')
plt.close('all')

