import nibabel as nib
import numpy as np


path ='/home/jschuurmans/Documents/02_recurrentSF_3T/data-bids/derivatives/'
map_path = f'{path}whole_brain_second_level_group/surfaces/'

hemispheres = ['lh','rh']
for hemi in hemispheres:
    pmap = f'{map_path}{hemi}.duration_sf_pval_N16_fdr_smooth8_noart_8'

    pimg = nib.load(f'{pmap}.mgh')
    pdata = pimg.get_fdata()
    maps = [f'{map_path}{hemi}.HSF_slope_N16_fdr_smooth8_noart_8',
    f'{map_path}{hemi}.LSF_slope_N16_fdr_smooth8_noart_8']

    for mghfile in maps:
        img = nib.load(f'{mghfile}.mgh')
        data = img.get_fdata()
        signif = list(np.where(pdata<.95)[0])
        data[signif] = np.nan
        new = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new, f'{mghfile}_sigthresh.mgh')
