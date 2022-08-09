import nibabel as nib
import numpy as np

path ='/home/jschuurmans/Documents/02_recurrentSF_3T/data-bids/derivatives/'
# path ='/home/joliens/Documents/02_recurrentSF_3T/data-bids/derivatives/'
map_path = f'{path}whole_brain_second_level_group/surfaces/'
surf_path =f'{path}freesurfer/fsaverage/surf/'

maps = [
        f'HSF_neg_minus_LSF_neg_N16_fdr_smooth8_noart_8',
        f'HSF_neg_slope_N16_fdr_smooth8_noart_8',
        f'HSF_pos_minus_LSF_pos_N16_fdr_smooth8_noart_8',
        f'HSF_pos_slope_N16_fdr_smooth8_noart_8',
        f'HSF_slope_N16_fdr_smooth8_noart_8',
        f'LSF_neg_slope_N16_fdr_smooth8_noart_8',
        f'LSF_pos_slope_N16_fdr_smooth8_noart_8',
        f'LSF_slope_N16_fdr_smooth8_noart_8'
        ]

hemispheres = ['lh','rh']
for hemi in hemispheres:
    for mghfile in maps:
        Fimg = nib.load(f'{map_path}{hemi}.{mghfile}.mgh')
        p = nib.load(f'{map_path}{hemi}.duration_sf_pval_N16_fdr_smooth8_noart_8.mgh').get_fdata()

        F = Fimg.get_fdata()

        F[p<0.95]=np.nan

        new = nib.Nifti1Image(F, Fimg.affine, Fimg.header)
        nib.save(new, f'{map_path}{hemi}.{mghfile}_signif.mgh')

