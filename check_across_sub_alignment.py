# conda activate /home/joliens/Documents/02_recurrentSF_3T/analysis/analysis_MB/fmri
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

base_path = "/home/joliens/Documents/02_recurrentSF_3T/"
data_path = f'{base_path}data-bids/derivatives/firstlevel-main/'

subs = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '18']
runs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
conditions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

nifti_paths = []
for subject in subs:
    for run in runs:
        for condition in conditions:
            sub_path = f'{data_path}sub-{subject}/run-{run}.feat/stats/'
            file_name = f'{sub_path}cope{condition}_MNI.nii.gz'
            nifti_paths.append(file_name)

nifti = nib.load(nifti_paths[0])
data = nifti.get_fdata()
all_niftis = np.zeros((data.size, len(nifti_paths)), dtype=bool)
# for A, compute A'... that's it
for i, file_name in enumerate(nifti_paths):
    print(i)
    nifti = nib.load(file_name)
    data = nifti.get_fdata()
    where = data!=0
    all_niftis[:,i] = where.flatten()
all_niftis = all_niftis * 1
out = np.matmul(np.transpose(all_niftis), all_niftis)
out.shape

plt.matshow(out)


