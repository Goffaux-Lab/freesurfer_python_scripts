# freesurfer_python_scripts

Several of these scripts use some functions that are now part of the
[MRI_surface_functions](https://github.com/Goffaux-Lab/MRI_surface_functions)
repository (which means that they could be rewritten importing that library and
using the up to date functions directly like so:

```python
import mesh_functions as mf

mask = mf.get_multi_neighbours(G, artifacts, dilation)
```

#### check_across_sub_alignment.py
This script (I think... from looking at it) makes a matrix showing the
proportion of overlap in in the location of the brain across
subjects/runs/conditions. I think I was using this to ensure that the MNI
transform was moving all subjects to the same space...

#### post_ant_plot.py
This was part of an analysis to look at effects along the posterior-anterior
axis. The functions I wrote ended up in the
[MRI_volume_functions](https://github.com/Goffaux-Lab/MRI_volume_functions).
So this could now be rewritten like so: 

```python
import volumes as vol

labels = vol.mask_data(labels, mask)
```

#### proj_specific.py
Setting common path names, subjects numbers, conditions, runs etc. so they
don't have to be entered in at the top of every script.

#### remove_artifacts.py
This script removes some annoying artifacts from surface maps that resulted
from interpolation from MNI to surface space. 
Uses [MRI_surface_functions](https://github.com/Goffaux-Lab/MRI_surface_functions).

#### sigthresh.py
Script to make non-significant part of map into NaN values (this was needed for
showing in freesurfer, since these maps were made outside of freesurfer and
thus freesurfer behaved strangely with them).

Same as
[threshold_mgh.py](https://github.com/Goffaux-Lab/freesurfer_python_scripts#threshold_mghpy).
I don't know which script we used in the end. One should be removed from this
repository.

#### smooth_mgh_files.py
Smoothing some surface maps since freesurfer wouldn't apply the smoothing when
called [from a script](https://github.com/Goffaux-Lab/freesurfer_bash_scripts/blob/main/screenshots.sh)
that we were using to make a lot of screenshots automatically.
Uses [MRI_surface_functions](https://github.com/Goffaux-Lab/MRI_surface_functions).

#### smooth_roi_labels.py
In order to create ROI outlines we use the fsl commands `mri_binarize` and
`mri_cor2label` - but there was a problem since the outlines were very wiggly
and looked unprofessional. So we smoothed the maps a lot before applying the
commands after which the outlines look more sensible.
Uses [MRI_surface_functions](https://github.com/Goffaux-Lab/MRI_surface_functions).

#### threshold_mgh.py
Same as
[sigthresh.py](https://github.com/Goffaux-Lab/freesurfer_python_scripts#sigthreshpy).
I don't know which script we used in the end. One should be removed from this
repository.

