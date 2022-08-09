import os
import nibabel as nib
import numpy as np
import networkx as nx
import itertools
import mesh_functions as mesh 
## LOAD FILES #################################################################
path ='/home/jschuurmans/Documents/02_recurrentSF_3T/data-bids/derivatives/'
map_path = f'{path}masks/mni_masks_per_roi/'
surf_path =f'{path}freesurfer/fsaverage/surf/'

smoothing_iter = 5
kern_size = 1
threshold = 3

## CONVERT NODES/EDGES/ TO GRAPH OBJECT #######################################

hemispheres = ['lh','rh']
for hemi in hemispheres:
    mesh_filename = f'{hemi}.inflated_pre'
    # each row of mesh_coords are the x, y, z coordinates of a node
    # each row of mesh_faces are the nodes that define that face
    mesh_coords, mesh_faces = nib.freesurfer.io.read_geometry(f'{surf_path}{mesh_filename}')

    nodes_to_add = np.unique(mesh_faces)

    G = nx.Graph()
    # construct the graph nodes and edges
    G.add_nodes_from(nodes_to_add)
    for i, row in enumerate(mesh_faces):
        G.add_edges_from(list(itertools.combinations(row, 2)))

    maps = [f'{map_path}{hemi}.{hemi[0]}FFA',
            f'{map_path}{hemi}.{hemi[0]}V1']
    
    for mghfile in maps:
        img = nib.load(f'{mghfile}.mgh')
        map_data = img.get_fdata()
        
        # This gives an array of (136107,) - for one per node - from retmap giftiImage
        # essentaially the colours of the retmap (going from blue in fov, to red in peri)
        # nans are presumerably parts of the mesh that weren't in the functional slab
        # map_data = img.darrays[0].data

        # dictionary of attributes to add to graph
        color_map_dict = {}
        for i, color in zip(G.nodes, map_data):
            color_map_dict[i] = {"retmap_val": color}

        # add the attributes
        nx.set_node_attributes(G, color_map_dict)

        # smooth the map
        G = mesh.smooth_graph(G, None, smoothing_iter,kern_size)

        # extract the smoothed retmap values
        mgh_value = np.array(mesh.get_node_attributes_as_list(G, key="retmap_val"))
        mgh_value.resize(map_data.shape) 
        data = np.memmap('filename.dat', mode='w+', shape=map_data.shape)
        data[:] = mgh_value[:]

        index = list(np.where(data<threshold)[0])
        data[index] = 0
        new = nib.Nifti1Image(data, img.affine, img.header)
        name = f'{mghfile}_smooth{smoothing_iter}x{kern_size}_thresh{threshold}'
        nib.save(new, f'{name}.mgh')
        
        os.system(f'mri_binarize --i {name}.mgh --min {threshold} --o {name}_bin.mgh')
        os.system(f'mri_cor2label --i {name}_bin.mgh --surf fsaverage {hemi} --id 0 --l {name}.label')

