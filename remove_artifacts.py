import nibabel as nib
import numpy as np
import networkx as nx
import itertools
import mesh_functions
## LOAD FILES #################################################################
path ='/home/jschuurmans/Documents/02_recurrentSF_3T/data-bids/derivatives/'
map_path = f'{path}whole_brain_second_level_group/surfaces/'
surf_path =f'{path}freesurfer/fsaverage/surf/'

# number of dilation steps around artifacts (p > 1) to remove
dilation = 8

## DEFINE FUNCTIONS ###########################################################

def get_neighbours(G, nodes):
    '''Get neighbours and associated values of set of nodes as dictionary'''
    if isinstance(nodes, int):
        nodes = [nodes]
    node_neighbours = []
    vals = []
    for node in nodes:
        for neighbours, _ in G.adj[node].items():
            node_neighbours.append(neighbours)
    return node_neighbours

def get_multi_neighbours(G, nodes, neighbourhood_size):
    neighbourhood = nodes
    neighbours = nodes
    for i in range(neighbourhood_size):
        neighbours = get_neighbours(G, neighbours)
        neighbours = list(set(neighbours) - set(neighbourhood))
        neighbourhood += neighbours
    return list(set(neighbourhood))

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

    maps = [f'{map_path}{hemi}.duration_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_sf_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_sf_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_sf_stimtype_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_sf_stimtype_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_stimtype_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.duration_stimtype_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.empty_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_minus_LSF_N16_sigthresh_fdr_smooth8',
    f'{map_path}{hemi}.HSF_neg_minus_LSF_neg_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_neg_minus_LSF_neg_N16_fdr_smooth8_signif',
    f'{map_path}{hemi}.HSF_neg_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_pos_minus_LSF_pos_MINUS_HSF_neg_minus_LSF_neg_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_pos_minus_LSF_pos_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_pos_minus_LSF_pos_N16_fdr_smooth8_signif',
    f'{map_path}{hemi}.HSF_pos_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.HSF_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.LSF_neg_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.LSF_pos_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.LSF_slope_N16_fdr_smooth8',
    f'{map_path}{hemi}.sf_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.sf_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.sf_stimtype_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.sf_stimtype_pval_N16_fdr_smooth8',
    f'{map_path}{hemi}.stimtype_F_N16_fdr_smooth8',
    f'{map_path}{hemi}.stimtype_pval_N16_fdr_smooth8']
    
    for mghfile in maps:
        img = nib.load(f'{mghfile}.mgh')
        data = img.get_fdata()
        artifacts = list(np.where(data>1)[0])
        mask = get_multi_neighbours(G, artifacts, dilation)
        data[mask] = 0
        new = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new, f'{mghfile}_noart_{dilation}.mgh')
