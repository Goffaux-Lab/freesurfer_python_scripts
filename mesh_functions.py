import nibabel
import numpy as np
import networkx as nx
import itertools

## DEFINE FUNCTIONS ###########################################################

def get_node_attributes_as_list(G, nodes=None, key=None):
    '''Extract node attributes: from dictionary take the values based on <key>
    (which must by a string).'''
    if not nodes:
        nodes = G.nodes()
    # extract attribute
    tmp = []
    for i in nodes:
        tmp.append(G.nodes[i][key])
    return tmp


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

def get_neighbours_and_vals(G, nodes):
    '''Get neighbours and associated values of set of nodes as dictionary'''
    if isinstance(nodes, int):
        nodes = [nodes]
    node_neighbours = []
    vals = []
    for node in nodes:
        for neighbours, _ in G.adj[node].items():
            node_neighbours.append(neighbours)
            vals.append(G.nodes[neighbours]["retmap_val"])
    return dict(zip(node_neighbours, vals))

def get_multi_neighbours_and_vals(G, nodes, neighbourhood_size):
    neighbourhood = {}
    for i in range(neighbourhood_size):
        neighbours = get_neighbours_and_vals(G, nodes)
        neighbourhood.update(neighbours)
    return neighbourhood

def is_node_on_region_border(G, region_nodes, node):
    '''For a graph <G>, and a patch-like subset of its nodes <region_nodes>,
    does a particular <node> lay on the border of that subset?'''
    neighbours = get_neighbours_and_vals(G, [node])
    total_neighbours = len(set(neighbours.keys()))
    n_nodes_in_region = len(set(neighbours.keys()).intersection(label_coords))
    return n_nodes_in_region < total_neighbours

def find_region_border(G, nodes):
    '''Return the nodes that have neighbours in the graph that don't appear in
    the original set of nodes'''
    border_nodes = []
    for node in nodes:
        if is_node_on_region_border(G, nodes, node):
            border_nodes.append(node)
    return border_nodes

def remove_out_of_region_nodes(G, region_nodes, nodes):
    # intersection is taking the overlapping part in a venn diagram
    return list(set(region_nodes).intersection(set(nodes)))

# def expand_nodes(G, nodes, stepsize=1):
#     orig_nodes = nodes[:]
#     neighbours = get_multi_neighbours_and_vals(G, nodes, stepsize)
#     nodes = neighbours.keys()
#     new_nodes=list(set(nodes)-set(orig_nodes))
#     return nodes, new_nodes

def expand_nodes(G, nodes, stepsize=1):
    orig_nodes = nodes[:]
    for i in range(stepsize):
        neighbours = get_neighbours_and_vals(G, nodes)
        nodes += neighbours.keys()
    new_nodes=list(set(nodes)-set(orig_nodes))
    return nodes, new_nodes


# will be useful for gradient ascent
def max_neighbour(G, node, neighbourhood_size=1):
    '''Return node with maximum retmap value amoung neighbours (and neighbours
    of neighbours etc...)'''
    neighbours = get_multi_neighbours_and_vals(G, [node], neighbourhood_size)
    return (max(neighbours, key=neighbours.get), max(neighbours.values()))

# makes the whole path to take a step in the right direction
# each node here is treated independently
def nodes_gradient_step(G, nodes, stepsize=1):
    '''Return new nodes positions where each node is replaced by that node's
    maximum neighbour in <stepsize>'''
    retmap_values = get_node_attributes_as_list(G, nodes, key="retmap_val")
    new_positions = []
    for node, retval in zip(nodes, retmap_values):
        max_info = max_neighbour(G, node, neighbourhood_size=stepsize)
        if retval < max_info[1]:
            new_positions.append(max_info[0])
        else:
            new_positions.append(node)
    return new_positions

# def smooth_graph(G, n_its=1):
#     '''Smooth all nodes of retmap (replace each node with mean of neighbours)'''
#     G_smooth = G.copy()
#     for it in range(n_its+1):
#         print(f'Smoothing iteration: {it}/{n_its}')
#         color_map_dict = {}
#         for i, node in enumerate(G.nodes()):
#             out = get_neighbours_and_vals(G_smooth, [node])
#             mean = np.nanmean(list(out.values()))
#             color_map_dict[i] = {"retmap_val": mean}
#         nx.set_node_attributes(G_smooth, color_map_dict)
#     return G_smooth

def smooth_graph(G, nodes=None, n_its=1, kernel_size=1):
    '''Smooth all nodes of retmap (replace each node with mean of neighbours)'''
    if not isinstance(nodes, list):
        nodes = G.nodes()
    G_smooth = G.copy()
    for it in range(n_its):
        print(f'Smoothing iteration: {it+1}/{n_its}')
        color_map_dict = {}
        for node in nodes:
            out = get_multi_neighbours_and_vals(G_smooth, [node], kernel_size)
            mean = np.nanmean(list(out.values()))
            color_map_dict[node] = {"retmap_val": mean}
        nx.set_node_attributes(G_smooth, color_map_dict)
    return G_smooth

# some functions for plotting
def setzoomed3Dview(ax):
    ax.azim =-77.20329531963876
    ax.elev =-3.8354678562436106
    ax.dist = 2.0
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_facecolor('black')
    return None

def set3Dview(ax):
    ax.azim = -60
    ax.elev = -16
    ax.dist = 5
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_facecolor('black')
    return None

def plot_nodes(mesh_coords, retmap_data, node_sets, colors = ['white', 'black', 'pink']):
    '''nodes_sets is a list of upto 3 sets of nodes to draw - each will have a
    different colour'''
    ax = plt.axes(projection='3d')
    ax.scatter3D(mesh_coords[:, 0], mesh_coords[:, 1],
                 mesh_coords[:, 2], s=1, c=retmap_data, cmap='jet')
    for nodes, color in zip(node_sets, colors):
        ax.scatter3D(mesh_coords[nodes, 0], mesh_coords[nodes, 1],
                     mesh_coords[nodes, 2], marker='o', s=30, c=color)
    setzoomed3Dview(ax)
    return ax

