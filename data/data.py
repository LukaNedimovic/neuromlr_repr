import pickle           # Loading and writing data
from tqdm import tqdm   # Loading progress visualization
import geopandas as gpd # Geospatial data processing

from .paths import * # Import the path constants

from argparse import Namespace # Typing

from collections import defaultdict

import numpy as np
import random

import multiprocessing as mp
import networkx as nx

df_nodes = gpd.read_file(NODE_DATA) # Format: [y, x, osmid, highway, ref, geometry]
df_edges = gpd.read_file(EDGE_DATA) # Format: [osmid, oneway, name, highway, length, lanes, tunnel, bridge, access,
                                    #          maxspeed, ref, width, service, junction, u, v, key, fid, geometry]

# Explore data yourself!
# print(df_nodes.head())
# print(df_edges.head())

col_osmid = df_nodes["osmid"].to_numpy()    # column  "osmid"
cols_yx   = df_nodes[["y", "x"]].to_numpy() # columns "y" and "x"

# Creates dictionary, where key is the edge (osmid of junction), and value is a pair of (y, x) coordinates
nodes_coords = {osmid: (y, x) for osmid, (y, x) in zip(col_osmid, cols_yx)}

edge_id_to_uv = df_edges[["u", "v"]].to_numpy() # columns "u" and "v"
uv_to_edge_id = {(u, v): idx for idx, (u, v) in enumerate(edge_id_to_uv)} # Map each entry (edge) to its index
unique_edge_labels = list(uv_to_edge_id.values()) # Replaceable with range(len(map_u_v_to_edge_id)) ??? TODO: Check


def condense_edges(edge_route: list) -> list:
    """
    TODO: Desc.
    """
    
    global edge_id_to_uv, uv_to_edge_id
    route = [uv_to_edge_id[tuple(edge_id_to_uv[e])] for e in edge_route]
    
    return route


def fetch_map_fid_to_zero_indexed(data: list) -> dict:
    """
    Maps edges to fixed zero-indexed id.
    
    Parameters
    ----------
    data : list
        Trips data.
    
    Returns
    -------
    mapping : dict
        Mapping of edges to their 0-indexed fixed dataset-specific counterpart.
    """
    
    s = set()
    for _, t, _ in data:
        s.update(set(t)) # Insert every edge into the set
    
    mapping = {el: idx for idx, el in enumerate(s)} # Enumerate the set and map edge to its position in set
    
    return mapping 


def relabel_trips(data : list, mapping : dict) -> list:
    """
    Relabel trips such that edges get new 0-indexed dataset-specific index.
    
    Parameters
    ----------
    data : list
        Trips data.
    mapping : dict
        Dictionary that contains respective counterpart (0-indexed ID) of each edge.
        
    Returns
    -------
    mapped_trips : list
        Trips with now remapped edges to their 0-index counterpart.
    """
    
    mapped_trips = [(idx, [mapping[e] for e in trip], timestamps) for (idx, trip, timestamps) in data]
    
    return mapped_trips


def condense_edges_in_trips_dataset(data: list) -> list:
    """
    Condense edges in trips dataset.
    
    Parameters
    ----------
    data : list
        Trips data.
        
    Returns
    -------
    condensed_edges_data : list
        Trips data, now with condensed edges.
    """
    
    condensed_edges_data = [(idx, condense_edges(t), timestamps) for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
    return condensed_edges_data


def load_data(args: Namespace, less: bool=False, sample: int=1000, file_name: str=TRAIN_TRIP_SMALL_FIXED_DATA_PICKLED_TIMESTAMPS_PATH) -> tuple:
    """
    Loads data from pickled file.
    
    Parameters
    ----------
    args : Namespace
        Command line arguments. Needed for checking for loop removals.
    less : bool (False)
        Used for creating smaller dataset (size: `sample`).
    sample : int (1000)
        Maximum data size. Only applicable if `less` is true.
    file_name : str
        Path to the pickled file to read data from.
        
    Returns
    -------
    data, edges_0indexed_mapping : tuple
        Tuple containing trips data and respective edges-to-0index-id mapping.
    """
    
    print(f"(LOAD START): ({file_name})") # Reading data is started
    
    input_file = open(file_name, "rb") # Reading file in binary form
    data = pickle.load(input_file)     # Load pickled file
    input_file.close()                 # Close the file after reading
    
    if less:
        data = data[:sample] # Clip the data to first `sample` samples
        
    data = condense_edges_in_trips_dataset(data) # Condense the edges
    edges_0indexed_mapping = fetch_map_fid_to_zero_indexed(data) # Get 0-indexes for each edge

    data = relabel_trips(data, edges_0indexed_mapping) # Relabel edges with respective 0-index
    
    print(f"(LOAD END): ({file_name})") # Reading data is done
    
    return data, edges_0indexed_mapping
    
    
def load_test_data(args: list, edges_0indexed_mapping: dict, less: bool=False, sample: int=1000, file_name: str=TEST_TRIP_DATA_PICKLED_TIMESTAMPS_PATH) -> list:
    """
    Loads test data, given 0-index edge mapping (calculated from loading training data).
    
    Parameters
    ----------
    args : list
        Command line arguments. TODO: Check whether staying unused.
    edges_0indexed_mapping : dict
        0-index edge mapping, calculated while loading training data (check `fetch_map_fid_to_zero_indexed`).
    less : bool (False)
        Used for creating smaller dataset (size: `simple`).
    sample : int (1000)
        Maximum data size. Only applicable if `less` is true.
    file_name : str
        Path to the pickled file to read test data from.
    
    Returns
    -------
    data : list
        Trips, now with mapped unknown nodes (to training dataset).
    """

    print(f"(LOAD TEST START): ({file_name})") # Reading data is started
    
    input_file = open(file_name, "rb")   # Reading file in binary form
    data       = pickle.load(input_file) # Load pickled file
    input_file.close()                   # Close the file after reading
    
    if less:
        data = data[:sample] # Clip the data to first `sample` samples
    
    data = condense_edges_in_trips_dataset(data) # Condense the edges
    original_len = len(data)
    print(f"Number of trips in test data (initially): {original_len}")
    
    
    precomputed_mapping_file_data_path  = file_name[:-4] + "_precomputed_data_mapping.pkl"
    precomputed_mapping_file_edges_path = file_name[:-4] + "_precomputed_edges_mapping.pkl"
    
    # Check if the precomputed file for this specific dataset already exists
    if os.path.exists(precomputed_mapping_file_data_path) and os.path.exists(precomputed_mapping_file_edges_path):
        data = pickle.load(open(precomputed_mapping_file_data_path, "rb"))
        edges_0indexed_mapping = pickle.load(open(precomputed_mapping_file_edges_path, "rb"))
        
        print("LOADED FROM PICKLE FILE SIZE: ", len(edges_0indexed_mapping))
        
        print(f"Data loaded from pickle files: {precomputed_mapping_file_data_path} || {precomputed_mapping_file_edges_path}")
            
    else:        
        # In new data, we may have nodes that we have not really seen before
        # Therefore, we don't even know their mapping to 0-index based ID
        # We wish to create a list of all trips that test data has, that have previously not seen nodes
        # Hence, when set of edges trip takes and has even one unknown node, we add it to the list
        trips_with_unseen_edges = [trip for trip in data if not set(trip[1]).issubset(edges_0indexed_mapping)] 
        for (_, edges, _) in trips_with_unseen_edges: 
            for e in edges: # For each edge
                if e not in edges_0indexed_mapping: # If the edge has not been previously seen (and, therefore, not mapped)
                    # Unknown edges get assigned index of length of mapping
                    # This is the simplest way that preserves the continuity (no "gaps" in integer representations)
                    edges_0indexed_mapping[e] = len(edges_0indexed_mapping) 
    
        print(f"Number of trips with unseen nodes: {len(trips_with_unseen_edges)}")
        print("Keeping these trips!")
        print("Relabelling trips with updated forward map")
        
        # Relabel trips with now updated mapping (accounting for unknown nodes)
        data = relabel_trips(data, edges_0indexed_mapping)

        # Write data to disk so no recalculation happens
        with open(precomputed_mapping_file_data_path, "wb") as output_file:
            pickle.dump(data, output_file)
            print(f"Data written to pickle file: {precomputed_mapping_file_data_path}")
        
        # Write data to disk so no recalculation happens
        with open(precomputed_mapping_file_edges_path, "wb") as output_file:
            pickle.dump(edges_0indexed_mapping, output_file)
            print(f"Data written to pickle file: {precomputed_mapping_file_edges_path}")

    print("BEFORE RETURNING LEN: ", len(edges_0indexed_mapping))
    print(f"(LOAD TEST END): ({file_name})") # Reading data is done
    
    return data, edges_0indexed_mapping

def create_node_neighbors(forward):
    start_nodes = defaultdict(set)
    
    for e in forward:
        u, v = edge_id_to_uv[e]
        start_nodes[u].add(forward[e])
    
    node_neighbors = {}	# here nodes are actually edges of the road network
    for e in forward:
        _,v = edge_id_to_uv[e]
        # print(e, forward[e])
        node_neighbors[forward[e]] = list(start_nodes[v])
        
    return node_neighbors

def neighbors_check(node_neighbors, data):
    """"
    Checks whether the neighbors properly formated 
    """
    for _, t, _ in tqdm(data, dynamic_ncols=True):
        for i in range(len(t) - 1):
            if t[i + 1] not in node_neighbors[t[i]]:
                print("AYO: ", node_neighbors[t[i]])
            
                assert t[i+1] in node_neighbors[t[i]], "How did this happen?"
    
    print("OK")


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    distances = {}

    for node in node_range:
        distances[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight="haversine")

    return distances


def merge_dicts(dictionaries):
    """
    Merges multiple dictionaries into a single uniform dictionary.
    
    Parameters
    ----------
    dictionaries : list
        List of dictionaries to be merged.
    
    Returns
    -------
    merged_dictionary : dict
        Union of all the dictionaries present within the `dictionaries` parameter.
    """
    
    merged_dictionary = {}

    for dictionary in dictionaries:
        merged_dictionary.update(dictionary)

    return merged_dictionary


def lipschitz_node_embeddings(edges_0indexed_mapping: dict, graph, k: int):
    """
    Creates lipshitz node embeddings.
    
    Parameters
    ----------
    edges_0indexed_mapping : dict
        Mapping of edges to 0-indexed ID.
    graph:
        Complete graph loaded from storage.
    k:
        Number of anchor nodes.    
    Returns
    -------
    embeddings : np.ndarray
        2-D array of dimension (number of nodes) x (number of anchor nodes).
        Each row represents a single node's embedding in latent space.
    """
    
    nodes          = list(edges_0indexed_mapping.keys()) # List of all nodes within the graph
    graph_reversed = graph.reverse(copy=True) # Reverse complete graph
    
    # Anchor nodes are chosen at random. 
    # We choose K of them
    anchor_nodes = random.sample(nodes, k) 
    
    # Computing path lengths using Dijkstra
    
    print("Starting Dijkstra")
    
    num_workers = 32                              # Number of worker processes to be used for parallel computation
    cutoff      = None                            # Calculate shortest path to all reachable nodes
    pool        = mp.Pool(processes=num_workers)  # Multiprocessing pool, containing `num_workers` processes
    results     = [
                    pool.apply_async(
                                    single_source_shortest_path_length_range,
                                    args=(graph_reversed, anchor_nodes[int(k / num_workers * i): int(k / num_workers * (i + 1))], cutoff)) for i in range(num_workers)
                  ]
    
    output = [p.get() for p in results] # Collect results from each pool
    distances = merge_dicts(output)     # Merge all the results into a single dictionary
    
    pool.close() # No more tasks will be ran
    pool.join()  # Wati for pools to finish and synchronize
    
    print("Finished Dijkstra")
    
    # Creating embedding matrix, dimensions: len(nodes) x `k`
    # `k` is the number of anchor nodes, which means we are creating embedding relative to those
    embeddings = np.zeros((len(nodes), k))
    
    for i, anchor_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        shortest_dist = distances[anchor_i]

        for j, node_j in enumerate(nodes):
            # Find shortest distance from `anchor_i` to `node_j` (every node)
            # Return -1 if node does not exist
            dist = shortest_dist.get(node_j, -1) 
            
            if dist != -1: # If node does exist
                # Place the distance found within the embedding representation
                embeddings[edges_0indexed_mapping[node_j], i] = 1 / (dist + 1) # TODO: Elaborate
    
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0) # TODO: Elaborate a bit. Just standardization with deviation of 1.
    
    return embeddings
