from collections import OrderedDict
import datetime
import pickle
import time

from data.argparser import parse_args
from data.paths import *
from data.data import *

from data.traffic import fetch_traffic_features_stored, find_interval_1, find_interval_2

import networkx as nx
from haversine import haversine

import torch
from torch import nn
import torch_geometric

from model.model import Model

args = parse_args() # Parse arguments from cmd
# print(args) # Show them off!

# Constants relevant for training
TRIPS_TO_EVALUATE = 100_000 
MAX_ITERS         = 300
BATCH             = args.batch_size
PRINT_FREQ        = 1000            # log data into terminal every 1000 batches
EVAL_FREQ         = 1               # args.eval_frequency # Go through validation set every ... epochs
JUMP              = 10_000


#if args.check_script:
#    # Evaluate smaller dataset for the purposes of checking script's funcitonality 
#    TRIPS_TO_EVALUATE = 10_000
#    PRINT_FREQ = 50

    
def save_model(model_path: str=MODEL_SAVE_PATH, aux_path: str=MODEL_SUPPORT_PATH):
    """
    Saves the model and auxiliary data.
    
    Parameters
    ----------
    model_path : str (MODEL_SAVE_PATH)
        Path where model ought to be stored.
        
    aux_path : str (MODEL_SUPPORT_PATH)
        Path where additional data ought to be stored.
    """
    
    global nodes_coords, edge_id_to_uv, edges_0indexed_mapping, model
    torch.save(model, model_path) # Write the model to disk
    
    aux_data_out_file = open(aux_path, "wb")
    # Write the auxiliary data to disk in .pkl format
    pickle.dump((edges_0indexed_mapping, nodes_coords, edge_id_to_uv), aux_data_out_file)
    aux_data_out_file.close()


def trip_length(trip):
    """
    Calculate trip length, given sequence of edges.
    
    Returns
    -------
        Path length.
    """
    global graph, index0_edges_mapping
    
    # Sum the length of eache in path
    return sum([graph[edge_id_to_uv[index0_edges_mapping[e]][0]][edge_id_to_uv[index0_edges_mapping[e]][1]][0]["length"] for e in trip])


def intersections_and_unions(path_1, path_2) -> tuple:
    """ 
    Calculates the number of unique nodes (from each path) in their intersection and union.    
    
    Returns
    -------
    intersection, union : tuple: 
        Aforementioned.
    """
    global graph, index0_edges_mapping
    
    # Only take into consideration unique edges
    path_1 = set(path_1)
    path_2 = set(path_2) 

    intersection = sum([graph[edge_id_to_uv[index0_edges_mapping[e]][0]][edge_id_to_uv[index0_edges_mapping[e]][1]][0]["length"] for e in path_1.intersection(path_2)])
    union        = sum([graph[edge_id_to_uv[index0_edges_mapping[e]][0]][edge_id_to_uv[index0_edges_mapping[e]][1]][0]["length"] for e in path_1.union(path_2)])

    return intersection, union

def dijkstra(true_trip):
    global args, transformed_graph, max_neighbors
    
    _, (src, *_, dest), (s, _) = true_trip # Extract sources and destinations
    print(true_trip)

    g = transformed_graph
    model.eval() # Enter evaluation mode
    
    with torch.no_grad(): # Disable gradient calculation
        current_nodes = [c for c in g.nodes()] # All nodes in the graph
        
        current  = [c    for c in current_nodes for _   in (node_neighbors[c] if c in node_neighbors else [])]
        pot_next = [nbr  for c in current_nodes for nbr in (node_neighbors[c] if c in node_neighbors else [])]
        dests    = [dest for c in current_nodes for _   in (node_neighbors[c] if c in node_neighbors else [])]
        
        traffic = None
        if args.traffic:
            traffic = [forward_interval_map[s] for c in current_nodes for _ in (node_neighbors[c] if c in node_neighbors else [])]

        unnormalized_confidence = model(current, dests, pot_next, traffic) # Calculate unnormalized confidence scores (prediction)
        # Calculate negative log-likelihoods from calculated confidences
        unnormalized_confidence = -1 * torch.nn.functional.log_softmax(unnormalized_confidence.reshape(-1, max_neighbors), 
                                                                       dim=1)
        transition_nll = unnormalized_confidence.detach().cpu().tolist() # Move tensor to CPU and convert to a standard list
    
    model.train()            # Enter training mode
    torch.cuda.empty_cache() # Empty CUDA cache
    
    count = 0 
    for u in g.nodes():
        for i, neighbor in enumerate(node_neighbors[u]):
            if neighbor == -1:
                break
            g[u][neighbor]["nll"] = transition_nll[count][i] # Assign the likeliness of traveling to node `neighbor` from `u`
        count += 1
    
    path =  nx.dijkstra_path(g, src, dest, weight="nll") # Find the shortest path from `src` to `dest`, in graph `g`
    path = [x for x in path] # Convert path to list
    
    return path

def shorten_path(path, true_dest):
    global edge_id_to_uv, index0_edges_mapping, nodes_coords
    
    dest_node = edge_id_to_uv[index0_edges_mapping[true_dest]][0]
    
    # Calculate the distance from end node of eache edge, to the destination node
    # Additionally, find the index of the edge with the shortest distance
    _, index = min([(haversine(nodes_coords[edge_id_to_uv[index0_edges_mapping[edge]][1]], nodes_coords[dest_node]), i) for i, edge in enumerate(path)])
    
    return path[:index+1]

def gen_paths_aux(all_paths):
    global model, node_nbrs, max_nbrs, edge_to_node_mapping
    global forward_interval_map
    
    if args.traffic:
        intervals = [forward_interval_map[(s)] for _ ,_ ,(s, _) in all_paths]

    true_paths = [p for _, p, _ in all_paths]
    model.eval() # Turn into evaluation mode
    gens = [[t[0]] for t in true_paths]
    
    pending = OrderedDict({i: None for i in range(len(all_paths))})
    
    with torch.no_grad(): # No gradient is going to be calculated here
        for _ in tqdm(range(MAX_ITERS), desc="[GEN_PATH_AUX] Genetring trips", dynamic_ncols=True):
            true_paths   = [all_paths[i][1] for i in pending]
            current_temp = [gens[i][-1] for i in pending]
            
            current  = [c     for c    in current_temp                  for _ in node_neighbors[c]]   # Current
            pot_next = [nbr   for c    in current_temp                  for nbr in node_neighbors[c]] # Neighbors from current 
            dests    = [t[-1] for c, t in zip(current_temp, true_paths) for _ in (node_neighbors[c] if c in node_neighbors else [])]
            
            traffic = None
            if args.traffic:
                traffic_chosen = [intervals[i] for i in pending] # Get all intervals from interval-0 to interval-pending
                traffic        = [t for c, t in zip(current_temp, traffic_chosen) 
                                    for _ in (node_neighbors[c] if c in node_neighbors else [])]
            
            unnormalized_confidence = model(current, dests, pot_next, traffic) # Generate predictions

            chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_neighbors), dim=1) # Get the index of most probable value
            chosen = chosen.detach().cpu().tolist() # Push to CPU and convert to list
            pending_trip_ids = list(pending.keys()) 
            
            for id, choice in zip(pending_trip_ids, chosen):
                choice = node_neighbors[gens[id][-1]][choice]
                if choice == -1:
                    del pending[id]
                    continue	
                    
                gens[id].append(choice)
                if choice == all_paths[id][1][-1]:
                    del pending[id] # Delete the key, easier for iteration
                    
            if len(pending) == 0: # If no more pending to be proccessed
                break
            
            torch.cuda.empty_cache() # Clear CUDA cache
    
    gens = [shorten_path(gen, true[1][-1]) if gen[-1]!=true[1][-1] else gen for gen, true in (zip(gens, all_paths))]
    model.train() # Shift again intro training mode
    
    return gens 

def gen_paths(all_paths):
    global JUMP
    ans = []
    
    for i in tqdm(list(range(0, len(all_paths), JUMP)), desc="batch_eval", dynamic_ncols=True):
        # Generate ae batch of patchs
        temp = all_paths[i : i + JUMP]
        ans.append(gen_paths_aux(temp))
    
    return [t for sublist in ans for t in sublist] # List flattening

def compare_with_dijkstra(generated, other_time = None):
    time_start = time.time() # Store the starting time
    
    dijkstra_output = [dijkstra(t) for (t, _) in tqdm(generated, desc="Dijkstra for generation", unit="trip", dynamic_ncols=True)]
    
    elapsed = time.time() - time_start
    with_greedy   = [(t[1], g) for t, g in generated if len(t) > 1]
    with_dijkstra = [(t[1],g) for (t,_),g in zip(generated, dijkstra_output) if len(t)>1]	
    
    reached_with_greedy   = [(t, g) for t, g        in with_greedy                     if t[-1] == g[-1]]
    reached_with_dijkstra = [a      for (a, (t, g)) in zip(with_dijkstra, with_greedy) if t[-1] == g[-1]]
    
    percent_reached = round(100*(len(reached_with_greedy)/len(generated)), 2)
    comparisons = {"all": "all queries", "reached": "only those queries where greedy reached"}
    cols = ["cyan", "cyan"]
    descriptions = ["", "({}% trips)".format(percent_reached)]
    to_compare = [(with_greedy, with_dijkstra), (reached_with_greedy,reached_with_dijkstra)]
    
    t_greedy = round(other_time, 2)                  # Generation time for greedy
    s_greedy = round(len(generated) / other_time, 2) # Generation speed for greedy
    
    t_dijkstra = round(elapsed, 2)                   # Generation time for Dijkstra
    s_dijkstra = round(len(generated) / elapsed, 2)  # Generation speed for Dijkstra
    
    print(f"\nGreedy vs Dijktsra")
    print(f"Comparing generation times - {t_greedy}s and {t_dijkstra}s for greedy and dijkstra to generate {len(generated)} trips")
    print(f"Comparing generation speeds - {s_greedy} trips/s and {s_dijkstra} trips/s for greedy and dijkstra")
    
    results = {}
    for comparison_type, _, desc, generation in zip(comparisons, cols, descriptions, to_compare):
        precisions = []
        recalls = []
        dst = []
        
        print()
        
        if comparison_type == "reached" and len(reached_with_greedy) == 0:
            print("No trip reached with greedy, so cannot run that comparison")
            return
         
        for gen in generation:
            lengths     = [(trip_length(t), trip_length(g)) for t, g in gen]
            inter_union = [intersections_and_unions(t, g) for t, g in gen]
            
            intersections = [intersection for intersection, _ in inter_union]
            # unions = [union for _, union in inter_union]
            
            lengths_gen = [l_g for l_t,l_g in lengths]
            lengths_true = [l_t for l_t,l_g in lengths]
            
            precs = [i / l if l > 0 else 0 for i, l in zip(intersections, lengths_gen) ]
            precision1 = round(100 * sum(precs) / len(precs), 2)
            precisions.append(precision1)
            
            recs = [i / l if l > 0 else 0 for i, l in zip(intersections, lengths_true) ]
            recall1 = round(100 * sum(recs) / len(recs), 2)
            recalls.append(recall1)
            
            deepst_accs = [i / max(l1, l2) for i, l1, l2 in zip(intersections, lengths_true, lengths_gen) if max(l1, l2) > 0]
            deepst = round(100 * sum(deepst_accs) / len(deepst_accs), 2)
            dst.append(deepst)
        
        if comparison_type == "reached":
            results["precision_reached"] = precisions[1]
            results["recall_reached"]    = recalls[1]
        
        else:
            results["precision_all"] = precisions[1]
            results["recall_all"]    = recalls[1]
            
        print(f"Comparing {comparisons[comparison_type]} {desc}")
        print(f"Precision:  Greedy & Dijkstra are {precisions}% and {precisions}%")
        print(f"Recall:     Greedy & Dijkstra are {recalls}% and {recalls}%\n")
        print(f"DeepST acc: Greedy & Dijkstra are {dst}% and {dst}%")
    
    print()

    return results


def evaluate(data, 
             num_of_trips: int=1000,
             with_dijkstra: bool=False) -> dict:

    """
    Evaluates.
    
    Parameters
    ----------
    data : _ 
    num_of_trips : int(1000)
    with_dijkstra : bool (False)
        Use Dijkstra's algorithm for comparison.
        
    Returns
    -------
    results : dict
        Self-explanatory: dictionary with relevant results.
    """
    
    # TODO -> Continue
    
    global nodes_coords, edge_id_to_uv, index0_edges_mapping 
    
    # List of results model will be compared upon
    categories_of_interest = ["precision", "recall", "reachability", "avg_reachability", \
                              "acc", "nll", "generated"]
    
    # Dictionary keeping the results
    # Results set to None in the beginning
    results = {category: None for category in categories_of_interest}
    
    print(f"[EVALUATION] Evaluating {num_of_trips} of trips")
    
    sample = random.sample(data, num_of_trips) # Sample `num_of_trips` trips at randpm
    time_start = time.time() # Get starting time of evaluation
    
    if with_dijkstra:
        gens = [dijkstra(t) for t in tqdm(sample, desc="Dijkstra for generation", unit="trip", dynamic_ncols=True)]
    
    else:
        gens = gen_paths(sample)
    
    time_elapsed = time.time() -time_start # Calculate the elapsed time
    results["time"] = time_elapsed         # Store the time elapsed in results
    
    preserved_with_stamps = sample.copy()  # Keep the copy of sample used
    sample = [e for _, e, _ in sample]     
    
    print("[EVALUATION] Without correction")
    
    generated = list(zip(sample, gens))
    generated = [(t, g) for t, g in generated if len(t) > 1] # Generated trip for each trip from evaluation sample
    
    lengths = [(trip_length(t), trip_length(g)) for (t, g) in generated]   # Get length of both trips
    lengths_gen  = [l_g for _, l_g in lengths] # Take the length of every generated path
    lengths_true = [l_t for l_t, _ in lengths] # Take the length of every true path
    
    
    inter_union = [intersections_and_unions(t, g) for (t, g) in generated] # Get the intersection and union size of each path and its respective generation 
    m = len(generated)
    # Isolate intersections and unions
    intersections = [intersection for intersection, _ in inter_union]
    unions        = [union        for _, union        in inter_union]

    # Calculate precision and round it to 2 decimal places
    precisions = [i / l if l > 0 else 0 for i, l in zip(intersections, lengths_gen) ]
    precision1 = round(100 * sum(precisions) / len(precisions), 2)

    # Calculate recalland round it to 2 decimal places
    recalls = [i / l if l > 0 else 0 for i, l in zip(intersections, lengths_true) ]
    recall1 = round(100 * sum(recalls) / len(recalls), 2)
    
    # Calculate DeepST accuracy
    deepst_accs = [i / max(l1, l2) for i, l1, l2 in zip(intersections, lengths_true, lengths_gen) if max(l1, l2)>0]
    deepst = round(100 * sum(deepst_accs) / len(deepst_accs), 2)
    
    num_reached = len([None for t, g in generated if t[-1] == g[-1]])
    lefts  = [haversine(nodes_coords[edge_id_to_uv[index0_edges_mapping[g[-1]]][0]], nodes_coords[edge_id_to_uv[index0_edges_mapping[t[-1]]][0]]) for t, g in generated]
    rights = [haversine(nodes_coords[edge_id_to_uv[index0_edges_mapping[g[-1]]][1]], nodes_coords[edge_id_to_uv[index0_edges_mapping[t[-1]]][1]]) for t, g in generated]
    
    reachability = [(left + right) / 2 for left, right in zip(lefts, rights)]
    all_reach = np.mean(reachability)      # Calculate standard mean value
    all_reach = round(1000 * all_reach, 2)
    
    if len(reachability) != num_reached:
        across_trips_not_reach = sum(reachability) / (len(reachability) - num_reached)
    else:
        across_trips_not_reach = 0

    across_trips_not_reach = round(1000 * across_trips_not_reach, 2)

    percent_reached = round(100*(num_reached/len(reachability)), 2)
    
    # Print results of model evaluation
    print()
    print(f"Precision is                            {precision1}%")
    print(f"Recall is                               {recall1}")
    print()
    print(f"% of trips reached is                   {percent_reached}%")
    print(f"Avg Reachability(across all trips) is   {all_reach}m")
    print(f"Avg Reach(across trips not reached) is  {across_trips_not_reach}m")
    print()
    print(f"Deepst's Accuracy metric is               {deepst}%")
    print()
    
    results["precision"]        = precision1
    results["reachability"]     = percent_reached
    results["avg_reachability"] = (all_reach, across_trips_not_reach)
    results["recall"]           = recall1
    results["generated"]        = list(zip(preserved_with_stamps, gens))
    
    return results
    

# Program start
if __name__ == "__main__":
    time_start = datetime.now()      # Current date and time
    print(f"Program started on {time_start}") # Log
    
    graph_file = open(PICKLED_GRAPH, "rb") # Graph data stored on disk
    graph = pickle.load(graph_file)        # Load graph data into a variable
    graph_file.close()                     # Always close the file!

    # Divide the length by 1000, for each edge
    for edge in graph.edges(data=True):
        # Edge is in format (node_start, node_end, edge_metadata)
        edge[2]["length"] = edge[2]["length"] / 1000
    
    
    # Load pickled train / validation / test data from disk
    train_data, edges_0indexed_mapping      = load_data(args=args,
                                                        file_name=TRAIN_TRIP_DATA_PICKLED_TIMESTAMPS_PATH)
    
    validation_data, edges_0indexed_mapping = load_test_data(args, 
                                                             edges_0indexed_mapping, 
                                                             file_name=VALIDATION_TRIP_DATA_PICKLED_TIMESTAMPS_PATH)
    
    test_data, edges_0indexed_mapping       = load_test_data(args, 
                                                             edges_0indexed_mapping, 
                                                             file_name=TEST_TRIP_DATA_PICKLED_TIMESTAMPS_PATH)
    
    test_data_fixed, edges_0indexed_mapping = load_test_data(args, 
                                                             edges_0indexed_mapping, 
                                                             file_name=TEST_TRIP_SMALL_FIXED_DATA_PICKLED_TIMESTAMPS_PATH)

    
    # print(len(edges_0indexed_mapping))
    
    index0_edges_mapping = {edges_0indexed_mapping[k]: k for k in edges_0indexed_mapping}
    node_neighbors = create_node_neighbors(edges_0indexed_mapping) # Neighbors of nodes in directed graph

    # Checking to make sure graph data is properly structured
    # Not mandatory - is fast, though
    neighbors_check(node_neighbors, train_data)
    neighbors_check(node_neighbors, validation_data)
    neighbors_check(node_neighbors, test_data)
   
    transformed_graph = nx.DiGraph() # Directed graph
    # Add edges from node to node
    for e1 in node_neighbors:
        for e2 in node_neighbors[e1]:
            if e2 != -1:
                transformed_graph.add_edge(e1, e2)
                # print(f"{e1} -> {e2}")
    
    # device = torch.device("cuda:{}".format(args.gpu_index) if ((not args.force_cpu) and torch.cuda.is_available()) else "cpu")
    device = "cuda"
    # print(f"Running main.py on {device}")
    
    # Traffic matrix formation
    traffic_matrix = None
    if args.traffic:
        find_interval         = find_interval_1 if not args.ignore_day else find_interval_2 
        traffic_feature_store = fetch_traffic_features_stored(device=device, find_interval=find_interval)
        
        all_intervals = list(traffic_feature_store.keys()) # List of all intervals in dataset
        all_intervals.sort()                               # Sort the intervals
        
        forward_interval_map  = {interval: index for index, interval in enumerate(all_intervals)}
        backward_interval_map = all_intervals
        
        traffic_matrix = torch.empty((len(all_intervals), 10))
        for idx, interval in enumerate(all_intervals):
            traffic_matrix[idx] = traffic_feature_store[interval]
        
        traffic_matrix = traffic_matrix.float().to(device) # Port trafic matrix to CUDA, ideally
        
        # Transforming complete data
        train_data      = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in tqdm(train_data, desc="marking intervals")]
        validation_data = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in tqdm(validation_data)]
        test_data       = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in tqdm(test_data)]
        test_data_fixed = [(i, t, (find_interval(s), e)) for (i, t, (s, e)) in tqdm(test_data_fixed)]
        
    
    nodes_used = set()
    for e in edges_0indexed_mapping:
        u, v = edge_id_to_uv[e] # Extract endpoints of the edge
        
        # Add them both into the list of nodes used
        nodes_used.add(u)
        nodes_used.add(v) 
    
    nodes_used = list(nodes_used) # List of unique nodes used		
    nodes_forward = {node: idx for idx, node in enumerate(nodes_used)} # Pair up nodes with their respective indices

    # Map: [from] zero-indexed edges ; [to] zero-indexed nodes
    edge_to_node_mapping = {edges_0indexed_mapping[edge]: (nodes_forward[edge_id_to_uv[edge][0]], nodes_forward[edge_id_to_uv[edge][1]]) 
                           for edge in edges_0indexed_mapping}
    
    edge_to_node_mapping[-1] = (-1, -1) # In case of -1 edges, they map to (-1, -1)
    
    # Embeddings initialization
    embeddings = None
    if args.initialize_embeddings_lipschitz:
        # Create Lipschitz node embeddings for each node, relative to anchor nodes
        embeddings = lipschitz_node_embeddings(edges_0indexed_mapping,
                                               graph,
                                               args.embedding_size)
        
        map_node_zero_indexed_to_coords = {nodes_forward[n]: nodes_coords[n] for n in nodes_forward}


    # Graph neural network (module of the complete network) initialization
    if args.gnn is not None:
        # Convert node embeddings into torch.Tensor
        node_embeddings = torch.from_numpy(embeddings).float() if embeddings is not None else None
        node_feats = node_embeddings
        edge_index = []
        
        for u, v in edge_id_to_uv:
            if u in nodes_forward and v in nodes_forward:
                u = nodes_forward[u] # Start node
                v = nodes_forward[v] # End node
                
                edge_index.append((u, v)) # start -> end pair
        
        # Convert graph created in loop into torch.Tensor
        edge_index = torch.LongTensor(edge_index).T
        torch_graph = torch_geometric.data.Data(x=node_feats, edge_index=edge_index) 
        torch_graph = torch_graph.to(device) # Port `torch_graph` to CUDA, ideally
        
        
        # IMPORTANT PART: Model created!
        model = Model(
                       num_nodes=len(nodes_forward), 
                       graph=torch_graph, 
                       device=device, 
                       args=args, 
                       embeddings=node_embeddings, 
                       mapping=edge_to_node_mapping,
                       traffic_matrix=traffic_matrix
                    ).to(device) # ... and ported to CUDA, ideally :)
        
    else: # Create model without GNN part
        model = Model(
                       num_nodes=len(nodes_forward),
                       device=device, 
                       args=args,
                       embeddings=(None if embeddings is None else torch.from_numpy(embeddings)), 
                       mapping=edge_to_node_mapping,
                       traffic_matrix=traffic_matrix
                 ).to(device)
        
    # Node with maximum neighbors
    max_neighbors = max(len(neighbors_array) for neighbors_array in node_neighbors.values())

    num_nodes = len(edges_0indexed_mapping) # Number of nodes in the graph    
    # Fix the node neighbors dataset, by appending -1 distance to standardize the second dimension
    for u in range(num_nodes):
        if u in node_neighbors:
            node_neighbors[u].extend([-1] * (max_neighbors - len(node_neighbors[u])))
        
        else:
            node_neighbors[u] = [-1] * max_neighbors

    # print(f"[NODE_NEIGHBORS] {node_neighbors}")
    
    # Training setup
        
    cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")                  
    sigmoid_activation = nn.Sigmoid()                                          
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)    
    
    loss_curve      = [] # loss curve over total training 
    train_acc_curve = [] # Train accuracy
    test_acc_curve  = [] # Test accuracy
    max_path_len = 1 + max(len(travel) for _, travel, _ in train_data) # Longest travel in training dataset
    
    # Data gathered per `PRINT_FREQ` batches
    total_loss  = 0 
    total_trajs = 0 
    preds       = 0 
    correct     = 0 
    prob_sum    = 0 
    
    level = 0
    validation_evals_till_now_reachability = []
    validation_evals_till_now_precision    = []
    validation_evals_till_now_recall       = []
    
    # Training - START
    
    for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epochs", dynamic_ncols=True):
        random.shuffle(train_data) # Shuffle the training data 
        model.train()              # Enter training mode
        
        for batch_num, k in tqdm(list(enumerate((range(0, len(train_data), BATCH)))), 
                                 desc="Batch", 
                                 unit="steps", 
                                 leave=True, 
                                 dynamic_ncols=True
                                ):
            # Batch - START 
            
            train_sample = random.sample(train_data, BATCH) # Travels can reoccur
            valid_trajs  = len(train_sample) # Number of travels 
            
            next_node    = [neighbor   for _, travel, _ in train_sample for i in range(len(travel) - 1) for neighbor in node_neighbors[travel[i]]]
            current      = [travel[i]  for _, travel, _ in train_sample for i in range(len(travel) - 1) for _        in node_neighbors[travel[i]]]
            destinations = [travel[-1] for _, travel, _ in train_sample for i in range(len(travel) - 1) for _        in node_neighbors[travel[i]]]
            
            # print(f"[NEXT_NODE] {next_node}")
            # print(f"[CURRENT] {current}")
            # print(f"[DESTINATIONS] {destinations}")
            
            traffic = None
            if args.traffic:
                traffic = [forward_interval_map[(s)] 
                           for _, t, (s, _) in train_sample 
                           for i in range(len(t) - 1) 
                           for neighbor in node_neighbors[t[i]]]

                # print(f"[TRAFFIC] {traffic}") 
                # print(f"[TRAFFIC] Shape: {traffic.shape}")
            
            # PERFORM FORWARD PASS
            unnormalized_dist = model(current, destinations, next_node, traffic) 

            # Total length of travel in current batch
            num_preds = sum(len(travel) - 1 for _, travel, _ in train_sample) 	

            true_nbr_class = torch.LongTensor([
                                               (node_neighbors[travel[i]].index(travel[i + 1])) 
                                               for _, travel, _ in train_sample 
                                               for i in range(len(travel) - 1)]
                                              ).to(device) # Port 
            
            # Compute Cross entropy loss
            loss = cross_entropy_loss(unnormalized_dist.reshape(-1, max_neighbors), 
                                      true_nbr_class.to(device))
            
            preds += num_preds # Increase number of predictions 
            preds_in_this_iteration = num_preds
            total_loss += loss.item()  # Increase total loss 
            total_trajs += valid_trajs # Increase total valid trajectories
            
            if valid_trajs > 0:
                if (batch_num + 1) % PRINT_FREQ == 0: # Print ever PRINT_FREQ batches
                    # Log current batch status
                    tqdm.write(f"Epoch: {epoch}, Batch: {batch_num+1}, loss({args.loss}) - per trip: {round(total_loss/total_trajs, 2)}, per pred: {round(total_loss/preds, 3)}")
                    
                    loss_curve.append(total_loss / total_trajs) # Store the loss for this batch
                   
                    # Reset the counters for next `PRINT_FREQ` batches
                    # Therefore, these values are stored per `PRINT_FREQ` batches 
                    total_loss  = 0
                    total_trajs = 0
                    preds       = 0
                    correct     = 0
                    prob_sum    = 0

                loss /= valid_trajs
                
                optimizer.zero_grad()    # Zero out gradients
                loss.backward()          # Backpropagate through network
                optimizer.step()         # Adjust the parameters 
                torch.cuda.empty_cache() # Flush the cache
        
            # Batch - END
            
        # Epoch - END
        
        # Validation - START
        if (epoch + 1) % args.eval_frequency == 0: 
            save_model()
            # Training results
            tqdm.write("\n[EVALUATION] Doing a partial evaluation on train set")
            tqdm.write("\nStandard")
            train_results = evaluate(data = train_data, 
                                     num_of_trips = min(TRIPS_TO_EVALUATE, len(train_data)),
                                     with_dijkstra=False)
            
            # Validation results
            tqdm.write("\[EVALUATION] Evaluation on the validation set (size = {})".format(len(validation_data)))
            tqdm.write("\nStandard")
            validation_results = evaluate(data=validation_data, 
                                          num_of_trips=len(validation_data),
                                          with_dijkstra=False)
            
            # Validation with dijkstra
            if args.with_dijkstra:
                tqdm.write(f"\n[EVALUATION] Evaluation on the validation set (size = {len(validation_data)})")
                tqdm.write(f"\nDIJKSTRA")			
                validation_results = evaluate(data=validation_data,
                                             num_of_trips=len(validation_data),
                                             with_dijkstra=True)

            # Store the validation statistics
            validation_evals_till_now_reachability.append(validation_results["reachability"])
            validation_evals_till_now_precision.append(validation_results["precision"])
            validation_evals_till_now_recall.append(validation_results["recall"])

            # Log validation statistics
            tqdm.write(f"[VALIDATION] Validation Reachability for the previous evals: {validation_evals_till_now_reachability}")
            tqdm.write(f"[VALIDATION] Validation Precision for the previous evals:    {validation_evals_till_now_precision}")
            tqdm.write(f"[VALIDATION] Validation Recall for the previous evals:       {validation_evals_till_now_recall}")
            
        # Validation - END
    
    # Training - END
    print(f"[TRAINING] Finished after {epoch + 1} epochs")
    
    # Test - BEGIN 

    print(f"[TEST] Evaluation on test results!!!")
    
    test_results = evaluate(data=test_data, 
                            num_of_trips=len(test_data),
                            with_dijkstra=False)

    # If model ought to be tested with Dijkstra, then perform checking with it
    if args.end_dijkstra or args.with_dijkstra:
        num_dijkstra = 100 if args.check_script else len(test_data_fixed)
        tqdm.write("(Partial) fixed test")
        test_results_small = evaluate(data = test_data_fixed, 
                                      num_of_trips = num_dijkstra,
                                      with_dijkstra = False)
        tqdm.write("Taking Dijkstra's help, comparing on fixed test")
        dijkstra_results = compare_with_dijkstra(generated=test_results_small["generated"], other_time=test_results_small["time"])
    
    
    # print("The script that was run here was - \n{}{}".format("python -i "," ".join(sys.argv)))
    if args.result_file is None:
        result_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Write data to disk
        
    result_file = open(result_file, "w")
    result_file.write("precision, recall, %age reach, avg_reach_all, avg_reach_specific, epochs_to_train, ")
    result_file.write("dijkstra_precision_reached, dijkstra_precision_all, ")
    result_file.write("dijkstra_recall_reached, dijkstra_recall_all\n")
    
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    
    result_file.write(f'{test_results["precision"]}, {test_results["recall"]}, {test_results["reachability"]}, {test_results["avg_reachability"][0]}, {test_results["avg_reachability"][1]}, {epoch + 1}, \
             {dijkstra_results["precision_reached"]}, {dijkstra_results["precision_all"]} {dijkstra_results["recall_reached"]} {dijkstra_results["recall_all"]}\n')
    
    result_file.write("[VALIDATION] Reachabality scores - \n")
    result_file.write(", ".join([str(x) for x in (validation_evals_till_now_reachability)]) + "\n")
    result_file.close()
    
    # Finished! :)