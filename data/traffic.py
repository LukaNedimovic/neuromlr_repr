from .paths import *
import pickle           # Loading and writing data

import geopandas as gpd # Geospatial data processing
from tqdm import tqdm   # Loading progress visualization
from haversine import haversine # Distance calculation 

from datetime import datetime, timezone 

from typing import Callable

import numpy as np                    # Basic linear algebra
import torch                          # Tensor creation, at the end
from sklearn.decomposition import PCA # Dimensionality reduction
from sklearn.manifold import TSNE     # Dimensionality reduction

import matplotlib.pyplot as plt       # Plotting


if args.check_script:
    DATA_THRESHOLD = 10
else:
    DATA_THRESHOLD = 200

# Read data from files
df_edges = gpd.read_file(EDGE_DATA)
df_nodes = gpd.read_file(NODE_DATA)

col_osmid = df_nodes["osmid"].to_numpy()    # column  "osmid"
cols_yx   = df_nodes[["y", "x"]].to_numpy() # columns "y" and "x"

# Creates dictionary, where key is the edge (osmid of junction), and value is a pair of (y, x) coordinates
nodes_coords = {osmid: (y, x) for osmid, (y, x) in zip(col_osmid, cols_yx)}

edge_id_to_uv = df_edges[["u", "v"]].to_numpy() # columns "u" and "v"
num_edges = edge_id_to_uv.shape[0]              # Counts the number of edges present in the graph


def find_interval_1(timestamp: int) -> tuple:
    """
    Calculates the date, given `timestamp` (POSIX).
    
    Parameters
    ----------
    timestamp : int
        POSIX timestamp of certain moment in dataset.
        
    Returns
    -------
    _ : tuple
        Tuple containing:
            (1) Date, respective to POSIX timestamp ("yy-mm-dd" format)  [str]
            (2) Hour, respective to POSIX timestamp                      [int] 
    """
    
    datetime_from_timestamp = datetime.fromtimestamp(timestamp, timezone.utc)
    date = str(datetime_from_timestamp).split()[0]
    hour = datetime_from_timestamp.hour
    
    return (date, hour)


def find_interval_2(timestamp):
    """
    Calculates the date and time given `timestamp` (POSIX).
    
    Parameters:
    timestamp : int
        POSIX timestamp of certain moment in dataset.
        
    Returns
    -------
    (-1, _) : tuple
        Tuple containing:
            (1) -1            [int]
            (2) Date and time [datetime]
    """
    
    return (-1, datetime.fromtimestamp(timestamp, timezone.utc).hour)


# Calculate haversine distance for each in the graph
haversine_distances = {}
for edge_index in tqdm(range(num_edges)):
    # Edge u -> v
    u, v = edge_id_to_uv[edge_index] # Get `u` and `v`'s that make up the edge
    coords_u = nodes_coords[u] # Get latitude and longitude of `u`
    coords_v = nodes_coords[v] # Get latitude and longitude of `v`
    
    haversine_distances[edge_index] = haversine(coords_u, coords_v, unit="km") # Store the distance for that particular edge


def get_traffic_features(file_name: str=None, num_days: int=1, train: bool=False, num_components: int=10, device: str="cpu", find_interval: Callable=find_interval_1) -> int:  
    print(f"(LOAD START): ({file_name})") # Reading data is started
    
    input_file = open(file_name, "rb")   # Reading file in binary form
    data       = pickle.load(input_file) # Load pickled file
    input_file.close()                   # Close the file after reading
    
    trip_lens   = {} # Length of each trip
    trip_speeds = {} # Speed of each trip (km/hr used)
    counts      = {} # Counting traversal of edge `e` on interval `i`, as in counts[(e, i)] TODO
    speed_sums  = {} # Sum of speeds on edge `e` during interval `i`, as in speed_sums[(e, i)] ??? TODO

    tot_sum   = 0 # Total sum of speed in whole dataset
    tot_count = 0 # Total count of edges traversed in whole dataset

    intervals = set()
    for id, trip, (time_start, time_end) in tqdm(data):
        # print(id, trip, (time_start, time_end))
    
        if args.dataset == "cityindia": # Case-specific resizing
            time_start /= 1000 # the units were ms
            time_end   /= 1000   # the units were ms
        
        if time_start == 0 or time_end == 0: # Timestamps need to be available for feature extraction
            print("Timestamps missing. Can't calculate. Try running without traffic.")
            raise SystemExit
        
        time = (time_end - time_start) / 3600 # Calculate the length of the trip (in seconds)
            
        if time == 0: # This is only if time_start == time_end
            print("Start and end times are the same. Not useful for feature extraction.")
            continue
            
        # Make sure data is constructed properly
        assert len(trip) >= 1, "Length of each trip must at least be one edge long. If not, trips are not filtered properly."
        
        # Length of trip is the sum of all haversine distances (physical distances on globe)
        # between two nodes travelen among (i.e. length of the respective edge)
        # We have them precalculated in the array
        trip_length   = sum([haversine_distances[edge] for edge in trip])
        trip_lens[id] = trip_length
        
        # Average speed is just length of trip divided by time spent
        speed = trip_length / time
        trip_speeds[id] = speed

        # Interval is characterized by date and specific hour TODO: CHECK, MYB INTERVAL IS JUST HOURS
        interval = find_interval(time_start)
        intervals.add(interval)
        
        # Calculate, for each edge, on each day, at each hour, total number of tours includding it and the sum of speeds
        # Sum of speeds will be used in the future for average speed calculation
        for edge in trip:
            tup = (edge, interval)
            # If respective edge has not previously been traversed at given date and hour 
            if tup not in counts:  
                counts[tup] = 0     # Add it into the keys of both counts 
                speed_sums[tup] = 0 # and speed_sums

            counts[tup] += 1          # Increment traversal by one
            speed_sums[tup] += speed  # Increment sum of speeds by speed
            
            # Another edge traversed, therefore counter is incremented, and speed accumulated
            tot_count += 1    
            tot_sum += speed 


    max_day = max(d for (d, _) in intervals) # Get the latest date present in dataset
    min_day = min(d for (d, _) in intervals) # Get the earliest date present in dataset

    # print("Assumption that no date coincides, unique dates")
    all_days = sorted(list(set((d for (d, _) in intervals)))) # All the unique days
    
    # print(f"[all_days]: {all_days}")
    
    print(f"[min_day]: {min_day} ;; [max_day]: {max_day}")
    all_intervals = [(d, hr) for hr in range(24) for d in all_days]

    avg_speed = tot_sum / tot_count # Total average speed in the dataset
    print(f"[avg_speed] across the network (dataset) was found to be {round(avg_speed,2)} km/hr")

    if not train:
        pca, chosen_edges, avg_speed = pickle.load(open(LEARNT_FEATURE_REP, "rb"))
        num_components = pca.n_components


    print(len(all_intervals))
    
    speeds = {}
    passed_threshold = set()
    for e in range(num_edges):
        for interval in all_intervals:
            k = (e, interval)
            # print(f"[k]: {k}")
            # If edge `e` has been traversed enough, take data from such edge into consideration
            if k in counts and counts[k] >= DATA_THRESHOLD:
                speeds[k] = speed_sums[k] / counts[k] # Calculate the average speed for given edge throughout the interval
                passed_threshold.add(e)
    
            else:
                if k in counts:
                    speeds[k] = (speed_sums[k] + (DATA_THRESHOLD - counts[k]) * avg_speed) / DATA_THRESHOLD
                else:
                    speeds[k] = avg_speed # Not enough data - set it to general average speed

    if train:
        chosen_edges = passed_threshold # Chosen are only the edges that pass the lower bound

    assert len(chosen_edges) > 0, "No data to work with - not a single edge passes the threshold!"

    print(f"Number of chosen edges is {len(chosen_edges)}")

    data_array   = np.empty((len(chosen_edges), len(all_intervals)), dtype=np.float32)
    chosen_edges = list(chosen_edges) # Convert it to list, from set 

    for idx, edge in enumerate(chosen_edges):
        # Create a vector of average speed for each edge and for each interval throughout the day
        average_speed_on_edge_for_each_interval = np.array([speeds[(edge, interval)] for interval in all_intervals], dtype=np.float32)
    
        # Fill the given position in numpy array with aforementionedvector
        data_array[idx] = average_speed_on_edge_for_each_interval
        
    data_array = data_array.T # Transpose the array. New shape: len(all_intervals) x len(chosen_edges)

    print(data_array)
    print(data_array.shape)

    if train:
        # Perform Principal Component Analysis
        pca = PCA(n_components=num_components)
        pca.fit(data_array)
        
        # print(pca.explained_variance_ratio_)
        print(f"Retained {round(sum(pca.explained_variance_ratio_*100), 2)}% variance by creating just {num_components} out of the original {data_array.shape[1]} features")
        
        # Write data to disk
        pickle.dump((pca, chosen_edges, avg_speed), open(LEARNT_FEATURE_REP, "wb"))
            
    # Vertically stack array of average speeds and general average speeds, and apply PCA on them
    reduced = pca.transform(np.vstack((np.array([avg_speed for _ in range(data_array.shape[1])]), data_array)))
        
    # Return dictionary of tensors, where key is the interval, and the value reduced representation 
    return {interval: torch.from_numpy(reduced[idx]).float().to(device).reshape(1, -1) for idx, interval in enumerate(all_intervals)}

def fetch_traffic_features_stored(train_file=TRAIN_TRIP_DATA_PICKLED_TIMESTAMPS_PATH,
                                  test_file=TEST_TRIP_DATA_PICKLED_TIMESTAMPS_PATH, 
                                  device="cpu",
                                  find_interval=find_interval_1) -> dict:

    # Get traffic features from training data
    train_data = get_traffic_features(file_name=TRAIN_TRIP_DATA_PICKLED_TIMESTAMPS_PATH,
                                      train=True, 
                                      device=device, 
                                      find_interval=find_interval)

    # Get traffic features from validation data
    validation_data = get_traffic_features(file_name=VALIDATION_TRIP_DATA_PICKLED_TIMESTAMPS_PATH, 
                                           train=False, 
                                           device=device, 
                                           find_interval=find_interval)

    # Get traffic features from test data
    test_data = get_traffic_features(file_name=TEST_TRIP_DATA_PICKLED_TIMESTAMPS_PATH,
                                     train=False, 
                                     device=device, 
                                     find_interval=find_interval)
    
    #print("KEYS:",train_data.keys())
    
    # Make sure intervals are meaningful
    #try:
    #    assert len(set(train_data.keys()).intersection(set(validation_data.keys()))) == 0, "problem with intervals"
    #    assert len(set(train_data.keys()).intersection(set(test_data.keys()))) == 0,       "problem with intervals"
    #    
    #except Exception as e:
    #    print(e)

    train_data.update(validation_data)
    train_data.update(test_data)
        
    return train_data

if __name__ == "__main__":
    store = fetch_traffic_features_stored(device="cpu", find_interval=find_interval_2)
    
    all_intervals = list(store.keys())
    all_intervals.sort()
    
    forward_interval_map = {interval: index for index, interval in enumerate(all_intervals)}
    backward_interval_map = all_intervals
    
    traffic_matrix = torch.empty((len(all_intervals), 10))
    for idx, interval in enumerate(all_intervals):
        print(idx, interval, store[interval], end="\n\n")
        traffic_matrix[idx] = store[interval] 
    
    traffic_matrix = traffic_matrix.float().numpy()
    
    perplexity = 10
    X_embedded = TSNE(n_components=2, perplexity = perplexity).fit_transform(traffic_matrix)
    
    plt.clf()
    plt.scatter(X_embedded[:,0], X_embedded[:,1])
    
    for i, interval in enumerate(all_intervals):
        plt.annotate(str(interval), (X_embedded[i][0], X_embedded[i][1]))


    plt.show()