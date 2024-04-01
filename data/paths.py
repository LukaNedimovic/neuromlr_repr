"""
This file contains path constants relevant for program's execution. Paths are related to already present datasets.
Don't forget to download data through the link provided.
File to be imported. 
"""

import os
from dotenv import load_dotenv
from .argparser import parse_args

args = parse_args()

load_dotenv()

# Path to the desired dataset, e.g. {DATASETS_PATH}/beijing_data/
DATASET_PATH = os.environ.get("DATASETS_PATH") + f"{args.dataset}_data/"

# Make sure the path has been set properly
assert len(DATASET_PATH) > 0, "Datasets path is not properly set. (length == 0)"

NODE_DATA = DATASET_PATH + "map/nodes.shp"
EDGE_DATA = DATASET_PATH + "map/edges.shp"

TRAIN_TRIP_DATA_PICKLED_TIMESTAMPS_PATH      = DATASET_PATH + "preprocessed_train_trips_all.pkl"
VALIDATION_TRIP_DATA_PICKLED_TIMESTAMPS_PATH = DATASET_PATH + "preprocessed_validation_trips_all.pkl"
TEST_TRIP_DATA_PICKLED_TIMESTAMPS_PATH       = DATASET_PATH + "preprocessed_test_trips_all.pkl"

TRAIN_TRIP_SMALL_FIXED_DATA_PICKLED_TIMESTAMPS_PATH      = DATASET_PATH + "preprocessed_train_trips_small.pkl"
VALIDATION_TRIP_SMALL_FIXED_DATA_PICKLED_TIMESTAMPS_PATH = DATASET_PATH + "preprocessed_validation_trips_small.pkl"
TEST_TRIP_SMALL_FIXED_DATA_PICKLED_TIMESTAMPS_PATH       = DATASET_PATH + "preprocessed_test_trips_small.pkl"

PICKLED_GRAPH               = DATASET_PATH + "map/graph_with_haversine.pkl"
LEARNT_FEATURE_REP          = DATASET_PATH + "pickled_pca_information_for_traffic_representation.pkl"
CRUCIAL_PAIRS               = DATASET_PATH + "crucial_pairs.pkl"
CACHED_DATA_FILE            = DATASET_PATH + "cached_data.pkl"
INITIALIZED_EMBEDDINGS_PATH = "results/" 

# model_path = args.model_path if args.model_path != '' else PREFIX_PATH + 'pretrained_models\\'
model_path = DATASET_PATH + "pretrained_models/"

from datetime import datetime

MODEL_SAVE_PATH    = model_path + f"{datetime.now()}_model.pt"
MODEL_SUPPORT_PATH = model_path + f"{datetime.now()}_model_support.pkl"