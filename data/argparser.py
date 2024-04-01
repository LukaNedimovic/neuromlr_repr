from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    """
    Parses the command line passed arguments.
    Useful for environment creation and flexibility reasons.
    
    Example: Creating scripts with various hyperparameters to  experimentally.
    
    Returns
    -------
    args : Namespace
        Parsed arguments in the form of Namespace object.
        To be used in main.setup().
    """
    
    parser = ArgumentParser()
     
    # Environment Configuration
    
    parser.add_argument("-cpu", action="store_true",  dest="force_cpu", default=False)
    
    parser.add_argument("-eval_frequency", type=int,     default=100, dest="eval_frequency", help="evaluate afetr every these many epochs")
    parser.add_argument("-loss",           default="v2", type=str,  dest="loss")    
    parser.add_argument("-epochs", type=int, default=4, dest = "epochs", help="number of epochs")
    parser.add_argument("-gpu_index", type=int, default=0, dest = "gpu_index", help="which gpu")
    parser.add_argument("-batch_size", type=int, default=32, dest = "batch_size", help="Stochastic Gradient Descent mei batch ka size")
    parser.add_argument("-lr", type=float, default=0.001, dest = "lr", help="Learning Rate")
    parser.add_argument("-weight", type=float, default=1, dest = "weight", help="weight for one hot vectors")

    parser.add_argument("-check_script", action="store_true", dest="check_script", default=False)
 
    parser.add_argument("-percent_data", type=float, default=None, dest = "percent_data", help="percentage of training data to use")
    parser.add_argument("-result_file", default = None, type = str, dest = "result_file")    
    
    # MODEL Configuration
    
    parser.add_argument("-embedding_size", type=int, default=128, dest = "embedding_size")
    parser.add_argument("-hidden_size", type=int, default=256, dest = "hidden_size")
    parser.add_argument("-num_layers", type=int, default=3, dest = "num_layers")
    
    parser.add_argument("-fixed_embeddings", action="store_false", dest="trainable_embeddings", default=True)
    parser.add_argument("-lipschitz", action="store_true",  dest="initialize_embeddings_lipschitz", default=False)

    parser.add_argument("-dataset", default="beijing",    type=str, dest="dataset")
    parser.add_argument("-traffic", action="store_true", dest="traffic", default=False)

    parser.add_argument("-ignore_day",          action="store_true",  dest="ignore_day", default=False)
    
    parser.add_argument("-gnn",              default=None,         type=str, dest="gnn")
    parser.add_argument("-gnn_layers",     type=int,     default=1, dest="gnn_layers",     help="number of layers in the GCN")
    parser.add_argument("-attention", action="store_true", dest="attention", default=False)
    
    parser.add_argument("-num_heads", type=int, default=2, dest = "num_heads")

    parser.add_argument("-end_destinations", action="store_false", dest="intermediate_destinations")
    parser.add_argument("-ignore_unknown_args", action="store_true", dest="ignore_unknown_args", default=False)
    parser.add_argument("-with_dijkstra", action="store_true", dest="with_dijkstra", default=False)

    parser.add_argument("-end_dijkstra", action="store_true", dest="end_dijkstra", default=True)
    parser.add_argument("-no_end_dijkstra", action="store_false", dest="end_dijkstra")
    
    parser.add_argument("-model_path", default="", type=str, dest="model_path")
    
    
    args, unknown_args = parser.parse_known_args() # Parse arguments
    
    # If not ignoring unknown arguments passed, assert that there may not be any unknown arguments
    if not args.ignore_unknown_args:
        assert len(unknown_args) == 0, f"Unrecognized arguments passed: {unknown_args}"
    
    
    return args