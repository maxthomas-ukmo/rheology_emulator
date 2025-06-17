import argparse 
import logging
import shutil
import os
import yaml

from datetime import datetime

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--get_data', action='store_true', help="Enable data retrieval mode")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    parser.add_argument('--evaluate', action='store_true', help="Enable evaluation mode")

    # Optional arguments for raw and processed data retrieval
    parser.add_argument('--inputs', nargs='*', default=['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr'], help="List of potential input variables to retrieve from raw data")
    parser.add_argument('--raw_path', type=str, required=False, help="Path to the raw data file, for use with --get_data")
    parser.add_argument('--interim_path', type=str, required=False, help="Path to the intermediate data, for use with --get_data")
    parser.add_argument('--features', nargs='*', default=['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr'], required=False, help="Features for the processed data")
    parser.add_argument('--labels', nargs='*', default=['sivelv'], required=False, help="Labels for the processed data")
    parser.add_argument('--subset_region', type=str, required=False, default='Arctic', help="Region to subset the data to, default is 'Arctic'")


    # Run configuration file
    parser.add_argument('--training_cfg', type=str, help="Name of the training configuration file to load")

    # Optional arguments for training
    parser.add_argument('--pairs_path', type=str, required=False, help="Path to the pairs data file for training")
    parser.add_argument('--results_path', type=str, required=False, help="Path to where model results will be saved")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--val_fraction', type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument('--test_fraction', type=float, default=0.1, help="Fraction of data to use for testing")
    parser.add_argument('--scale_features', action='store_true', default=True, help="Enable feature scaling for training")
    parser.add_argument('--architecture', type=int, default=0, help="Full path to yaml with architecture")

    return vars(parser.parse_args())

def setup_logging(log_file="main.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Optional: Also log to console

def load_training_config(confg_name, arguments):
    
    confg_path = '../configs/training/' + confg_name + '.yaml'
    with open(confg_path, 'r') as file:
        config = yaml.safe_load(file)
    
    arguments.update(config)

def setup_results(args):
    # Get the current time
    current_time = datetime.now()

    # Format the time as yyyymmdd_HHMM
    current_time = current_time.strftime("%Y%m%d_%H%M")

    args['results_path'] = args['results_path'] + current_time +'/'

    os.makedirs(args['results_path'], exist_ok=True) 
    logging.info(f"Results directory set up at {args['results_path']}")


def retrieve_data(args):
    if not args['get_data']:
        print("Data retrieval mode is not enabled. Use --get_data to enable it.")
        return
    
    setup_logging(log_file='retrieve_data.log')  # Set up logging
    logging.info(f"Arguments for data retrieval: {args}")

    from src.data_manager import DataManager
    data_manager = DataManager(                            
                                interim_path=args['interim_path'],
                                pairs_path=args['pairs_path'],
                                raw_path=args['raw_path'],
                                arguments=args
    )
    logging.info("Data retrieval completed successfully.")

    # Copy log to interim and pairs paths
    if data_manager.created_interim:
        shutil.copy('retrieve_data.log', args['interim_path']+'.log')

    if data_manager.created_pairs:
        shutil.copy('retrieve_data.log', args['pairs_path']+'.log')

    os.remove('retrieve_data.log')  # Clean up log file after copying
                               
def train_model(args):

    # Setup logging for training
    setup_logging(log_file='train_model.log')  # Set up logging

    # Load in config file for training, overwriting any duplications in args
    if not args['training_cfg'] is None:
        load_training_config(args['training_cfg'], args)

    logging.info(args)

    setup_results(args)  # Set up results directory

    if not args['train']:
        print("Training mode is not enabled. Use --train to enable it.")
        return

    if not args['pairs_path'] or not args['results_path']:
        print("Both --pairs_path and --results_path must be provided for training.")
        return

    logging.info("Training model...")
    from src.train_nn import train_save_eval
    train_save_eval(args)

def main():
    
    args = parse_arguments()
    print(args)

    # Check if multiple modes are enabled
    if args['get_data'] + args['train'] + args['evaluate'] > 1:
        print("Error: Only one mode can be enabled at a time. Please choose one of --get_processed_data, --train, or --evaluate.")
        return

    elif args['get_data']:
        retrieve_data(args)

    elif args['train']:
        train_model(args)
        

if __name__ == "__main__":
    
    main()
        