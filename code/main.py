import argparse 
import os

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--get_raw_data', action='store_true', help="Enable raw data retrieval mode")
    parser.add_argument('--get_intermediate_data', action='store_true', help="Enable intermediate data retrieval mode")
    parser.add_argument('--get_pairs_data', action='store_true', help="Enable processed data retrieval mode")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    parser.add_argument('--evaluate', action='store_true', help="Enable evaluation mode")

    # Optional arguments for raw and processed data retrieval
    parser.add_argument('--inputs', nargs='*', default=['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr'], help="List of potential input variables to retrieve from raw data")
    parser.add_argument('--input_path', type=str, required=False, help="Path to the raw data file, for use with --get_raw_data and --get_processed_data")
    parser.add_argument('--output_path', type=str, required=False, help="Path to save the processed data, for use with --get_raw_data and --get_processed_data")
    parser.add_argument('--features', nargs='*', default=['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr'], required=False, help="Features for the processed data")
    parser.add_argument('--labels', nargs='*', default=['sivelv'], required=False, help="Labels for the processed data")
    parser.add_argument('--subset_region', type=str, required=False, default='Arctic', help="Region to subset the data to, default is 'Arctic'")

    return vars(parser.parse_args())

def raw_data_retrieval(args):

    if not args['get_raw_data']:
        print("Raw data retrieval mode is not enabled. Use --get_raw_data to enable it.")
        return

    if not args['input_path'] or not args['output_path']:
        print("Both --input_path and --output_path must be provided for raw data retrieval.")
        return

    print("Retrieving raw data...")

    from src.get_raw_data import get_and_save

    get_and_save(args['input_path'], args['output_path'], args['inputs'])

    # check if the data was saved successfully
    if not os.path.exists(args['output_path']):
        print(f"Failed to save raw data to {args['output_path']}. Please check the path and try again.")
    else:
        print(f"Raw data saved to {args['output_path']}")

def intermediate_data_retrieval(args):
    
    if not args['get_intermediate_data']:
        print("Processed data retrieval mode is not enabled. Use --get_processed_data to enable it.")
        return

    if not args['input_path'] or not args['output_path']:
        print("Both --input_path and --output_path must be provided for processed data retrieval.")
        return
    
    print("Retrieving processed data...")
    from src.raw_processing import process_save_intermediate_data
    process_save_intermediate_data(args['input_path'], args['output_path'], args)

def pairs_retrieval(args):
    if not args['get_pairs_data']:
        print("Processed data retrieval mode is not enabled. Use --get_processed_data to enable it.")
        return

    if not args['input_path'] or not args['output_path']:
        print("Both --input_path and --output_path must be provided for processed data retrieval.")
        return
    
    print("Retrieving processed data...")
    from src.raw_processing import process_save_pairs
    process_save_pairs(args['input_path'], args['output_path'], args)



def main():
    
    args = parse_arguments()
    print(args)

    # Check if multiple modes are enabled
    if args['get_raw_data'] + args['get_intermediate_data'] + args['train'] + args['evaluate'] > 1:
        print("Error: Only one mode can be enabled at a time. Please choose one of --get_raw_data, --get_processed_data, --train, or --evaluate.")
        return
    
    if args['get_raw_data']:
        raw_data_retrieval(args)

    elif args['get_intermediate_data']:
        intermediate_data_retrieval(args)

    elif args['get_pairs_data']:
        pairs_retrieval(args)
        

if __name__ == "__main__":
    
    main()
        