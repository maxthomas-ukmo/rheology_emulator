import argparse 
import os

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--get_raw_data', action='store_true', help="Enable raw data retrieval mode")
    parser.add_argument('--train', action='store_true', help="Enable training mode")
    parser.add_argument('--evaluate', action='store_true', help="Enable evaluation mode")

    # Optional arguments for raw data retrieval
    parser.add_argument('--potential_inputs', nargs='*', default=['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr'], help="List of potential input variables to retrieve from raw data")
    parser.add_argument('--raw_path', type=str, required=False, help="Path to the raw data file, for use with --get_raw_data")
    parser.add_argument('--save_path', type=str, required=False, help="Path to save the processed data, for use with --get_raw_data")

    return vars(parser.parse_args())

def raw_data_retrieval(args):

    if not args['get_raw_data']:
        print("Raw data retrieval mode is not enabled. Use --get_raw_data to enable it.")
        return

    if not args['raw_path'] or not args['save_path']:
        print("Both --raw_path and --save_path must be provided for raw data retrieval.")
        return

    print("Retrieving raw data...")

    from src.get_raw_data import get_and_save

    get_and_save(args['raw_path'], args['save_path'], args['potential_inputs'])

    # check if the data was saved successfully
    if not os.path.exists(args['save_path']):
        print(f"Failed to save raw data to {args['save_path']}. Please check the path and try again.")
    else:
        print(f"Raw data saved to {args['save_path']}")


def main():
    
    args = parse_arguments()
    print(args)

    # Check if multiple modes are enabled
    
    if args['get_raw_data']:
        raw_data_retrieval(args)

if __name__ == "__main__":
    
    main()
        