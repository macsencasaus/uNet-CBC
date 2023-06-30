"""
Trains the U-net based on hdf_file data and config parameters
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import argparse
import time

from utils.configfiles import read_json_config
from utils.data_processing import get_normalized_data
from utils.model import get_model

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Trains and saves the U-net.')

    parser.add_argument('--training-config',
                        default='default_training.json',
                        type=str,
                        help='Name the JSON file the program uses to '
                        'compile the model, Default: default_training.json')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=True,
                        help='Choose to save the model after fitting, Default: True')
    
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # PRELIMINARIES
    # -------------------------------------------------------------------------

    print('')
    print('TRAINING AND SAVING MODEL')
    print('')

    # Start stopwatch
    script_start = time.time()

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f'config_files/{args.training_config}'
    config = read_json_config(config_path)

    # Set seeds
    seed = config['random_seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------
    # ACQUIRING THE TRAINING DATA
    # -------------------------------------------------------------------------
    
    hdf_file_name = config['training_hdf_file_name']
    X_train, y_train = get_normalized_data(hdf_file_name)

    # -------------------------------------------------------------------------
    # ACQUIRING, COMPILING, AND TRAINING THE 
    # -------------------------------------------------------------------------

    # Load model
    print('Loading model... ', end='', flush=True)
    model = get_model()
    print('Done!', flush=True)

    # Compile model
    print('Compiling model... ', end='', flush=True)
    model.compile(learning_rate=config['learning_rate'], loss=config['loss'], accuracy_metrics=config['accuracy_metrics'])
    print('Done!', flush=True)

    # Fit model
    print('')
    print('FITTING MODEL: ', flush=True)
    print('')
    model.fit(X_train, y_train, validation_split=config['validation_split'], epochs=config['epochs'], batch_size=config['batch_size'])

    # Save model
    if args.save_model:
        print('Saving model... ', end='', flush=True)
        model_file_path = f"outputs/models/{config['model_name']}"
        model.save(model_file_path)
        print('Done!', flush=True)

    # Print total runtime
    print('Total runtime: {:.1f}.'.format(time.time()-script_start))
    print('')