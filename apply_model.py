"""
Applies the U-net based on the testing data
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import argparse
import time
import os

from keras import models
from utils.configfiles import read_json_config
from utils.data_processing import get_normalized_data

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description='Tests and saves the results of the U-net.')

    parser.add_argument('--testing-config',
                        default='testing.json',
                        type=str,
                        help='Name the JSON file the program uses to '
                        'evaluate the model, Default: testing.json')
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=True,
                        help='Choose to save the model\'s predictions of the '
                        'testing data as numpy objects, Default: True')
    
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # PRELIMINARIES
    # -------------------------------------------------------------------------

    print('')
    print('TESTING & EVALUATING MODEL', flush=True)
    print('')

    # Start stopwatch
    script_start = time.time()

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f'config_files/{args.testing_config}'
    config = read_json_config(config_path)

    # Set seeds
    seed = config['random_seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------
    # ACQUIRING THE MODEL & TESTING DATA
    # -------------------------------------------------------------------------
    
    # Get model
    model_name = config['model_name']
    model_path = f'outputs/models/{model_name}'
    model = models.load_model(model_path)

    hdf_file_name = config['testing_hdf_file_name']
    X_test, y_test = get_normalized_data(hdf_file_name)

    # Save predictions
    if args.save_predictions:
        print('Predicting and saving predictions: ', flush=True)

        batch_size = int(config['batch_size'])
        preds = model.predict(X_test, batch_size=batch_size)
        preds_name = f'{model_name[:-3]}_predictions.npy'
        preds_path = f'outputs/predictions/{preds_name}'

        try:
            np.save(preds_path, preds)
        except FileNotFoundError:
            os.mkdir('outputs/predictions')
            np.save(preds_path, preds)
            
        print('Done!', flush=True)
        print('')

    # Evaluating model against accuracy metrics
    print('Evaluating model: ', flush=True)
    model.evaluate(X_test, y_test, batch_size=config['batch_size'])
    print('Done!', flush=True)

    # Print total runtime
    print('')
    print('Total runtime: {:.1f}s'.format(time.time()-script_start))
    print('')