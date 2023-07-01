"""
Plots the input, label, and prediction of a certain sample in the testing sample
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from utils.configfiles import read_json_config
from utils.data_processing import get_injection_parameters, get_raw_data, get_event_time

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Plots the input, label, and prediction of one sample '
                                     'from both detectors')
    
    parser.add_argument('--testing-config',
                        default='default_testing.json',
                        type=str,
                        help='Name of the JSON file the model used to test, Default: default_testing.json')
    parser.add_argument('--sample-id',
                        help='ID of the sample to be view (an integer '
                        'between 0 and n_injection_samples + n_noise_samples),'
                        'Default: 0',
                        default=0)
    
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # Start stopwatch
    script_start = time.time()

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f'config_files/{args.training_config}'
    config = read_json_config(config_path)

    # Get input data & labels
    hdf_file_name = config['testing_hdf_file_name']
    inputs, labels = get_raw_data(hdf_file_name)

    # Get predictions
    model_name = config['model']
    preds_name = f'{model_name[:-3]}_predictions.npy'
    preds_path = f'outputs/predictions/{preds_name}'
    preds = np.load(preds_path)

    # Unnormalize predictions
    preds = preds * 2 - 1
    preds = np.reshape(preds, labels.shape)

    # Get event time
    event_time = get_event_time(hdf_file_name, args.sample_id)
    signal_length = np.arange(0, IMG_SIZE) / POLLING_RATE - 

    # Create figure
    fig, ax1 = plt.subplots(nrows=3, ncols=2)
    
    ax1[0][0].set_ylabel('Whitened Signal Strain', color = 'C0', fontsize= 8)   
    ax1[1][0].set_ylabel('Raw Signal Strain', color = 'C1', fontsize= 8)
    ax1[2][0].set_ylabel('U-Net Prediction Strain', color = 'C2', fontsize= 8)
    ax1[0][1].set_yticklabels([])
    ax1[1][1].set_yticklabels([])
    ax1[2][1].set_yticklabels([])
    ax1[0][1].tick_params(left = False)
    ax1[1][1].tick_params(left = False)
    ax1[2][1].tick_params(left = False)
        
    for idx, _ in enumerate(['H1', 'L1']):
      
      ax1[0][idx].set_title(_)
      
      ax1[0][idx].plot(time, X_train[image_x,:,i])    