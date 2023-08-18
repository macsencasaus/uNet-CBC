"""
Graphs the relationship between mse and snr of the entire testing set
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from numpy import array
from utils.configfiles import read_json_config
from utils.data_processing import get_raw_data, get_injection_parameters

# -----------------------------------------------------------------------------
# FUNCTIONS DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Saves a plot of the MSEs over the SNRs of the testing dataset')

    parser.add_argument('--testing-config',
                        default='testing.json',
                        type=str,
                        help='Name of the JSON file the model used to test')
    
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # PRELIMINARIES
    # -------------------------------------------------------------------------

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f'config_files/{args.testing_config}'
    config = read_json_config(config_path)

    # -------------------------------------------------------------------------
    # ACQUIRING DATA & GRAPHING
    # -------------------------------------------------------------------------

    # Get actual
    hdf_file_name = config['testing_hdf_file_name']
    labels = get_raw_data(hdf_file_name)[1]

    # Get predictions
    model_name = config['model_name']
    preds_name = f'{model_name[:-3]}_predictions.npy'
    preds_path = f'outputs/predictions/{preds_name}'
    preds = np.load(preds_path)
    preds = preds * 2 - 1
    preds = map(lambda x: x/np.max(np.abs(x)), preds)

    # Get snr values
    snrs = np.array(get_injection_parameters(hdf_file_name)['injection_snr'])

    # Get mse between predictions and actual
    def mse(actual: np.ndarray, predictions: np.ndarray) -> float:
        return ((actual - predictions) ** 2).mean()
    
    mses = array(list(map(mse, labels, preds)))[:len(snrs)]

    # Plot and save figure
    plt.scatter(snrs, mses, s=10)
    plt.xlim(5,20)

    plt.xlabel('Signal-to-Noise Ratio', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.title('MSE vs. SNR', fontsize=14, y=1.05)
    plt.gcf().set_size_inches(10,8,forward=True)
    
    print('Saving figure... ', end='', flush=True)
    figure_name = f'{model_name[:-3]}_mse_vs_snr.png'
    figure_path = f'outputs/figures/{figure_name}'
    
    try:
        plt.savefig(figure_path)
    except FileNotFoundError:
        os.mkdir('outputs/figures')
        plt.savefig(figure_path)
    print('Done!', flush=True)