"""
Graphs the relationship between mse and snr of the entire testing set
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.configfiles import read_json_config
from utils.data_processing import get_normalized_data, get_injection_parameters

# -----------------------------------------------------------------------------
# FUNCTIONS DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Saves a plot of the MSEs over the SNRs of the testing dataset')

    parser.add_argument('--testing-config',
                        default='default_testing.json',
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

    # Get predictions
    model_name = config['model']
    preds_name = f'{model_name[:-3]}_predictions.npy'
    preds_path = f'outputs/predictions/{preds_name}'
    preds = np.load(preds_path)

    # Get actual
    hdf_file_name = config['testing_hdf_file_name']
    y_test = get_normalized_data(hdf_file_name)[1]

    # Get snr values
    snrs = np.array(get_injection_parameters(hdf_file_name)['injection_snr'])

    # Get mse between predictions and actual
    def mse(actual: np.ndarray, predictions: np.ndarray) -> float:
        return ((actual - predictions) ** 2).mean()
    
    mses = np.array([mse(y, pred) for y, pred in zip(y_test, preds)])

    # Plot and save figure
    plt.scatter(snrs, mses, s=10)
    plt.xlim(5,20)

    plt.xlabel('Signal-to-Noise Ratio', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.title('MSE vs. SNR', fontsize=14, y=1.05)
    
    print('Saving figure... ', end='', flush=True)
    figure_name = f'{model_name[:-3]}_mse_vs_snr.png'
    figure_path = f'outputs/figures/{figure_name}'
    plt.savefig(figure_path)
    print('Done!', flush=True)