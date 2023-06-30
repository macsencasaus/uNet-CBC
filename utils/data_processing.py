"""
Preprocesses .hdf files containing the waveforms and metadata.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import h5py
import numpy as np

from h5py import Dataset, Group
from numpy import concatenate, array, zeros, newaxis

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def hdf_file_check(hdf_file_path: str):
    # Check if the HDF file actually exists
    if not os.path.exists(hdf_file_path):
        raise FileNotFoundError(f'Hdf file "{hdf_file_path}" does not exist!')
    
def get_normalized_data(hdf_file_name: str) -> tuple:
    """
    Pre-process and retrieve the input and label data

    Args:
        hdf_file_name: Path to the HDF file containg the data

    Returns:
        A tuple of the normalized inputs and labels
    """

    hdf_file_path = f'hdf/{hdf_file_name}'

    hdf_file_check(hdf_file_path)
    
    with h5py.File(hdf_file_path, 'r') as hdf_file:

        # Input data
        inputs_h1 = array(Group(hdf_file['injection_samples'])['h1_strain'])[:, newaxis]
        inputs_l1 = array(Group(hdf_file['injection_samples'])['l1_strain'])[:, newaxis]

        # Label data   
        labels_h1 = array(Group(hdf_file['injection_parameters'])['h1_signal'])[:, newaxis]
        labels_l1 = array(Group(hdf_file['injection_parameters'])['l1_signal'])[:, newaxis]

        # Noise samples
        noise_samples_h1 = array(Group(hdf_file['noise_samples'])['h1_strain'])[:, newaxis]
        noise_samples_l1 = array(Group(hdf_file['noise_samples'])['l1_strain'])[:, newaxis]

    # Merging and normalizing input data
    inputs = concatenate((inputs_h1, inputs_l1), axis = 2)
    for idx, input in enumerate(inputs):
        inputs[idx] = (input - input.mean()) / input.std() + 0.5

    # Merging and normalizing label data
    labels = concatenate((labels_h1, labels_l1), axis=2)
    for idx, label in enumerate(labels):
        labels[idx] /= np.max(np.abs(label))
    labels = (labels + 1) / 2

    # Merging and normalizing noise samples
    noise_samples = concatenate((noise_samples_h1, noise_samples_l1), axis=2)
    for idx, _ in enumerate(noise_samples):
        noise_samples[idx] = (_ - _.mean()) / _.std() + 0.5


    # Concatenate noise samples to inputs
    inputs = concatenate((inputs, noise_samples), axis=0)
    
    # Concatenate noise sample labels to labels
    labels = concatenate((labels, zeros(noise_samples.shape) + 0.5), axis=0)

    return inputs, labels

def get_raw_data(hdf_file_name: str) -> tuple:
    """
    Retrieves the raw data for plotting

    Args:
        hdf_file_name: name of the HDF file used for testing

    Returns:
        A tuple of the input data and the labels for plotting
    """

    hdf_file_path = f'hdf/{hdf_file_name}'

    hdf_file_check(hdf_file_path)
    
    with h5py.File(hdf_file_path, 'r') as hdf_file:

        # Input data
        inputs_h1 = array(Group(hdf_file['injection_samples'])['h1_strain'])
        inputs_l1 = array(Group(hdf_file['injection_samples'])['l1_strain'])

        # Label data   
        labels_h1 = array(Group(hdf_file['injection_parameters'])['h1_signal'])
        labels_l1 = array(Group(hdf_file['injection_parameters'])['l1_signal'])

        # Noise samples
        noise_samples_h1 = array(Group(hdf_file['noise_samples'])['h1_strain'])
        noise_samples_l1 = array(Group(hdf_file['noise_samples'])['l1_strain'])

    # Merging and normalizing input data
    inputs = concatenate((inputs_h1, inputs_l1), axis = 2)

    # Merging and normalizing label data
    labels = concatenate((labels_h1, labels_l1), axis=2)
    for idx, label in enumerate(labels):
        labels[idx] /= np.max(np.abs(label))

    # Merging and normalizing noise samples
    noise_samples = concatenate((noise_samples_h1, noise_samples_l1), axis=2)

    # Concatenate noise samples to inputs
    inputs = concatenate((inputs, noise_samples), axis=0)
    
    # Concatenate noise sample labels to labels
    labels = concatenate((labels, zeros(noise_samples.shape)), axis=0)

    return inputs, labels

def get_injection_parameters(hdf_file_name: str) -> dict:
    """
    Retrieves the injection parameters of HDF file

    Returns:
        Numpy array of dictionaries of the injection parameters
    """

    hdf_file_path = f'hdf/{hdf_file_name}'

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, 'r') as hdf_file:
        keys = ('mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec', 'cao_phase', 'inclination', 'polarization', 'injection_snr')
        injection_parameters_dict = {}
        for key in keys:
            injection_parameters_dict[key] = array(Group(hdf_file['injection_parameters'])[key])
    
    return injection_parameters_dict

def get_event_time(hdf_file_name: str, idx: int) -> float:
    """
    Retrieves the event time of an event of a given sample

    Args:
        hdf_file_path: path to the hdf file with data
        idx: index of the sample of the event time to be retrieved

    Returns:
        A float of the event time
    """

    hdf_file_path = f'hdf/{hdf_file_name}'

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, 'r') as hdf_file:
        event_times = Group(hdf_file['injection_samples'])['event_time']
        return float(Dataset(event_times)[idx])