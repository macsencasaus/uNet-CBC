# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

def mean_squared_error(output: np.ndarray, label: np.ndarray) -> float:
    """
    Computes the mean squared error between an output from the model and the corresponding label

    Args:
        output: one output waveform from the model
        label: one label of the raw waveform

    Returns:
        the computed mean squared error
    """

    # Mean squared error formula
    mean_squared_error = float(((label - output) ** 2).mean())

    return mean_squared_error