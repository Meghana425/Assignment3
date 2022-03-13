# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:57:15 2022

@author: Meghana
"""
# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load the runner for the latest ScikitLearn model we just saved
runner = bentoml.sklearn.load_runner("movie_popularity:latest")


svc = bentoml.Service("movie_popularity", runners=[runner])

# Create API function with pre- and post- processing logic with your new "svc" annotation
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    # Define pre-processing logic
    result = runner.run(input_series)
    # Define post-processing logic
    return result
