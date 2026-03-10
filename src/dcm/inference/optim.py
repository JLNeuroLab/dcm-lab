import numpy as np

from dcm.inference.forward_adapter import ForwardAdapter
from dcm.inference.likelihoods import gaussian_log_likelihood
from dcm.inference.priors import gaussian_log_prior
from dcm.inference.objectives import gaussian_log_posterior

def map_estimation()