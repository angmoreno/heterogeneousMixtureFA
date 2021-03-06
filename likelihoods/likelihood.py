
import torch
import numpy as np

class Likelihood(torch.nn.Module):
    """
    Base class for likelihoods
    """
    def __init__(self):
        super(Likelihood, self).__init__()
