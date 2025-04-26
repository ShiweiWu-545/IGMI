import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('..')
from scripts.datasets import *


class run_model():

    def __init__(self):
        super().__init__()

    def make_validation_step(self, model, batch):

        x_validation = recursive_to(batch, config.model.device)
        model.eval()
        yhat = model(x_validation['wt'], x_validation['mut'])
        return yhat