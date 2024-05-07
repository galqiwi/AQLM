import pickle
import torch.nn as nn


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'transformers.activations' and name == 'SiLUActivation':
            return nn.ReLU
        return super().find_class(module, name)


def load(file, *argv, **kwargs):
    return Unpickler(file, *argv, **kwargs).load()
