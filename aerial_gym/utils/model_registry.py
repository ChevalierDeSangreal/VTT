import os
import torch
import numpy as np


class Modelregistry():
    def __init__(self):
        self.model_classes = {}
        self.model_cfgs = {}

    def register(self, name: str, model_class, model_cfg):
        self.model_classes[name] = model_class
        self.model_cfgs[name] = model_cfg
    
    def get_model_class(self, name: str):
        return self.model_classes[name]

    def get_cfgs(self, name: str):
        model_cfg = self.model_cfgs[name]
        return model_cfg
    
    def make_model(self, name: str):
        """
        Creates a model either from a registered name using registered config
        Args:
            name (string): Name of a registered env.
            model_cfg (Dict, optional): Model config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            Class ?: The created model
            Dict: the corresponding config file
        
        """
        if name in self.model_classes:
            model_class = self.get_model_class(name)
        else:
            raise ValueError(f"Model with name: {name} was not registered")

        model_cfg = self.get_cfgs(name)
        
        return model_class, model_cfg
    
model_registry = Modelregistry()