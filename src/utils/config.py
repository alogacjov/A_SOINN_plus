import os
import yaml
import json


class Config:
    def __init__(self, path):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.DS_PATH = cfg['DS_PATH']
        self.FE_PATH = cfg['FE_PATH']
        self.BC = cfg['BC']
        self.TRAIN_OBJECTS = cfg['TRAIN_OBJECTS']
        self.TEST_OBJECTS = cfg['TEST_OBJECTS']
        self.SEED = cfg['SEED']

        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']
        self.ALGORITHM = self.ALGORITHM_ARGS['ALGORITHM']

    @property
    def sessions(self):
        """Which sessions to use, based on background complexity]"""
        with open(os.path.join(self.DS_PATH,
                               'background_complexities.json')) as f:
            bc_dict = json.load(f)
        return bc_dict[self.BC]

    @property
    def labels(self):
        """Category and instance names"""
        with open(os.path.join(self.DS_PATH,
                               'labels.json')) as f:
            labels_dict = json.load(f)
        return labels_dict

    @property
    def train_labels(self):
        """Category and instance names of the training subset"""
        return self._get_label_subset(self.TRAIN_OBJECTS)

    @property
    def test_labels(self):
        """Category and instance names of the training subset"""
        return self._get_label_subset(self.TEST_OBJECTS)

    def _get_label_subset(self, list_of_objects):
        labels_dict = {}
        for category in self.labels.keys():
            labels_dict[category] = {}
            for instance, object in self.labels[category].items():
                if object in list_of_objects:
                    labels_dict[category][instance] = object
        return labels_dict

    @property
    def category_indices(self):
        """label index of each category"""
        return {c:i for i, c in enumerate(self.labels.keys())}

    @property
    def instance_indices(self):
        """label index of each instance/object"""
        inst_indices = {}
        for category in self.labels.keys():
            for instance, object in self.labels[category].items():
                # Append object index
                inst_indices[instance] = int(object.split('o')[-1])
        return inst_indices
