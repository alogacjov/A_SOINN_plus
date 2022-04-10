import os
import random
import numpy as np
from PIL import Image


class DataLoader():
    def __init__(self, ds_path, objects, sessions,
                 category_indices, instance_indices,
                 fe_path=None, random_seed=None,
                 category_order=None,
                 keep_prev_batch=False):
        '''A data loader to yield LL data batches

        Parameters
        ----------
        ds_path: str
            Path to the dataset
        fe_path: str
            Path to the feature extractor .pkl. If None, images returned
        objects: dict of str
            Dict of objects/instances to consider. Required form:
            {"category_i": {"instance_a":"o_a", "instance_b":"o_b",...},
             "category_j": {"instance_a":"o_a", "instance_b":"o_b",...},
             ...}
        sessions: list of str
            List of sessions to consider (depends on background complexity)
        category_indices, instance_indices: dict of str
            Map category/instance to a single integer value
        random_seed: int or None
            Whether to shuffle categories with a given seed
        category_order: list of str or None
            Whether to predefine the order of the categories
        keep_prev_batch: bool
            Whether to keep the batch of the previous iteration in memory
            for the next iteration. Leads to an increased batch size after
            each iteration. Is helpful for forgetting computation on a test
            set.

        '''
        self.ds_path = ds_path
        self.objects = objects
        self.sessions = sessions
        self.category_indices = category_indices
        self.instance_indices = instance_indices
        self.keep_prev_batch = keep_prev_batch
        self.prev_batch = []
        self.prev_labels = []
        self.fe = None
        if fe_path is not None:
            from utils import featurizer
            self.fe = featurizer.Featurizer(fe_path)
        if category_order is not None:
            self.categories = category_order
        else:
            # For iteration
            self.categories = list(self.objects.keys())
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(self.categories)
        # To pop in correct order
        self.categories.reverse()
    
    def __next__(self):
        if len(self.categories) != 0:
            category = self.categories.pop()
            category_batch = []
            labels = []
            for session in self.sessions:
                for instance, object in self.objects[category].items():
                    path_to_files = os.path.join(
                        self.ds_path,
                        session,
                        object
                    )
                    arrs = self._get_all_frames(path_to_files)
                    if self.fe is not None:
                        arrs = self.fe.create_features(arrs)
                    category_batch.append(arrs)
                    _labels = [[self.instance_indices[instance],
                                self.category_indices[category]]]
                    labels += _labels*len(arrs)
            category_batch = np.vstack(np.array(category_batch))
            labels = np.array(labels)
            if self.keep_prev_batch:
                self.prev_batch.append(category_batch)
                self.prev_labels.append(labels)
                return self.prev_batch, self.prev_labels
            else:
                return category_batch, labels
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.objects)

    def _get_all_frames(self, path):
        '''Returns numpy arrays for each frame in a given path

        It can either load .png images or .npz files

        '''
        arrs = []
        if os.path.isdir(path):  # Load images
            for filename in os.listdir(path):
                if not filename.lower().endswith('.png'): continue
                with Image.open(os.path.join(path,filename)) as f:
                    img = np.array(f)
                arrs.append(img)
            return np.array(arrs)
        else:  # Load npz file
            return np.load(path+'.npz')['arr_0']
