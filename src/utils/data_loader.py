import os
import random
import numpy as np
from PIL import Image


class DataLoader():
    def __init__(self, ds_path, objects, sessions,
                 category_indices, instance_indices,
                 fe_path=None, shuffle=False, keep_prev_batch=False):
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
        shuffle: bool
            Whether to shuffle categories
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
        self.fe = None
        if fe_path is not None:
            from utils import featurizer
            self.fe = featurizer.Featurizer(fe_path)
        # For iteration
        self.categories = list(self.objects.keys())
        if shuffle: random.shuffle(self.categories)
        self.categories.reverse()  # To pop in correct order
    
    def __next__(self):
        current_batch = []
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
            return category_batch, labels
            # if self.keep_prev_batch:
            #     current_batch.append(category_batch)
            #     for curr_batch in current_batch:
            #         return curr_batch
            # else:
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
