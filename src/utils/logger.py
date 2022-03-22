import os
import csv
import pickle
import numpy as np
import tempfile


class Logger():
    def __init__(self, output_path, *args):
        ''' Class for logging of LL metrics

        Average forgetting and average accuracy as defined in:
        A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr,
        “Riemannian Walk for Incremental Learning: Understanding
        Forgetting and Intransigence,”
        arXiv:1801.10112 [cs], vol. 11215, pp. 556–572, 2018,
        doi: 10.1007/978-3-030-01252-6_33.
        'time' allows logging of time in seconds after every iteration
        'space' allows logging the space requirements of the given
        model in bytes assuming a .pkl file format.

        Parameters
        ----------
        output_path: str
            Where to save the log file
        args: strings
            which metrics to log, allowed:
            ['average_accuracy', 'average_forgetting',
             'num_units', 'time', 'space']

        '''
        self.output_path = output_path
        if not os.path.exists(output_path): os.makedirs(output_path)
        self.log_avrg_acc = 'average_accuracy' in args
        self.log_avrg_frgt = 'average_forgetting' in args
        self.log_num_units = 'num_units' in args
        self.log_time = 'time' in args
        self.log_space = 'space' in args
        self.avrg_accuracies = []
        self.avrg_forgettings = []
        self.num_units = []
        self.batch_times = []
        self.model_sizes = []
        self.all_t_accs = []  # Accuracy results of ever task

    def log(self, task_accuracies,
            unit_num=None,
            batch_duration_sec=None,
            model=None):
        '''logs the given metrics

        Parameters
        ----------
        task_accuracies : list of float
            Accuracies of tasks j while learning task k
        unit_num : int
            If num_units shall be logged,
            it has to be given as a parameter
        batch_duration_sec : float
            seconds to log
        model : object
            log size in bytes of given object, when saved as .pkl file

        '''
        if self.log_avrg_acc:
            self.avrg_accuracies.append(np.mean(task_accuracies))
        if self.log_avrg_frgt:
            self.avrg_forgettings.append(
                self.average_forgetting(task_accuracies)
            )
        self.all_t_accs.append(task_accuracies)
        if self.log_num_units and unit_num is not None:
            self.num_units.append(unit_num)
        if self.log_time and batch_duration_sec is not None:
            self.batch_times.append(batch_duration_sec)
        if self.log_space and model is not None:
            self.model_sizes.append(self.get_size(model))
        self.write()

    def average_forgetting(self, task_accuracies):
        '''Computes average forgetting of the current task k

        Parameters
        ----------
        task_accuracies : list of float
            Accuracies of tasks j while learning task k

        Returns
        -------
        float

        '''
        # No task can be forgotten when learning the first task
        if len(self.all_t_accs) == 0: return None
        forgetting_k = []
        for j in range(len(self.all_t_accs)):
            # Maximum previous accuracy of task j
            max_t_acc = np.max([a[j] for a in self.all_t_accs if j<len(a)])
            forgetting_j_k = max_t_acc - task_accuracies[j]
            forgetting_k.append(forgetting_j_k)
        return np.mean(forgetting_k)

    def get_size(self, object):
        '''Get size in bytes of given object when saved as .pkl file'''
        _temp_file = tempfile.NamedTemporaryFile()
        pickle.dump(object, _temp_file)
        filesize = os.stat(_temp_file.name).st_size
        _temp_file.close()
        return filesize


    def write(self):
        '''Writes current log state to output file'''
        logs = {}
        if self.log_avrg_acc:
            logs['average_accuracy'] = self.avrg_accuracies
        if self.log_avrg_frgt:
            logs['average_forgetting'] = self.avrg_forgettings
        if self.log_num_units:
            logs['num_units'] = self.num_units
        if self.log_time:
            logs['task_time_sec'] = self.batch_times
        if self.log_space:
            logs['model_size_bytes'] = self.model_sizes
        # Write to file:
        with open(os.path.join(self.output_path, 'logs.csv'), "w") as f:
            writer = csv.writer(f)
            writer.writerow(logs.keys())
            writer.writerows(zip(*logs.values()))
