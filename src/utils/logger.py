import numpy as np
import pandas as pd


class Logger():
    def __init__(self, output_path, *args):
        ''' Class for logging of LL metrics

        Average forgetting and average accuracy as defined in:
        A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr,
        “Riemannian Walk for Incremental Learning: Understanding
        Forgetting and Intransigence,”
        arXiv:1801.10112 [cs], vol. 11215, pp. 556–572, 2018,
        doi: 10.1007/978-3-030-01252-6_33.

        Parameters
        ----------
        output_path: str
            Where to save the log file
        args: strings
            which metrics to log, allowed:
            ['average_accuracy', 'average_forgetting', 'num_units']

        '''
        self.output_path = output_path
        self.log_avrg_acc =  'average_accuracy' in args
        self.log_avrg_frgt =  'average_forgetting' in args
        self.log_num_units =  'num_units' in args
        self.avrg_accuracies = []
        self.avrg_forgettings = []
        self.num_units = []
        self.all_t_accs = []  # Accuracy results of ever task

    def log(self, task_accuracies, unit_num=None):
        '''logs the given metrics

        Parameters
        ----------
        task_accuracies : list of float
            Accuracies of tasks j while learning task k
        unit_num : int
            If num_units shall be logged,
            it has to be given as a parameter

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


    def write(self):
        '''Writes current log state to output file'''
        log_df = pd.DataFrame()
        # TODO
