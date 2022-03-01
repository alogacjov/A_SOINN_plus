##################################################################
# A reimplementation of the Growing Dual-Memory                  #
# approach of Parisi et al.:                                     #
#                                                                #
# Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018)          #
# Lifelong Learning of Spatiotemporal Representations            #
# with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966 #
#                                                                #
# Many parts of the original implementation are used here. See:  #
# https://github.com/giparisi/GDM                                #
#                                                                #
# The modifications are only made to make the GDM usable with    #
# our experimental setup                                         #
#                                                                #
##################################################################


import numpy as np
import math
from tqdm import tqdm
from models.GWR.gammagwr import GammaGWR


class EpisodicGWR(GammaGWR):
    def __init__(self):
        '''Modified version of Parisi et al.'s episodic_gwr.EpisodicGWR'''
        self.iterations=0

    def init_network(self, input_dimension, e_labels, num_context) -> None:
        '''Network initialization Overrides GammaGWR's init_network

        Parameters
        ----------
        input_dimension: int
            Nodes' dimension
        e_labels: int
            Number of labels for each node (2: category+instance)
        num_context: int
            K in original paper

        '''
        assert self.iterations < 1, "Can't initialize a trained network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2

        self.num_nodes_removed = None
        self.num_nodes_added = None

        self.dimension = input_dimension

        self.num_context = num_context
        self.depth = self.num_context + 1
        # the first of the num_context+1 vectors
        # is the weight vector of the neuron
        # the remaining are the K=num_context
        # context vectors
        empty_neuron = np.zeros((self.depth, self.dimension))
        # These are the 2 nodes both with contextvector and weights
        self.weights = [empty_neuron.copy(), empty_neuron.copy()]

        # Each context vector is a N dim vector
        self.g_context = np.zeros((self.depth, self.dimension))
        self.g_prediction_context = np.zeros((self.depth, self.dimension))

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # Temporal connections (to enable memory replay)
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels
        self.alabels = []  # associative matrix
        # Since we have no label at the beginning, we start
        # with an empty label dict
        for _ in range(0, self.num_labels):
            alabels = []
            for _ in range(self.num_nodes):
                alabels.append({})
            self.alabels.append(alabels)

        init_ind = list(range(0, self.num_nodes))
        # init of none 0 current (zeroth entry) weights
        for i in range(0, len(init_ind)):
            self.weights[i][0] = np.random.rand(self.dimension)

        # Context coefficients
        # used to compute distance of input and units
        # are regulating influance of temporal context
        self.alphas = self.compute_alphas(self.depth)

    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if new_node:
            # Expand to consider new neuron but connections
            # are first set to 0
            self.temporal = super().expand_matrix(self.temporal)
        if previous_ind != -1 and previous_ind != current_ind:
            # Temporal connection between previous activated neuron and
            # current/new neuron are incremented
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        """Updating the associative matrix

        Overrides GammaGWR's update_labels
        """
        new_node = kwargs.get('new_node', False)
        # Check whether the label already exists
        # and create if not:
        for i, l in enumerate(label):
            for node_idx, node_alabels in enumerate(self.alabels[i]):
                if l not in node_alabels:
                    self.alabels[i][node_idx][l] = -1
        if not new_node:
            for i, curr_label in enumerate(label):
                bmu_labels = self.alabels[i][bmu]
                for alabel in bmu_labels:
                    if curr_label == alabel:
                        bmu_labels[alabel] += self.a_inc
                    else:
                        if bmu_labels[alabel] != -1:
                            bmu_labels[alabel] -= self.a_dec
                            if (bmu_labels[alabel] < 0):
                                bmu_labels[alabel] = 0
                self.alabels[i][bmu] = bmu_labels
        else:
            for l in range(0, self.num_labels):
                new_alabel_dct = {k: 0 for k in self.alabels[l][0]}
                if label[l] != -1:
                    new_alabel_dct[label[l]] += self.a_inc
                self.alabels[l].append(new_alabel_dct)

    def remove_isolated_nodes(self, verbose=True):
        '''Remove nodes without edges

        Overrides GammaGWR's remove_isolated_nodes
        '''
        if self.num_nodes > 2:
            ind_c = 0
            rem_c = 0
            # go trough all nodes:
            while (ind_c < self.num_nodes):
                # get all neighbors of node:
                neighbours = np.nonzero(self.edges[ind_c])
                # if no neighbors:
                if len(neighbours[0]) < 1:
                    if self.num_nodes > 2:
                        self.weights.pop(ind_c)
                        self.habn.pop(ind_c)
                        for l in range(0, self.num_labels):
                            self.alabels[l].pop(ind_c)
                        self.edges = np.delete(self.edges, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=1)
                        self.ages = np.delete(self.ages, ind_c, axis=0)
                        self.ages = np.delete(self.ages, ind_c, axis=1)
                        self.temporal = np.delete(self.temporal,
                                                  ind_c, axis=0)
                        self.temporal = np.delete(self.temporal,
                                                  ind_c, axis=1)
                        self.num_nodes -= 1
                        rem_c += 1
                    else:
                        return rem_c
                else:
                    ind_c += 1
            if verbose:
                print("(-- Removed %s neuron(s))" % rem_c)
            return rem_c

    def get_label_of_neuron(self, neuron_idx, label_level=0):
        """ Calculates the label of a neuron with the most counts

        Parameters
        ----------
        neuron_idx: int
        label_level: int (default 0)
            if multiple class levels (e.g. category+instance)
            give index of level.

        Returns
        -------
        int
            index of the label
        
        """
        if len(self.alabels[label_level][neuron_idx]) == 0:
            return -1
        return max(self.alabels[label_level][neuron_idx],
                   key=self.alabels[label_level][neuron_idx].get)

    def train(self, ds_vectors,
              ds_labels,
              epochs,
              beta,
              l_rates,
              context,
              max_age=math.inf,
              a_threshold=0.3,
              regulated=False,
              verbose=True) -> None:

        assert not self.locked, "Network is locked. Unlock to train."

        num_correct_classified = 0
        num_new_add_neurons = 0
        num_updated_weights = 0
        self.samples = ds_vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta
        self.regulated = regulated  # semantic layer True/False
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        self.hab_threshold = 0.1
        # tau determines how fast the hab_counter is decreased
        # of BMU faster than on neighbor
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = math.inf
        self.max_age = max_age
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        self.mod_rate = 0.01

        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))
        previous_ind = -1
        for epoch in range(0, self.max_epochs):
            for iteration in range(0, self.samples):

                # Generate input sample
                self.g_context[0] = ds_vectors[iteration]
                label = ds_labels[iteration]

                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (self.beta * previous_bmu[z]) + \
                                        ((1-self.beta) * previous_bmu[z-1])

                # Find the best and second-best matching neurons
                b_index, b_distance, s_index  = super().find_bmus(
                    self.g_context,
                    s_best=True
                )

                if regulated:
                    # For semantic
                    b_label = self.get_label_of_neuron(b_index, -1)
                    misclassified = b_label != label[-1]
                else:
                    # For episodic
                    b_label = self.get_label_of_neuron(b_index, 0)
                    misclassified = b_label != label[0]
                if not misclassified:
                    num_correct_classified += 1

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (a < self.a_threshold and \
                   self.habn[b_index] < self.hab_threshold and \
                   self.num_nodes < self.max_nodes) and \
                   ((not self.regulated) or (self.regulated and misclassified)):
                    num_new_add_neurons += 1
                    # Add new neuron
                    n_index = self.num_nodes
                    super().add_node(b_index)
                    # Add label histogram
                    self.update_labels(n_index, label,
                                       new_node=True)

                    # Update edges and ages
                    super().update_edges(b_index,
                                         s_index,
                                         new_index=n_index)

                    # Update temporal connections
                    self.update_temporal(n_index,
                                         previous_ind,
                                         new_node=True)

                    # Habituation counter
                    super().habituate_node(n_index,
                                           self.tau_b,
                                           new_node=True)

                else:
                    super().habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    # The learning reates of b and n with
                    # b_rate > n_rate:
                    b_rate, n_rate = self.epsilon_b, self.epsilon_n
                    # regulated in case of semantic
                    # decreasing learning rate
                    if self.regulated and misclassified:
                        self.epsilon_b *= self.mod_rate
                        self.epsilon_n *= self.mod_rate
                        b_rate, n_rate = self.epsilon_b, self.epsilon_n

                    if (not self.regulated) or \
                       (self.regulated and not misclassified):
                        num_updated_weights += 1

                        self.update_labels(b_index, label)

                        super().update_weight(b_index, b_rate)

                        # Update BMU's edges // Remove BMU's oldest ones
                        super().update_edges(b_index, s_index)

                        # Update temporal connections
                        self.update_temporal(b_index, previous_ind)

                        # Update BMU's neighbors
                        super().update_neighbors(b_index, n_rate)

                self.iterations += 1

                previous_ind = b_index

            # Remove old edges
            super().remove_old_edges()

            error_counter[epoch] /= self.samples
            if verbose:
                print(("(Epoch: %s, NN: %s, ATQE: %s, " +
                       "Num_correct_classified: %s, " +
                       "num_neurons_added: %s, " +
                       "num_updated_weights: %s)") % (epoch + 1,
                                                      self.num_nodes,
                                                      error_counter[epoch],
                                                      num_correct_classified,
                                                      num_new_add_neurons,
                                                      num_updated_weights))
        self.num_nodes_added = num_new_add_neurons
        # Remove isolated neurons
        num_rmvd_nodes = self.remove_isolated_nodes(verbose=verbose)
        self.num_nodes_removed = num_rmvd_nodes

    def test(self, ds_vectors, ds_labels, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', False)
        test_vecs = kwargs.get('ret_vecs', False)
        test_samples = ds_vectors.shape[0]
        self.bmus_index = -np.ones(test_samples)
        self.bmus_weight = np.zeros((test_samples, self.dimension))
        self.bmus_label = -np.ones((self.num_labels, test_samples))
        # How high was the activation f.e. test sample:
        self.bmus_activation = np.zeros(test_samples)
        # same dimension like global context or weights matrix:
        input_context = np.zeros((self.depth, self.dimension))

        if test_accuracy:
            acc_counter = np.zeros(self.num_labels)
        for i in range(0, test_samples):
            # Again first entry in context matrix is the input itself
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = super().find_bmus(input_context)
            # Fill the arrays, defined above
            self.bmus_index[i] = b_index
            self.bmus_weight[i] = self.weights[b_index][0]
            self.bmus_activation[i] = math.exp(-b_distance)
            for l_index in range(0, self.num_labels):
                self.bmus_label[l_index, i] = self.get_label_of_neuron(
                    b_index,
                    label_level=l_index
                )
            input_context_copy = input_context.copy()
            for j in range(1, self.depth):
                input_context_copy[j] = input_context[j-1]
            input_context = input_context_copy
            if test_accuracy:
                for l_index in range(0, self.num_labels):
                    if self.bmus_label[l_index, i] == ds_labels[i, l_index]:
                        acc_counter[l_index] += 1
        if test_accuracy:
            self.test_accuracy = acc_counter / ds_vectors.shape[0]
        if test_vecs:
            s_labels = -np.ones((test_samples, 1))
            s_labels[:, 0] = ds_labels[:, 1]
            return self.bmus_weight, s_labels


class GDM():
    def __init__(self, replay=True, semantic=True):
        """ The GDM class

        A Modified version of Parisi et al.'s gdm_demo

        Parameters
        ----------
        replay: bool
            True if replay should be used
        semantic: bool
            True if G-SM should be used.

        """
        self.replay = replay
        self.semantic = semantic


    def replay_samples(self, net, size) -> (np.ndarray, np.ndarray):
        """ Returns replay samples.

        Parameters
        ----------
        net: EpisodicGWR
            The network of which units we create RNATs.
        size: int
            Size of the RNATs

        Returns
        -------
        returns: (np.ndarray, np.ndarray)
            RNATs of size "size" and dimension "net.dimension"
            for each unit in the "net"

        """
        samples = np.zeros(size)
        r_weights = np.zeros((net.num_nodes, size, net.dimension))
        r_labels = np.zeros((net.num_nodes, size, net.num_labels))
        net_weights = np.asarray(net.weights)
        for i in range(0, net.num_nodes):
            samples = np.zeros(size)
            for r in range(0, size):
                if r == 0:
                    samples[r] = i
                else:
                    samples[r] = np.argmax(net.temporal[int(samples[r-1]),
                                                        :])
                r_weights[i, r] = net_weights[int(samples[r]), 0]
                for l_num in range(0, net.num_labels):
                    r_labels[i, r, l_num] = net.get_label_of_neuron(
                        int(samples[r]),
                        label_level=l_num
                    )
        return r_weights, r_labels

    def train(self,
              dataset,
              max_age,
              num_context=2,
              learning_rates=[0.5, 0.005],
              a_threshold=[0.3,0.03],
              input_dimension=128,
              only_each_nth=2,
              epochs=1):
        '''GDM training

        Parameters
        ----------
        dataset: data_loader.DataLoader
        max_age: int
            Maximal age of edges
        num_context: int
            number of context vectors to use (K in original paper)
        learning_rates: list of float
            learning rates for the BMU and its neighbors
        a_threshold: list of float
            activation thresholds for G-EM and G-SM
        input_dimension: int
            input feature dimension
        only_each_nth: int
            How many frames to skip
        epochs: int
            How many epochs to train.
            For continuous learning it has to be 1

        '''
        # Init G-EM:
        g_episodic = EpisodicGWR()
        g_episodic.init_network(
            input_dimension=input_dimension,
            e_labels=2,
            num_context=num_context
        )
        # Init G-SM:
        if self.semantic:
            g_semantic = EpisodicGWR()
            g_semantic.init_network(
                input_dimension=input_dimension,
                e_labels=1,
                num_context=num_context
            )
        replay_size = (num_context * 2) + 1
        replay_weights = None
        # Iterate over dataset
        for x, y in tqdm(dataset):
            x, y = x[::only_each_nth], y[::only_each_nth]  # subsampling
            # G-EM train:
            g_episodic.train(
                ds_vectors=x,
                ds_labels=y,
                epochs=epochs,
                beta=0.7,
                l_rates=learning_rates,
                context=True,  # Whether to use context learning
                regulated=False,
                a_threshold=a_threshold[0],
                max_age=max_age
            )
            # G-SM train:
            if self.semantic:
                # Compute G-EM output and forward it to G-SM:
                e_x, e_y = g_episodic.test(
                    ds_vectors=x,
                    ds_labels=y,
                    ret_vecs=True
                )
                g_semantic.train(
                    ds_vectors=e_x,
                    ds_labels=e_y,
                    epochs=epochs,
                    beta=0.7,
                    l_rates=learning_rates,
                    context=True,  # Whether to use context learning
                    regulated=True,
                    a_threshold=a_threshold[1],
                    max_age=max_age
                )

            # Memory replay
            if self.replay and replay_weights is not None:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    g_episodic.train(
                        ds_vectors=replay_weights[r],
                        ds_labels=replay_labels[r, :],
                        epochs=epochs,
                        beta=0.7,
                        l_rates=learning_rates,
                        context=False,  # No context learning during replay
                        regulated=False,
                        a_threshold=a_threshold[0],
                        max_age=max_age,
                        verbose=False
                    )
                    if self.semantic:
                        rl = replay_labels[r, :, 1]
                        rl = rl.reshape(len(rl), 1)
                        g_semantic.train(
                            ds_vectors=replay_weights[r],
                            ds_labels=rl,
                            epochs=epochs,
                            beta=0.7,
                            l_rates=learning_rates,
                            context=False,
                            regulated=True,
                            a_threshold=a_threshold[1],
                            max_age=max_age,
                            verbose=False
                        )

            # Generate pseudo-samples
            if self.replay:
                replay_weights, replay_labels = self.replay_samples(
                    net=g_episodic,
                    size=replay_size
                )
