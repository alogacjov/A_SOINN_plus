##################################################################
# A-SOINN+ implementation based on SOINN+ Wiwatcharakoses et al. #
#                                                                #
# Wiwatcharakoses, C., & Berrar, D.P. (2020). SOINN+, a          #
# Self-Organizing Incremental Neural Network for Unsupervised    #
# Learning from Noisy Data Streams. Expert Syst. Appl., 143.     #
#                                                                #
# It further uses parts of Parisi et al.'s Growing Dual-Memory   #
# and GammaGWR implementation:                                   #
#                                                                #
# Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018)          #
# Lifelong Learning of Spatiotemporal Representations            #
# with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966 #
# https://github.com/giparisi/GDM                                #
#                                                                #
# @author: Aleksej Logacjov (a.logacjov@gmail.com)               #
#                                                                #
##################################################################

import numpy as np
from heapq import nsmallest
import math
import time
from tqdm import tqdm
from collections import deque
from models.GWR.gammagwr import GammaGWR


class ASOINNPlus(GammaGWR):

    def __init__(self):
        super().__init__()

    def init_network(self, input_dimension, num_context=2, e_labels=1):
        """ Initialize the A-SOINN+ network.

        Overrides GammaGWR's init_network
        It has all properties of the SOINN+ implementation with some
        additional features of the GammaGWR and GDM model:
        context descriptors, labels, temporal connections

        Parameters
        ----------
        input_dimension: int
            Dimension of the input data used later during training.
            E.g. 128-dimensional feature vectors
        num_context: int (optional)
            The number K of context vectors to consider. Default is 0
        e_labels: int (optional)
            Number of different types of labels to consider.
            If e.g. category and instance label desirable: e_labels=2
            Default is 1

        """
        assert self.iterations < 1, "Can't initialize a trained network"

        # Lock to prevent training
        self.locked = False

        self.num_context = num_context
        self.depth = self.num_context + 1
        self.dimension = input_dimension

        # Start with 3 neurons
        self.num_nodes = 3

        empty_neuron = np.zeros((self.depth, self.dimension))
        # These are the 3 nodes both with contextvector and weights
        self.weights = [empty_neuron.copy(),
                        empty_neuron.copy(),
                        empty_neuron.copy()]
        init_ind = list(range(0, self.num_nodes))
        # set weights of all initial nodes to random values:
        for i in range(0, len(init_ind)):
            self.weights[i][0] = np.random.rand(self.dimension)

        # Global context:
        self.g_context = np.zeros((self.depth, self.dimension))

        # Instead of maintaining a set of unutilities of deleted
        # nodes, its properties are maintained:
        # self.unutilities_del = []
        self.mean_unutility_del = None
        self.len_unutility_del = 0
        self.sum_unutility_del = 0

        # No deleted nodes so far:
        self.num_del_nodes = 0

        # Set of lifetimes of all deleted edges:
        # Adel <- empty set
        # We do not need to save the lifetimes explicitly.

        # Instead of maintaining a set of lifetimes of deleted
        # edges, its properties are maintained:
        self.mean_lifetimes_del = None
        self.len_lifetimes_del = 0
        self.sum_lifetimes_del = 0

        # Winning time for each newly created node:
        self.winning_times = [1, 1, 1]

        # Idle time of each neuron:
        self.idle_times = [0, 0, 0]

        # Trustworthiness init:
        self.trustworthiness = [None, None, None]

        # For each edge the lifetime:
        # -1 means no edge between these nodes:
        self.lifetimes = -np.ones((self.num_nodes, self.num_nodes))

        # Init list of similarity thresholds of connected nodes
        # to later compute arithmetic means and standard deviations of it.
        self.bmu_thetas = []
        self.sec_bmu_thetas = []

        # Temporal connections (to enable memory replay)
        # In the paper replay is not used in A-SOINN+
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels
        self.alabels = []  # associative matrix

        # Since we have no label at the beginning, we start
        # with an empty label dict
        for l in range(0, self.num_labels):
            alabels = []
            for neuron_idx in range(self.num_nodes):
                alabels.append({})
            self.alabels.append(alabels)

        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)


    def expand_matrix(self, matrix, value=0) -> np.array:
        '''Expands a matrix on both axes with the given value

        Overrides GammaGWR's expand_matrix to be able to decide
        which initial value to set in the appended dimensions.
        '''
        ext_matrix = np.hstack((matrix,
                                np.zeros((matrix.shape[0], 1))+value))
        ext_matrix = np.vstack((ext_matrix,
                                np.zeros((1, ext_matrix.shape[1]))+value))
        return ext_matrix

    def get_neighbors_of(self, index):
        """ Returns a list of neighbors of given node index. """
        # -1 demonstrates: 'no connection'
        return list(np.where(self.lifetimes[index] != -1)[0])

    def get_num_edges(self):
        """ Returns the number of lifetime edges. """
        # Since lifetimes contains undirected edges we have
        # to divide the number of edges by 2:
        return len(np.where(self.lifetimes != -1)[0]) / 2

    def calc_similarity_thresholds(self, bmu, sec_bmu):
        """ Calculates the similarity thresholds of the BMU and sec BMU.

        The distance measurement is the euclidean distance.
        But it also considers the context descriptors of the weights.

        Parameters
        ----------
        bmu: int
            Index of BMU
        sec_bmu: int
            Index of second BMU

        Returns
        -------
        tuple of float
            The similarity thresholds for BMU and secBMU

        """
        similarity_thresholds = []
        for node in [bmu, sec_bmu]:
            neighbors = self.get_neighbors_of(node)
            if len(neighbors) != 0:
                # Get the neighbor with the highest distance
                # to BMU or secBMU:
                max_dist_to_neighbor = 0
                for n in neighbors:
                    # Weights consider also the context descriptors:
                    node_weights = self.weights[node]
                    neighbor_weights = self.weights[n]
                    eucl_dist = self.compute_distance(node_weights,
                                                         neighbor_weights)
                    if eucl_dist > max_dist_to_neighbor:
                        max_dist_to_neighbor = eucl_dist
                similarity_thresholds.append(max_dist_to_neighbor)
            else:
                # If the node has no neighbors to compare to
                # check which of all nodes is the nearest:
                min_dist_to_unit = math.inf
                # Go trough all nodes n with weights n_weights:
                for n, n_weights in enumerate(self.weights):
                    if n == node:
                        continue
                    node_weights = self.weights[node]
                    eucl_dist = self.compute_distance(node_weights,
                                                         n_weights)
                    if eucl_dist < min_dist_to_unit:
                        min_dist_to_unit = eucl_dist
                similarity_thresholds.append(min_dist_to_unit)
        return similarity_thresholds

    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        '''Updates temporal connections. Used in memory replay.'''
        # Required if using replay in SOINN+
        new_node = kwargs.get('new_node', False)
        if new_node:
            # Expand to consider new neuron but connections
            # are first set to 0
            self.temporal = self.expand_matrix(self.temporal, value=0)
        if previous_ind != -1 and previous_ind != current_ind:
            # Here the previous activated neuron and the current
            # activated (or newly created) are considered so
            # the temporal connection between both is incremented
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        """Updating the associative matrix

        Overrides GammaGWR's update_labels
        """
        new_node = kwargs.get('new_node', False)
        # Check whether the label already exists and create if not:
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

    def update_weight(self, index, epsilon) -> None:
        """ Is updating the weights and the context vectors.

        Overrides GammaGWR's update_weight
        Called 'Node Merging' in SOINN+.

        Parameters
        ----------
        index: int
            Index of neuron
        epsilon: float
            learning rate of BMU

        """
        delta = np.dot((self.g_context - self.weights[index]),
                       (epsilon * (1/self.winning_times[index])))
        self.weights[index] = self.weights[index] + delta

    def update_neighbors(self, index, epsilon) -> None:
        """ Updating the weights/contexts of all neighbors of BMU.

        Overrides GammaGWR's update_neighbors
        Called 'Node Merging' in SOINN+.

        Parameters
        ----------
        index: int
            Index of neuron
        epsilon: float
            Learning rate of neighbours

        """
        b_neighbors = self.get_neighbors_of(index)
        for neIndex in b_neighbors:
            # weights of neighbour, winning time of BMU
            delta = np.dot((self.g_context - self.weights[neIndex]),
                           (epsilon * (1/self.winning_times[index])))
            self.weights[neIndex] = self.weights[neIndex] + delta

    def node_linking(self, b_index, s_index,
                     b_similarity_th, s_similarity_th):
        """ Creates edges between the BMU and secBMU.

        But only if certain conditions are given.
        The trustworthiness of each node is updated here.
        Lifetime of new edge is set to 0.
        The lifetime of to BMU connected edges is increased.
        The graph is a undirected graph, hence 2 entries are
        created in the edge adjacent matrix.

        Parameters
        ----------
        b_index: int
            Index of BMU
        s_index: int
            Index of second BMU
        b_similarity_th: float
            Similarity threshold of BMU
        s_similarity_th: float
            Similarity threshold of second BMU

        """
        max_WT = max(self.winning_times)
        # Determine trustworthiness of each node
        for node_idx, _ in enumerate(self.weights):
            self.trustworthiness[node_idx] = (self.winning_times[node_idx]-1)/ \
                                             (max_WT-1)
        num_edges = self.get_num_edges()
        theta_1 = np.mean(self.bmu_thetas)
        theta_2 = np.mean(self.sec_bmu_thetas)
        sigma_1 = np.std(self.bmu_thetas)
        sigma_2 = np.std(self.sec_bmu_thetas)

        if num_edges < 3 or \
           b_similarity_th*(1-self.trustworthiness[b_index])<theta_1+2*sigma_1 or \
           s_similarity_th*(1-self.trustworthiness[s_index])<theta_2+2*sigma_2:
            # If no edge between BMU and secBMU:
            edge_exists = self.lifetimes[b_index, s_index] != -1
            if not edge_exists:
                # Create edge with lifetime of 0:
                self.lifetimes[b_index, s_index] = 0
                self.lifetimes[s_index, b_index] = 0
                # Update list of similarity thresholds:
                self.bmu_thetas.append(b_similarity_th)
                self.sec_bmu_thetas.append(s_similarity_th)
        # If an edge exists reset the lifetime:
        edge_exists = self.lifetimes[b_index, s_index] != -1
        if edge_exists:
            self.lifetimes[b_index, s_index] = 0
            self.lifetimes[s_index, b_index] = 0
        # Increment all to BMU connected edges by 1:
        b_neighbors = self.get_neighbors_of(b_index)
        for neIndex in b_neighbors:
            self.lifetimes[b_index, neIndex] += 1
            self.lifetimes[neIndex, b_index] += 1

    def node_deletion(self):
        '''Deletes isolated nodes with a high unutilty.'''
        # Unutilities of all connected neurons:
        unutilities_with_edges = []
        num_unconnected_nodes = 0
        for node_idx, _ in enumerate(self.weights):
            # The higher idle time compared to the winning time,
            # the higher the unutility of a node:
            node_unutility = self.idle_times[node_idx] / \
                             self.winning_times[node_idx]
            n_neighbors = self.get_neighbors_of(node_idx)
            # If node has at least one edge:
            if len(n_neighbors) > 0:
                unutilities_with_edges.append(node_unutility)
            else:
                num_unconnected_nodes += 1
        # If they are equal, there are no edges in the network:
        if num_unconnected_nodes != len(self.weights):
            unutilities_with_edges = np.array(unutilities_with_edges)
            # Measure of dispersion. Robust against outliers:
            smad = np.median(np.abs(unutilities_with_edges - \
                                    np.median(unutilities_with_edges)))
            smad = 1.4826 * smad
            # Measure of outlierness of an unutility:
            omega_node = np.median(unutilities_with_edges) + 2*smad
            # Ratio of unconnected nodes to all nodes:
            R_noise = num_unconnected_nodes / len(self.weights)
            # removed units ratio:
            l_S = len(self.weights)
            l_I = num_unconnected_nodes
            l_B_del = self.num_del_nodes
            # Ratio of deleted nodes and all connected nodes +
            # deleted nodes.
            B_del_ratio = l_B_del/(l_B_del+(l_S-l_I))
            if self.len_unutility_del !=0:
                lambda_node = self.mean_unutility_del * B_del_ratio
            else:
                # If there are no deleted edges, the first part
                # of the equation of lambda_edge will be 0:
                lambda_node = 0
            # outlierness threshold for nodes:
            lambda_node = lambda_node + \
                          omega_node * (1 - B_del_ratio) * (1 - R_noise)
            node_idx = 0
            # Since we are removing nodes during runtime we have to
            # ensure that the correct node_index is set. Therefore
            # the node_index is only incremented if a node is not
            # deleted:
            while node_idx < self.num_nodes:
                # index of node is changing unutility and unconnected nodes
                # indexing only possible with new calculation:
                node_unutility = self.idle_times[node_idx] / \
                                 self.winning_times[node_idx]
                n_neighbors = self.get_neighbors_of(node_idx)
                # If unutility too high and node has no edges:
                if node_unutility > lambda_node and \
                   len(n_neighbors) == 0:
                    self.num_del_nodes += 1
                    # Increment the new sum of deleted unutilities
                    # by the current to delete node:
                    self.sum_unutility_del += node_unutility
                    # Increment length of deleted unutilities:
                    self.len_unutility_del += 1
                    # Calculate the new mean of deleted unutilities:
                    self.mean_unutility_del = self.sum_unutility_del / \
                                              self.len_unutility_del
                    # pop from weights:
                    self.weights.pop(node_idx)
                    # The further removing steps are ensuring that
                    # indexing of nodes is working correctly
                    # after deletion procedure:
                    # Remove index from alabels:
                    for l in range(0, self.num_labels):
                        self.alabels[l].pop(node_idx)
                    # Remove from temporal connections on both axis:
                    self.temporal = np.delete(self.temporal,
                                              node_idx, axis=0)
                    self.temporal = np.delete(self.temporal,
                                              node_idx, axis=1)
                    # Adjust previous BMU:
                    if self.previous_ind > node_idx:
                        self.previous_ind -= 1
                        self.previous_bmu = self.weights[self.previous_ind]
                    elif self.previous_ind == node_idx:
                        # Remove previous BMU:
                        self.previous_ind = -1
                        self.previous_bmu = np.zeros((self.depth,
                                                      self.dimension))
                    # Remove from idle_times and winning_times:
                    self.idle_times.pop(node_idx)
                    self.winning_times.pop(node_idx)
                    # Remove from thrustworthiness:
                    self.trustworthiness.pop(node_idx)
                    # Remove from lifetime connections on both axis:
                    self.lifetimes = np.delete(self.lifetimes,
                                               node_idx, axis=0)
                    self.lifetimes = np.delete(self.lifetimes,
                                               node_idx, axis=1)
                    self.num_nodes -= 1
                    self.num_deleted_nodes_in_batch += 1
                else:
                    node_idx += 1
        # Update idle times:
        for node_idx, _ in enumerate(self.weights):
            self.idle_times[node_idx] += 1

    def edge_deletion(self, b_index):
        """ Deletes edges if the lifetime is greater than threshold.

        There are only edges considered that are connected to the
        BMU. The threshold is determined by the outlierness of an
        edge lifetime (calculated with IQR) and the mean of deleted
        lifetimes.

        Parameters
        ----------
        b_index: int

        """
        lifetimes_edges_to_b_index = []
        reachable_edges = self.get_all_reachable_edges(b_index)
        # No edges to delete if there are non:
        if len(reachable_edges) == 0:
            return
        # Lifetimes of all reachable edges of BMU
        for edge in reachable_edges:
            x, y = edge
            lifetime = self.lifetimes[x, y]
            lifetimes_edges_to_b_index.append(lifetime)
        A = lifetimes_edges_to_b_index
        A75 = np.quantile(A, .75)
        A25 = np.quantile(A, .25)
        A_IQR = A75 - A25
        # Outlierness of lifetimes:
        # A value greater than this, is probably an outlier:
        omega_edge = A75 + 2*A_IQR
        # How many removed edges in contrast to all edges
        # deleted and not deleted but reachable from BMU:
        A_del_ratio = ((self.len_lifetimes_del)/((self.len_lifetimes_del)+len(A)))
        # The mean is the same in directed as un undirected
        # graph since the edge lifetimes are equal in both directions.
        if self.len_lifetimes_del !=0:
            lambda_edge = self.mean_lifetimes_del * A_del_ratio
        else:
            # If there are no deleted edges, the first part
            # of the equation of lambda_edge will be 0:
            lambda_edge = 0
        lambda_edge = lambda_edge + omega_edge * (1-A_del_ratio)
        b_neighbors = self.get_neighbors_of(b_index)
        for neIndex in b_neighbors:
            if self.lifetimes[b_index, neIndex] > lambda_edge:
                # Increment the new sum of deleted lifetimes by the
                # current to delete lifetime:
                self.sum_lifetimes_del += self.lifetimes[b_index, neIndex]
                # Increment length of deleted lifetimes:
                self.len_lifetimes_del += 1
                # Calculate the new mean of deleted lifetimes:
                self.mean_lifetimes_del = self.sum_lifetimes_del / \
                                          self.len_lifetimes_del
                # Delete corresponding lifetimes for an undirected
                # connection:
                self.lifetimes[b_index, neIndex] = -1
                self.lifetimes[neIndex, b_index] = -1
                self.num_deleted_edges_in_batch += 1

    def get_all_reachable_edges(self, index):
        """ Returns all edges that can be reached from given node.

        Using Breadth First Search.

        Parameters
        ----------
        index: int
            node index

        Returns
        -------
        list of tuple
            List of edges

        """
        edges = []
        seen_indices = []
        remaining_indices = deque([index])
        while len(remaining_indices) != 0:
            current_idx = remaining_indices.popleft()
            seen_indices.append(current_idx)
            neighbors = self.get_neighbors_of(current_idx)
            for n in neighbors:
                if n not in seen_indices:
                    # Considering of only one direction since the
                    # other direction has the same properties:
                    edge = [current_idx, n]
                    edges.append(edge)
                    remaining_indices.append(n)
        return edges

    def get_label_of_neuron(self, neuron_idx, label_level=1):
        """ Calculates the label of a neuron with the most counts

        Parameters
        ----------
        neuron_idx: int
        label_level: int, optional
            if multiple class levels (e.g. category+instance)
            0 = instance, 1 = category, (default 1)

        Returns
        -------
        int
            index of the label

        """

        if len(self.alabels[label_level][neuron_idx]) == 0:
            # return 0
            # since first label is 0
            return -1
        return max(self.alabels[label_level][neuron_idx],
                   key=self.alabels[label_level][neuron_idx].get)

    def find_bs(self, dis):
        '''Finds BMU and sBMU with corresponding distances

        Overrides GammaGWR's find_bs to extract the sBMU's
        distance as well.
        '''
        bs = nsmallest(2, ((k, i) for i, k in enumerate(dis)))
        return bs[0][1], bs[0][0], bs[1][1], bs[1][0]

    def find_bmus(self, input_vector, **kwargs):
        '''Get the first and second BMU

        Overrides GammaGWR's find_bmus in order to return the
        distance to the second BMU as well
        '''
        second_best = kwargs.get('s_best', False)
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(self.weights[i],
                                                    input_vector)
        if second_best:
            # Compute best and second-best matching units
            return self.find_bs(distances)
        else:
            b_index = distances.argmin()
            b_distance = distances[b_index]
            return b_index, b_distance

    def train_step(self, ds_vectors, ds_labels, epochs, beta,
              l_rates, context, creation_constraint=True,
              adaptation_constraint=True,
              verbose=True) -> None:
        """One training step of the network

        Parameters
        ----------
        ds_vectors : numpy.array
            Training data
        ds_labels : numpy.array
            Labels of training data
        epochs : int
        beta : float
            Parameter for context learning
        l_rates : list of float
            Learning rates of BMU and its neighbors
            If working with one layered SOINN+, put [float, None]
        context : bool
            Whether to perform context learning
        creation_constraint, adaptation_constraint : bool, bool
            Whether to utilize the additional weight adaptation
            and node creation constraints proposed in the paper.

        """
        assert not self.locked, "Network is locked. Unlock to train."

        num_correct_classified = 0
        num_new_add_neurons = 0
        num_updated_weights = 0

        self.samples = ds_vectors.shape[0]
        self.max_epochs = epochs
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta
        self.creation_constraint = creation_constraint
        self.adaptation_constraint = adaptation_constraint
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        # If neuron has hab_counter<hab_threshold => no training
        # hab_counter is decreasing over time (each time fired)
        self.hab_threshold = 0.1
        # tau determines how fast the hab_counter is decreased
        # of BMU faster than on neighbor
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = math.inf
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        # Decreasing the learning rate by multiplication with this value
        self.mod_rate = 1

        # Start training
        error_counter = np.zeros(self.max_epochs)
        num_edges_before = self.get_num_edges()
        self.num_deleted_nodes_in_batch = 0
        self.num_deleted_edges_in_batch = 0
        self.previous_bmu = np.zeros((self.depth, self.dimension))
        self.previous_ind = -1
        for epoch in range(0, self.max_epochs):
            for iteration in range(0, self.samples):

                # Generate input sample
                # Remember (see above): input is 0'th global context vector
                self.g_context[0] = ds_vectors[iteration]

                label = ds_labels[iteration]

                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = self.beta * self.previous_bmu[z]
                    self.g_context[z] += (1-self.beta) * self.previous_bmu[z-1]

                # Find the best and second-best matching neurons
                bmus = self.find_bmus(self.g_context, s_best=True)
                b_index, b_distance, s_index, s_distance = bmus
                if creation_constraint:
                    b_label = self.get_label_of_neuron(b_index, -1)
                    misclassified = b_label != label[-1]
                else:
                    b_label = self.get_label_of_neuron(b_index, 0)
                    misclassified = b_label != label[0]

                if not misclassified:
                    num_correct_classified += 1

                # Quantization error
                error_counter[epoch] += b_distance

                # Store BMU at time t for t+1
                self.previous_bmu = self.weights[b_index]

                # Similarity thresholds of n1 and n2:
                similarity_thresholds = self.calc_similarity_thresholds(
                    b_index,
                    s_index
                )
                b_similarity_th, s_similarity_th = similarity_thresholds

                if (b_distance >= b_similarity_th or \
                    s_distance >= s_similarity_th) and \
                    ((not self.creation_constraint) or \
                     (self.creation_constraint and misclassified)):
                    num_new_add_neurons += 1
                    n_index = self.num_nodes
                    # Add input as new neuron:
                    self.weights.append(self.g_context.copy())
                    self.num_nodes += 1
                    # Expand the lifetimes matrix with -1 entries:
                    self.lifetimes = self.expand_matrix(
                        self.lifetimes,
                        value=-1
                    )

                    # Append the winning time of the new node (1):
                    self.winning_times.append(1)

                    # Appending idle time of new neuron:
                    self.idle_times.append(0)

                    # Create new trustworthiness entry:
                    self.trustworthiness.append(None)

                    # Add label histogram
                    self.update_labels(n_index, label,
                                       new_node=True)

                    # Append new neuron to temporal connections:
                    self.update_temporal(n_index, self.previous_ind,
                                         new_node=True)

                else:
                    # Update BMU's label histogram
                    self.update_labels(b_index, label)

                    # Update BMU's weight vector
                    # The learning reates of b and n with
                    # b_rate > n_rate:
                    b_rate, n_rate = self.epsilon_b, self.epsilon_n
                    # decreasing learning rate
                    if self.adaptation_constraint and misclassified:
                        self.epsilon_b *= self.mod_rate
                        self.epsilon_n *= self.mod_rate
                        b_rate, n_rate = self.epsilon_b, self.epsilon_n
                    if (not self.adaptation_constraint) or \
                       (self.adaptation_constraint and not misclassified):
                        num_updated_weights += 1
                        # Update winning time of BMU:
                        self.winning_times[b_index] += 1

                        # Update weight and context of BMU:
                        self.update_weight(b_index, b_rate)

                        # Check whether to link BMU and secBMU:
                        self.node_linking(b_index,
                                          s_index,
                                          b_similarity_th,
                                          s_similarity_th)

                        # Update temporal connections
                        # The temporal connection is a directed connection:
                        self.update_temporal(b_index, self.previous_ind)

                        # Update weights and contexts of BMU's neighbors
                        self.update_neighbors(b_index, n_rate)

                        # Reset the idle time of BMU
                        self.idle_times[b_index] = 0

                        # Remove old edges
                        self.edge_deletion(b_index)

                self.previous_ind = b_index
                self.node_deletion()

                self.iterations += 1

            num_edges_after = self.get_num_edges()
            error_counter[epoch] /= self.samples
            if verbose:
                print(("(Epoch: %s, NN: %s, ATQE: %s, " +
                       "Num_correct_classified: %s, " +
                       "num_neurons_added: %s, " +
                       "num_updated_weights: %s, " +
                       "num_edges before: %s, " +
                       "num_edges after: %s, " +
                       "num_deleted_edges_in_batch: %s, " +
                       "num deleted nodes: %s)") % 
                      (epoch + 1,
                       self.num_nodes,
                       error_counter[epoch],
                       num_correct_classified,
                       num_new_add_neurons,
                       num_updated_weights,
                       num_edges_before,
                       num_edges_after,
                       self.num_deleted_edges_in_batch,
                       self.num_deleted_nodes_in_batch))

    def train(self,
              dataset,
              test_dataset=None,
              num_context=2,
              learning_rates=[0.5, 0.005],
              input_dimension=128,
              only_each_nth=2,
              epochs=1,
              creation_constraint=True,
              adaptation_constraint=True,
              logger=None):
        '''A-SOINN+ training

        Parameters
        ----------
        dataset: data_loader.DataLoader
        num_context: int
            number of context vectors to use (K)
        learning_rates: list of float
            learning rates for the BMU and its neighbors
        input_dimension: int
            input feature dimension
        only_each_nth: int
            How many frames to skip
        epochs: int
            How many epochs to train.
            For continuous learning it has to be 1
        creation_constraint, adaptation_constraint : bool, bool
            Whether to utilize the additional weight adaptation
            and node creation constraints proposed in the paper.
        logger: utils.logger.Logger
            Whether to log test results

        '''
        self.init_network(
            input_dimension=input_dimension,
            num_context=num_context,
            e_labels=2)
        # Iterate over dataset
        for x, y in tqdm(dataset):
            x, y = x[::only_each_nth], y[::only_each_nth]  # subsampling
            self.train_step(
                ds_vectors=x,
                ds_labels=y,
                epochs=epochs,
                beta=0.7,
                l_rates=learning_rates,
                context=True,
                creation_constraint=creation_constraint,
                adaptation_constraint=adaptation_constraint
            )
            if test_dataset is not None and logger is not None:
                xts, yts = test_dataset.__next__()
                # Get the category accuracies of each test subset
                accs = [self.test(xt,yt)[2][1] for xt,yt in zip(xts, yts)]
                logger.log(accs)


    def test(self, ds_vectors, ds_labels):
        """Test the network on a given testset.

        Parameters
        ----------
        ds_vectors : np.array
            input x
        ds_labels : np.array
            labels y

        Returns
        -------
        np.array, np.array, np.array
            BMU's weights and corresponding labels as well as accuracies

        """
        test_samples = ds_vectors.shape[0]
        # F.e. test sample save which was/is the BMU. Init: -1:
        self.bmus_index = -np.ones(test_samples)
        # A weight vector f.e. test sample. Init: 0:
        self.bmus_weight = np.zeros((test_samples, self.dimension))
        # Each test sample can have mult. labels (categorie & instance):
        self.bmus_label = -np.ones((self.num_labels, test_samples))
        # same dimension like global context or weights matrix:
        input_context = np.zeros((self.depth, self.dimension))

        acc_counter = np.zeros(self.num_labels)
        avrg_b_dist = 0
        for i in range(0, test_samples):
            # Again first entry in context matrix is the input itself
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = self.find_bmus(input_context)
            avrg_b_dist += b_distance
            # Fill the arrays, defined above
            self.bmus_index[i] = b_index
            self.bmus_weight[i] = self.weights[b_index][0]
            for l in range(0, self.num_labels):
                self.bmus_label[l, i] = self.get_label_of_neuron(
                    b_index,
                    label_level=l
                )
            input_context_copy = input_context.copy()
            for j in range(1, self.depth):
                input_context_copy[j] = input_context[j-1]
            input_context = input_context_copy
            for l in range(0, self.num_labels):
                if self.bmus_label[l, i] == ds_labels[i, l]:
                    acc_counter[l] += 1
        acc = acc_counter / ds_vectors.shape[0]
        print('B distance during test: ', avrg_b_dist/test_samples)
        s_labels = -np.ones((test_samples, 1))
        s_labels[:, 0] = ds_labels[:, 1]
        return self.bmus_weight, s_labels, acc
