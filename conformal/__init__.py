from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import measures
import os
import numpy as np
import matplotlib as mpl

mpl.use('Agg')  # to plot graphs over a server shell since the default display is not available on servers.
import matplotlib.pyplot as plt


class ConformalPrediction:
    def __init__(self, model_output, actual, epsilon=5, measure=measures.SoftMax(), threshold_mode=0):
        """

        :param model_output: output of the network for the validation data
        :param actual: list of ground truth labels
        :param epsilon:
        :param measure: object of Measure used for determining non-conformity score
        :param threshold_mode: It can be 0 or 1 . If 0,we have seperate thresholds for each label.And, If
                                1, one threshold for all the labels
        """
        if type(model_output).__module__ != np.__name__:
            raise TypeError('model_output must be numpy array')
        if type(actual).__module__ != np.__name__ and type(actual).__module__ != 'list':
            raise TypeError('actual must be a list')
        if epsilon < 0 or epsilon > 100:
            raise ValueError('epsilon should be between 0 and 100 ,both inclusive')
        if type(threshold_mode).__name__ != 'int':
            raise TypeError('threshold_mode must be an integer in [0,1]')
        if threshold_mode not in [0, 1]:
            raise ValueError('threshold_mode must be an integer in [0,1]')

        self.selected_threshold_mode = threshold_mode
        self.threshold_modes = [0, 1]
        self.measure = measure
        self.epsilon = epsilon
        self.labels = model_output.shape[1]
        self.thresholds = self.__non_conformity_thresholds(model_output.copy(), actual)

    @staticmethod
    def evaluate(predicted_labels, actual_labels):
        """ A prediction is considered to be correct if the actual label
           is present in the set of predicted labels

        :param predicted_labels: list of list of predicted labels
        :param actual_labels: ground truth label
        :return: accuracy
        """
        if type(predicted_labels).__module__ != np.__name__ and type(predicted_labels).__name__ != 'list':
            raise TypeError('predicted_labels must be 2D-numpy array or 2D-list ')
        if type(actual_labels).__module__ != np.__name__ and type(predicted_labels).__name__ != 'list':
            raise TypeError('actual_labels must be a list')

        count = 0
        for i, labels in enumerate(predicted_labels):
            if np.where(actual_labels[i] == 1)[0][0] in labels:
                count += 1
        return count / len(actual_labels)

    def label_histogram(self, predictions, save_path=None, title='Label Histogram'):
        """

        :param predictions:  predicted output of the Conformal Prediction
        :param save_path: absolute path to save the histogram image (optional)
        :param title: Title of the histogram image (optional)
        :return: histogram
        """
        labels_count = [len(_) for _ in predictions]
        histogram = np.histogram(labels_count, bins=self.labels)
        if save_path is not None:
            self.__save_histogram_plot(labels_count, save_path, title)
        return histogram

    def __non_conformity_thresholds(self, model_output, actual):
        """Calculates thresholds for corresponding labels using given epsilon and non-conformity measure

        :param model_output: model output for a batch ; each row contains output of one input
        :param actual: list of corresponding ground truth labels
        """
        measure = {label: [] for label in range(model_output.shape[1])}
        for i, output in enumerate(model_output):
            measure[np.where(actual[i] == 1)[0][0]].append(self.measure.measure(output, np.where(actual[i] == 1)[0][0]))

        thresholds = None
        if self.selected_threshold_mode == 0:
            thresholds = {label: np.percentile(measure[label], 100 - self.epsilon) for label in measure.keys()}
        elif self.selected_threshold_mode == 1:
            _percentile = np.percentile(np.concatenate([measure[label] for label in measure.keys()]), 100 - self.epsilon)
            thresholds = {label: _percentile for label in measure.keys()}

        return thresholds

    def predict(self, model_output):
        """A label is included in the set of predicted labels
           If the non-conformity measure for a label lies within the threshold of that label

        :param model_output: model output for a batch ; each row contains output of one input
        :return: list of predicted labels for the batch
        """
        if type(model_output).__module__ != np.__name__:
            raise TypeError('model_output must be 2D-numpy array')
        model_output = model_output.copy()
        predictions = []
        for output in model_output:
            predictions.append([i for i in range(len(output)) if self.measure.measure(output, i) <= self.thresholds[i]])
        return predictions

    @staticmethod
    def __save_histogram_plot(labels_count, plots_path, append_title):
        title = append_title if append_title is not None else "Histogram of Count of Labels in Prediction "
        plt.hist(labels_count, bins='auto')
        plt.grid(True)
        plt.title(title)
        plt.ylabel('# of predictions')
        plt.xlabel('# bins')
        plt.savefig(os.path.join(plots_path + ".png"))
        plt.clf()
