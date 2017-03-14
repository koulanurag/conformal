class Measure:
    """Abstract Measure base class."""

    def __init__(self):
        pass


class Ratio(Measure):
    def __init__(self):
        pass

    @staticmethod
    def measure(output, label):
        """
        The ratio of best best scoring label for not equal to ground truth label
         and the score for the ground truth label

        :param output: score for each label
        :param label: ground truth label
        :return: non-conformity measure
        """
        return max(list(output[0:label]) + list(output[label + 1:])) / output[label]


class Binary(Measure):
    def __init__(self):
        pass

    @staticmethod
    def measure(output, label):
        return 1 - output[label]


class Diff(Measure):
    def __init__(self):
        pass

    @staticmethod
    def measure(output, label):
        """The difference between the best scoring label for not equal to ground truth label
         and the score for the ground truth label

        :param output: score for each label
        :param label: ground truth label
        :return: non-conformity measure
        """
        return max(list(output[0:label]) + list(output[label + 1:])) - output[label]
