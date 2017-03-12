class Measure:
    """Abstract Measure base class."""
    def __init__(self):
        pass


class SoftMax(Measure):
    def __init__(self):
        pass

    @staticmethod
    def measure(output, label):
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
        return max(list(output[0:label]) + list(output[label + 1:])) - output[label]