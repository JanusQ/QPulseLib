from abc import ABC, abstractmethod


class Method(ABC):
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.time = 0

    @abstractmethod
    def data_loader(self):
        pass

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def gen_pulse(self, m):
        pass
