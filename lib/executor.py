from .apps.graph_builder.graph_builder import GraphBuilder
from .apps.data_processor.data_processor import DataProcessor

from .config.config import config


class Executor:
    def __init__(self):
        pass

    @staticmethod
    def run():
        data = DataProcessor().run()
        reduced_data = GraphBuilder(data=data).run()
