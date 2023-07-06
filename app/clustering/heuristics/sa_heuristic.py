import time

from .IHeuristic import IHeuristic
from ...models import GraphObject, DataProps
from ...services import log_service


class SAHeuristic(IHeuristic):
    def __init__(self):
        super().__init__()

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int):
        start_time = time.time()

        end_time = time.time()
