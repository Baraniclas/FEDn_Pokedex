from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0 # keep track of training rounds
        self.lr = 0.1 # Initial learning rate

    def client_selection(self, client_ids: List[str]) -> List:
        # perform client selection
        pass

    def client_settings(self, global_model: List[np.ndarray]) -> Dict:
        # can be used to do costumization, example, decreasing learning rate every 10 rounds (see git serverfunctionsbase)
        pass

    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:

        pass