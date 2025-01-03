from pathlib import Path

from stable_baselines3 import SAC

class RlModel():
    def __init__(self,model_path):

        self.model = SAC.load(Path(__file__).parent / model_path)

    def get_action(self,obs):
        action, _states = self.model.predict(obs,deterministic=True)   

        return action    
    