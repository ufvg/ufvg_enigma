from stable_baselines3 import PPO
from models.rl_agent import TestnetTradingEnv
from models.attention_model import build_transformer_model
import numpy as np

# Load Testnet data
lob_data = np.load('data/processed/testnet_lob.npy')

# Initialize environment
env = TestnetTradingEnv(lob_data, window_size=10)

# Custom policy with Transformer
policy_kwargs = {
    'features_extractor_class': build_transformer_model,
    'features_extractor_kwargs': {'input_shape': (10, 40)}
}

model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=50000)
model.save('testnet_transformer_agent')