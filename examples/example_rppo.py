import gymnasium as gym
import torch
from sb3_contrib import RecurrentPPO

# Create environment
env = gym.make('CartPole-v1')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create RecurrentPPO model
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    device=device,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=4,
)

print("Training RecurrentPPO for 2000 steps...")
model.learn(total_timesteps=2000)

print("Testing trained model...")
obs, _ = env.reset()
total_reward = 0

for i in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode finished with reward: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0

print("RecurrentPPO example completed!")
