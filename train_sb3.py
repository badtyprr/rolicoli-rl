"""
Training script for Pokemon TCG using Stable Baselines3 with RecurrentPPO
Replaces all custom PyTorch training loops with SB3's built-in algorithms
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Stable Baselines3 imports
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

# Import our Gym environments
from pokemon_tcg_gym_env import PokemonTCGGymEnv
from pokemon_tcg_draft_env import PokemonTCGDraftEnv
from pokemon_tcg_rl import CardDatabase, EnergyType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for additional Pokemon TCG specific metrics
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log custom metrics from info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "winner" in info:
                    self.episode_count += 1
                    if info["winner"] == 1:
                        self.win_count += 1

                    # Log win rate
                    if self.episode_count > 0:
                        win_rate = self.win_count / self.episode_count
                        self.logger.record("pokemon/win_rate", win_rate)

                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

                    # Log episode metrics
                    self.logger.record("pokemon/episode_reward", info["episode"]["r"])
                    self.logger.record("pokemon/episode_length", info["episode"]["l"])

                    # Log action type distribution if available
                    if "action_type" in info:
                        self.logger.record(f"pokemon/action_{info['action_type']}", 1)

        return True


def make_env(card_database: CardDatabase,
             deck1_type: EnergyType = EnergyType.LIGHTNING,
             deck2_type: EnergyType = EnergyType.WATER,
             rank: int = 0,
             seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param card_database: Card database instance
    :param deck1_type: Player 1 deck energy type
    :param deck2_type: Player 2 deck energy type
    :param rank: Rank of the process
    :param seed: Random seed
    """

    def _init():
        env = PokemonTCGGymEnv(
            card_database=card_database,
            deck1_type=deck1_type,
            deck2_type=deck2_type,
            max_steps=200,
            reward_shaping=True,
            self_play=False
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def train_battle_agent(card_database: CardDatabase,
                       total_timesteps: int = 1_000_000,
                       n_envs: int = 4,
                       save_dir: str = "models/pokemon_tcg",
                       log_dir: str = "logs/pokemon_tcg",
                       eval_freq: int = 10_000,
                       checkpoint_freq: int = 50_000):
    """
    Train a RecurrentPPO agent on Pokemon TCG battles

    Args:
        card_database: Database of Pokemon cards
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation
        checkpoint_freq: Frequency of checkpoints
    """

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create vectorized training environments
    logger.info(f"Creating {n_envs} training environments...")
    env = make_vec_env(
        make_env(card_database, EnergyType.LIGHTNING, EnergyType.WATER),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )

    # Wrap with monitor for logging
    env = VecMonitor(env, log_dir)

    # Create evaluation environment
    eval_env = PokemonTCGGymEnv(
        card_database=card_database,
        deck1_type=EnergyType.LIGHTNING,
        deck2_type=EnergyType.FIRE,  # Different deck for evaluation
        max_steps=200,
        reward_shaping=True
    )

    # Configure RecurrentPPO hyperparameters
    # These approximate the original custom implementation
    model_kwargs = {
        "policy": "MlpLstmPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "n_steps": 256,  # Reduced from 2048 for LSTM
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,  # Don't use state dependent exploration
        "sde_sample_freq": -1,
        "policy_kwargs": {
            "lstm_hidden_size": 128,  # LSTM hidden state size
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # Separate networks
            "enable_critic_lstm": True,  # LSTM for critic too
            "ortho_init": True,
        },
        "tensorboard_log": log_dir,
        "verbose": 1,
    }

    logger.info("Initializing RecurrentPPO model...")
    model = RecurrentPPO(**model_kwargs)

    # Configure logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Create callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix="ppo_pokemon_tcg",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Custom metrics callback
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)

    # Combine callbacks
    callback = CallbackList(callbacks)

    # Train the model
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info(f"This will take approximately {total_timesteps / 20000:.1f} minutes")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=True,
            tb_log_name="RecurrentPPO",
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")

    # Close environments
    env.close()
    eval_env.close()

    return model


def train_draft_agent(card_database: CardDatabase,
                      total_timesteps: int = 500_000,
                      n_envs: int = 4,
                      save_dir: str = "models/pokemon_draft",
                      log_dir: str = "logs/pokemon_draft"):
    """
    Train a RecurrentPPO agent on Pokemon TCG Draft

    Args:
        card_database: Database of Pokemon cards
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    def make_draft_env(rank: int = 0, seed: int = 0):
        def _init():
            env = PokemonTCGDraftEnv(card_database=card_database)
            env.seed(seed + rank)
            return env

        return _init

    logger.info(f"Creating {n_envs} draft environments...")
    env = make_vec_env(
        make_draft_env(),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )

    # Wrap with monitor
    env = VecMonitor(env, log_dir)

    # Configure RecurrentPPO for draft
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=50,  # Shorter episodes in draft
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher exploration for draft
        policy_kwargs={
            "lstm_hidden_size": 64,
            "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
        },
        tensorboard_log=log_dir,
        verbose=1
    )

    # Train
    logger.info(f"Training draft agent for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        tb_log_name="RecurrentPPO_Draft",
        progress_bar=True
    )

    # Save model
    model.save(os.path.join(save_dir, "draft_model"))
    logger.info("Draft training complete!")

    env.close()

    return model


def load_card_database(filename: Optional[str] = None) -> CardDatabase:
    """Load card database from file or create sample data"""

    if filename and os.path.exists(filename):
        logger.info(f"Loading card database from {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return CardDatabase(json_data)
    else:
        logger.info("Creating sample card database for testing")
        # Create sample card data
        sample_data = {
            "cards": []
        }

        # Add Pokemon cards
        pokemon_names = ["Pikachu", "Charmander", "Squirtle", "Bulbasaur", "Eevee"]
        for i, name in enumerate(pokemon_names):
            for j in range(4):  # 4 copies of each
                sample_data["cards"].append({
                    "name": name,
                    "set_code": "TEST",
                    "number": str(i * 4 + j),
                    "card_type": "Pokemon",
                    "hp": 60 + i * 10,
                    "stage": "Basic",
                    "rarity": "Common",
                    "attacks": [
                        {"name": "Attack", "damage": str(20 + i * 10), "cost": "CC"}
                    ],
                    "weakness": "Fighting",
                    "retreat_cost": 1
                })

        # Add energy cards
        for i in range(30):
            sample_data["cards"].append({
                "name": "Lightning Energy",
                "set_code": "TEST",
                "number": str(100 + i),
                "card_type": "Energy",
                "energy_type": "Lightning"
            })

        # Add trainer cards
        trainers = ["Professor's Research", "Quick Ball", "Switch", "Potion"]
        for i, name in enumerate(trainers):
            for j in range(5):
                sample_data["cards"].append({
                    "name": name,
                    "set_code": "TEST",
                    "number": str(200 + i * 5 + j),
                    "card_type": "Trainer",
                    "trainer_type": "Item" if i > 0 else "Supporter",
                    "effect": "Draw cards" if i == 0 else "Item effect"
                })

        return CardDatabase(sample_data)


def evaluate_model(model_path: str, card_database: CardDatabase, n_episodes: int = 100):
    """
    Evaluate a trained model

    Args:
        model_path: Path to saved model
        card_database: Card database
        n_episodes: Number of evaluation episodes
    """

    logger.info(f"Loading model from {model_path}")
    model = RecurrentPPO.load(model_path)

    # Create evaluation environment
    env = PokemonTCGGymEnv(
        card_database=card_database,
        deck1_type=EnergyType.LIGHTNING,
        deck2_type=EnergyType.WATER,
        max_steps=200,
        reward_shaping=True
    )

    # Run evaluation
    wins = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            episode_starts = done

            if done and "winner" in info:
                if info["winner"] == 1:
                    wins += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{n_episodes}")

    # Print results
    win_rate = wins / n_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Episode Length: {avg_length:.1f}")
    logger.info("=" * 50)

    env.close()


def main():
    """Main training script"""

    print("╔══════════════════════════════════════════════════════╗")
    print("║     Pokemon TCG RL - Stable Baselines3 Training     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("This script trains RecurrentPPO agents for Pokemon TCG")
    print("Using LSTM policy for handling partial observability")
    print()

    # Load card database
    card_db = load_card_database("data/cards.json")  # Adjust path as needed
    logger.info(f"Loaded {len(card_db.cards)} cards")

    # Training configuration
    config = {
        "battle": {
            "total_timesteps": 1_000_000,
            "n_envs": 4,
            "eval_freq": 10_000,
            "checkpoint_freq": 50_000,
        },
        "draft": {
            "total_timesteps": 500_000,
            "n_envs": 4,
        }
    }

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train Pokemon TCG agents with SB3')
    parser.add_argument('--mode', type=str, default='battle', choices=['battle', 'draft', 'both'],
                        help='Training mode: battle, draft, or both')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to model for evaluation only')

    args = parser.parse_args()

    # Evaluation mode
    if args.eval:
        evaluate_model(args.eval, card_db)
        return

    # Override config if specified
    if args.timesteps:
        config["battle"]["total_timesteps"] = args.timesteps
        config["draft"]["total_timesteps"] = args.timesteps

    config["battle"]["n_envs"] = args.n_envs
    config["draft"]["n_envs"] = args.n_envs

    # Train models
    if args.mode in ['battle', 'both']:
        logger.info("Training battle agent...")
        battle_model = train_battle_agent(card_db, **config["battle"])

        # Quick evaluation
        logger.info("Evaluating battle agent...")
        evaluate_model("models/pokemon_tcg/best_model/best_model", card_db, n_episodes=20)

    if args.mode in ['draft', 'both']:
        logger.info("Training draft agent...")
        draft_model = train_draft_agent(card_db, **config["draft"])

    logger.info("Training complete!")
    logger.info("To monitor training: tensorboard --logdir logs/")
    logger.info("To evaluate: python train_sb3.py --eval models/pokemon_tcg/best_model/best_model")


if __name__ == "__main__":
    main()