"""
Unit tests for Pokemon TCG Gym environments
Tests the Gym wrappers and Stable Baselines3 integration
"""

import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

# Import after sys.path modification
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pokemon_tcg_gym_env import PokemonTCGGymEnv
from pokemon_tcg_draft_env import PokemonTCGDraftEnv
from pokemon_tcg_rl import CardDatabase, EnergyType
from train_sb3 import load_card_database


class TestPokemonTCGGymEnv:
    """Test the battle environment Gym wrapper"""

    @pytest.fixture
    def card_db(self):
        """Create test card database"""
        return load_card_database()

    @pytest.fixture
    def env(self, card_db):
        """Create test environment"""
        return PokemonTCGGymEnv(
            card_database=card_db,
            deck1_type=EnergyType.LIGHTNING,
            deck2_type=EnergyType.WATER,
            max_steps=100,
            reward_shaping=True
        )

    def test_env_creation(self, env):
        """Test environment can be created"""
        assert env is not None
        assert isinstance(env, gym.Env)

    def test_observation_space(self, env):
        """Test observation space is properly defined"""
        assert hasattr(env, 'observation_space')
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (32,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        """Test action space is properly defined"""
        assert hasattr(env, 'action_space')
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 200

    def test_reset(self, env):
        """Test environment reset"""
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == env.observation_space.dtype

        # Check observation bounds
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_step(self, env):
        """Test environment step"""
        obs = env.reset()

        # Take random action
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        # Check returns
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, (float, int))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check info contains expected keys
        assert 'current_player' in info
        assert 'turn_count' in info
        assert 'legal_actions_count' in info

    def test_multiple_episodes(self, env):
        """Test running multiple episodes"""
        for episode in range(3):
            obs = env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < 50:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

            # Episode should complete
            assert steps > 0
            assert isinstance(total_reward, (float, int))

    def test_render(self, env):
        """Test rendering methods"""
        env.reset()

        # Test ANSI rendering
        ansi_output = env.render(mode='ansi')
        assert isinstance(ansi_output, str)
        assert len(ansi_output) > 0

        # Test human rendering (should print)
        env.render(mode='human')

    def test_legal_action_mask(self, env):
        """Test legal action masking"""
        env.reset()

        mask = env.get_legal_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        assert np.any(mask)  # At least some actions should be legal

    def test_seed(self, env):
        """Test environment seeding"""
        # Set seed and run episode
        env.seed(42)
        obs1 = env.reset()
        actions1 = []
        for _ in range(5):
            actions1.append(env.action_space.sample())

        # Reset with same seed
        env.seed(42)
        obs2 = env.reset()
        actions2 = []
        for _ in range(5):
            actions2.append(env.action_space.sample())

        # Should get same initial observation
        np.testing.assert_array_equal(obs1, obs2)

    def test_reward_shaping(self, env):
        """Test reward shaping is applied"""
        env.reset()

        # Find action that should give shaped reward
        legal_actions = env.engine.get_legal_actions(1)
        play_pokemon_actions = [
            i for i, a in enumerate(legal_actions)
            if a.action_type.value == 'play_basic_pokemon'
        ]

        if play_pokemon_actions:
            _, reward, _, _ = env.step(play_pokemon_actions[0])
            # Should get positive shaped reward for playing Pokemon
            assert reward > 0

    def test_gym_checker(self, env):
        """Test environment passes Stable Baselines3 env checker"""
        # This will raise warnings/errors if environment is not properly implemented
        check_env(env, warn=True)


class TestPokemonTCGDraftEnv:
    """Test the draft environment Gym wrapper"""

    @pytest.fixture
    def card_db(self):
        """Create test card database"""
        return load_card_database()

    @pytest.fixture
    def env(self, card_db):
        """Create test draft environment"""
        return PokemonTCGDraftEnv(
            card_database=card_db,
            draft_format="booster",
            total_packs=2,
            cards_per_pack=5
        )

    def test_env_creation(self, env):
        """Test draft environment can be created"""
        assert env is not None
        assert isinstance(env, gym.Env)

    def test_observation_space(self, env):
        """Test draft observation space"""
        assert env.observation_space.shape == (42,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        """Test draft action space"""
        assert env.action_space.n == 5  # cards_per_pack

    def test_draft_episode(self, env):
        """Test complete draft episode"""
        obs = env.reset()
        done = False
        picks = 0

        while not done:
            # Pick first available card
            obs, reward, done, info = env.step(0)
            picks += 1

            # Safety check
            if picks > 20:
                break

        # Should complete draft
        assert done
        assert picks == 10  # 2 packs * 5 cards
        assert isinstance(reward, float)

    def test_gym_checker_draft(self, env):
        """Test draft environment passes env checker"""
        check_env(env, warn=True)


class TestStableBaselines3Integration:
    """Test SB3 integration with our environments"""

    @pytest.fixture
    def card_db(self):
        """Create test card database"""
        return load_card_database()

    def test_ppo_creation(self, card_db):
        """Test creating PPO with our environment"""
        env = PokemonTCGGymEnv(card_db)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=32,
            batch_size=8,
            verbose=0
        )

        assert model is not None

        # Test predict
        obs = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        assert isinstance(action, (int, np.integer))

        env.close()

    def test_recurrent_ppo_creation(self, card_db):
        """Test creating RecurrentPPO with our environment"""
        env = PokemonTCGGymEnv(card_db)

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=32,
            batch_size=8,
            verbose=0
        )

        assert model is not None

        # Test predict with LSTM states
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        assert isinstance(action, (int, np.integer))
        assert lstm_states is not None

        env.close()

    def test_training_step(self, card_db):
        """Test training for a few steps"""
        env = PokemonTCGGymEnv(card_db, max_steps=20)

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0
        )

        # Train for a few steps
        model.learn(total_timesteps=16, progress_bar=False)

        # Model should have trained
        assert model.num_timesteps == 16

        env.close()

    def test_vectorized_env(self, card_db):
        """Test with vectorized environments"""
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            return PokemonTCGGymEnv(card_db, max_steps=20)

        # Create vectorized environment
        vec_env = DummyVecEnv([make_env for _ in range(2)])

        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            n_steps=8,
            batch_size=4,
            verbose=0
        )

        # Test with vectorized env
        obs = vec_env.reset()
        assert obs.shape == (2, 32)  # 2 envs, 32 features

        action, _states = model.predict(obs)
        assert action.shape == (2,)

        vec_env.close()

    def test_save_load(self, card_db, tmp_path):
        """Test saving and loading models"""
        env = PokemonTCGGymEnv(card_db)

        # Create and train model
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=8,
            batch_size=4,
            verbose=0
        )

        model.learn(total_timesteps=16, progress_bar=False)

        # Save model
        save_path = tmp_path / "test_model"
        model.save(save_path)

        # Load model
        loaded_model = RecurrentPPO.load(save_path)

        # Test loaded model
        obs = env.reset()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)

        # Should predict same action
        assert action1 == action2

        env.close()


class TestCallbacks:
    """Test custom callbacks"""

    @pytest.fixture
    def card_db(self):
        """Create test card database"""
        return load_card_database()

    def test_tensorboard_callback(self, card_db):
        """Test custom TensorBoard callback"""
        from train_sb3 import TensorboardCallback

        env = PokemonTCGGymEnv(card_db, max_steps=20)
        model = PPO("MlpPolicy", env, n_steps=8, verbose=0)

        callback = TensorboardCallback()

        # Train with callback
        model.learn(
            total_timesteps=16,
            callback=callback,
            progress_bar=False
        )

        # Callback should have recorded some data
        assert callback.episode_count >= 0

        env.close()


def test_requirements():
    """Test all required packages are importable"""
    required = [
        'stable_baselines3',
        'sb3_contrib',
        'gymnasium',
        'torch',
        'numpy',
        'pandas',
        'tensorboard'
    ]

    for package in required:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])