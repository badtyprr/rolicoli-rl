"""
Unit tests for training system
Tests RL algorithms, neural networks, and training infrastructure
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from pokemon_tcg_rl import CardDatabase, GameEngine, EnergyType
from main import (
    PokemonTCGEnv, PPOAgent, PolicyValueNetwork,
    PokemonGraphEncoder, TransformerEncoder, CardEmbedding,
    PrioritizedReplayBuffer, SelfPlayManager, DeckBuilder
)
from training import (
    PPOConfig, GPUOptimizedPPOAgent, GPUOptimizedPolicyValueNet,
    ExperimentTracker, Tournament, TrainingOrchestrator
)

@pytest.fixture
def env():
    """Create test environment"""
    # ... implementation as above ...
    return env


@pytest.fixture
def tournament(env):
    """Create tournament with environment"""
    return Tournament(env)


# In test_agents.py, add the trained_agent fixture:

@pytest.fixture
def trained_agent():
    """Create a partially trained agent"""
    config = PPOConfig(
        learning_rate=1e-3,
        rollout_length=32,
        batch_size=16,
        use_cuda=False,
        compile_model=False
    )

    agent = GPUOptimizedPPOAgent(config, state_dim=32, action_dim=100)

    # Do some mock training...
    return agent

class TestNeuralNetworks:
    """Test neural network architectures"""
    
    def test_policy_value_network(self):
        """Test basic policy-value network"""
        net = PolicyValueNetwork(state_dim=32, action_dim=100, hidden_dim=64)
        
        # Test forward pass
        state = torch.randn(16, 32)  # Batch of 16 states
        policy_logits, values = net(state)
        
        assert policy_logits.shape == (16, 100)
        assert values.shape == (16, 1)
        
    def test_gpu_optimized_network(self):
        """Test GPU-optimized network with mixed precision"""
        net = GPUOptimizedPolicyValueNet(state_dim=32, action_dim=100)
        
        if torch.cuda.is_available():
            net = net.cuda()
            state = torch.randn(16, 32).cuda()
            
            # Test with autocast
            with torch.cuda.amp.autocast():
                policy_logits, values = net(state)
            
            assert policy_logits.shape == (16, 100)
            assert values.shape == (16, 1)
            assert policy_logits.dtype == torch.float16  # Mixed precision
        
    def test_graph_encoder(self):
        """Test GNN encoder"""
        encoder = PokemonGraphEncoder(card_embedding_dim=32, hidden_dim=64, output_dim=128)
        
        # Create sample graph
        node_features = torch.randn(5, 42)  # 5 nodes, 42 features
        edge_index = torch.tensor([[0, 1, 2, 3, 4],
                                  [1, 2, 3, 4, 0]], dtype=torch.long)
        
        output = encoder(node_features, edge_index)
        assert output.shape == (1, 128)  # Global pooling output
        
    def test_transformer_encoder(self):
        """Test Transformer encoder"""
        encoder = TransformerEncoder(
            card_embedding_dim=32, 
            hidden_dim=64, 
            num_heads=4, 
            output_dim=128
        )
        
        # Create sample sequence
        card_embeddings = torch.randn(4, 10, 32)  # Batch=4, Seq=10, Embed=32
        output = encoder(card_embeddings)
        
        assert output.shape == (4, 128)
        
    def test_card_embeddings(self):
        """Test card embedding layer"""
        embeddings = CardEmbedding(num_cards=100, embedding_dim=32)
        
        # Register some cards
        from pokemon_tcg_rl import PokemonCard, CardType
        card = PokemonCard(
            name="Pikachu",
            set_code="TEST",
            number="1",
            card_type=CardType.POKEMON,
            hp=60,
            stage="Basic"
        )
        embeddings.register_card(card, 0)
        
        # Test embedding retrieval
        embedding = embeddings.get_card_embedding(card)
        assert embedding.shape == (1, 32)


class TestEnvironment:
    """Test Pokemon TCG environment"""

    @pytest.fixture
    def env(self):
        """Create test environment"""
        card_db = CardDatabase()

        # Create comprehensive card set
        sample_data = {
            "cards": []
        }

        # Add 20 different Pokemon
        for i in range(20):
            sample_data["cards"].append({
                "name": f"Pokemon {i}",
                "set_code": "TEST",
                "number": str(i),
                "card_type": "Pokemon",
                "hp": 60 + i * 10,
                "stage": "Basic",
                "attacks": [
                    {"name": "Attack", "damage": "20", "cost": "C"}
                ],
                "retreat_cost": 1
            })

        # Add 20 trainer cards
        for i in range(20):
            sample_data["cards"].append({
                "name": f"Trainer {i}",
                "set_code": "TEST",
                "number": str(100 + i),
                "card_type": "Trainer",
                "trainer_type": "Item",
                "effect": "Draw 1 card"
            })

        # Add 20 energy cards
        for i in range(20):
            sample_data["cards"].append({
                "name": "Lightning Energy",
                "set_code": "TEST",
                "number": str(200 + i),
                "card_type": "Energy",
                "energy_type": "Lightning"
            })

        card_db.load_from_json(sample_data)

        # Create environment with correct parameters
        env = PokemonTCGEnv(card_db, use_graph_encoder=False, max_steps=100)
        return env
    
    def test_reset(self, env):
        """Test environment reset"""
        obs = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (128,)
        assert env.current_step == 0
        
    def test_step(self, env):
        """Test environment step"""
        obs = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    def test_action_masking(self, env):
        """Test action masking"""
        env.reset()
        
        mask = env.get_action_mask(1)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        
        # At least some actions should be legal
        assert np.sum(mask) > 0
        
    def test_curriculum_learning(self, env):
        """Test curriculum learning stages"""
        env.reset()
        
        # Check initial stage
        assert env.curriculum_stage == 0
        stage_config = env.curriculum_stages[0]
        assert stage_config["special_conditions"] is False
        
        # Advance curriculum
        env.advance_curriculum()
        assert env.curriculum_stage == 1
        
    def test_reward_shaping(self, env):
        """Test reward shaping functionality"""
        env.reset()
        
        old_metrics = {
            "bench_size": 0,
            "total_energy": 0,
            "prizes_left": 6,
            "opp_prizes_left": 6,
            "hand_size": 7,
            "total_damage": 0
        }
        
        new_metrics = {
            "bench_size": 1,
            "total_energy": 1,
            "prizes_left": 6,
            "opp_prizes_left": 5,
            "hand_size": 7,
            "total_damage": 30
        }
        
        from pokemon_tcg_rl import Action, ActionType
        action = Action(ActionType.PLAY_BASIC_POKEMON, card_index=0)
        
        reward = env._compute_shaped_reward(old_metrics, new_metrics, action)
        
        # Should get positive reward for progress
        assert reward > 0


class TestPPOAgent:
    """Test PPO agent implementation"""
    
    def test_basic_ppo_agent(self):
        """Test basic PPO agent"""
        agent = PPOAgent(state_dim=32, action_dim=100)
        
        # Test action selection
        state = np.random.randn(32).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < 100
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
    def test_gpu_optimized_agent(self):
        """Test GPU-optimized PPO agent"""
        config = PPOConfig(
            learning_rate=3e-4,
            rollout_length=64,
            batch_size=16,
            use_cuda=torch.cuda.is_available()
        )
        
        agent = GPUOptimizedPPOAgent(config, state_dim=32, action_dim=100)
        
        # Test action selection
        state = np.random.randn(32).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        
        assert isinstance(action, int)
        
        # Test with action mask
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True  # Only first 10 actions allowed
        
        action, log_prob, value = agent.select_action(state, mask)
        assert 0 <= action < 10
        
    def test_ppo_update(self):
        """Test PPO update mechanism"""
        config = PPOConfig(
            rollout_length=32,
            batch_size=8,
            update_epochs=2,
            use_cuda=False  # CPU for testing
        )
        
        agent = GPUOptimizedPPOAgent(config, state_dim=32, action_dim=100)
        
        # Generate fake rollout
        for _ in range(32):
            state = np.random.randn(32).astype(np.float32)
            action = np.random.randint(0, 100)
            reward = np.random.randn()
            next_state = np.random.randn(32).astype(np.float32)
            done = False
            log_prob = -1.0
            value = 0.0
            
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
        
        # Update
        stats = agent.update()
        
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy_loss' in stats
        assert 'kl_divergence' in stats
        assert 'learning_rate' in stats
        
    def test_gae_computation(self):
        """Test GAE computation"""
        agent = PPOAgent(state_dim=32, action_dim=100)
        
        # Add some transitions
        for i in range(10):
            agent.rollout_buffer.append({
                'state': np.zeros(32),
                'action': 0,
                'reward': 1.0,
                'next_state': np.zeros(32),
                'done': i == 9,
                'log_prob': -1.0,
                'value': 0.5
            })
        
        advantages, returns = agent.compute_gae()
        
        assert len(advantages) == 10
        assert len(returns) == 10
        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)


class TestReplayBuffer:
    """Test prioritized replay buffer"""
    
    def test_buffer_operations(self):
        """Test buffer push and sample"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(32)
            action = i % 10
            reward = np.random.randn()
            next_state = np.random.randn(32)
            done = i % 10 == 0
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        
        # Sample batch
        batch, weights, indices = buffer.sample(10)
        
        assert len(batch) == 10
        assert len(weights) == 10
        assert len(indices) == 10
        
    def test_priority_updates(self):
        """Test priority updates"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(20):
            buffer.push(np.zeros(32), 0, 0, np.zeros(32), False)
        
        # Sample and update priorities
        batch, weights, indices = buffer.sample(5)
        td_errors = np.array([0.1, 0.5, 0.2, 0.3, 0.4])
        
        buffer.update_priorities(indices, td_errors)
        
        # Check priorities were updated
        assert buffer.max_priority > 1.0


class TestSelfPlay:
    """Test self-play manager"""

    @pytest.fixture
    def test_env(self):
        """Create test environment"""
        card_db = CardDatabase()
        # ... (same card creation code as in other tests)
        env = PokemonTCGEnv(card_db, use_graph_encoder=False, max_steps=100)
        return env

    @pytest.fixture
    def self_play_env(self, test_env):
        """Create self-play manager"""
        return SelfPlayManager(test_env, PPOAgent, population_size=3)
    
    def test_population_management(self, self_play_env):
        """Test adding agents to population"""
        initial_size = len(self_play_env.population)
        
        self_play_env.add_agent_to_population("test_agent")
        
        assert len(self_play_env.population) == initial_size + 1
        assert "test_agent" in self_play_env.elo_ratings
        
    def test_opponent_selection(self, self_play_env):
        """Test opponent selection mechanism"""
        # Add some agents
        for i in range(3):
            self_play_env.add_agent_to_population(f"agent_{i}")
        
        # Select opponent
        name, agent = self_play_env.select_opponent(exclude="agent_0")
        
        assert name != "agent_0"
        assert agent is not None
        
    def test_elo_updates(self, self_play_env):
        """Test ELO rating updates"""
        self_play_env.add_agent_to_population("winner")
        self_play_env.add_agent_to_population("loser")
        
        initial_winner_elo = self_play_env.elo_ratings["winner"]
        initial_loser_elo = self_play_env.elo_ratings["loser"]
        
        self_play_env.update_elo("winner", "loser")
        
        assert self_play_env.elo_ratings["winner"] > initial_winner_elo
        assert self_play_env.elo_ratings["loser"] < initial_loser_elo


class TestExperimentTracking:
    """Test experiment tracking functionality"""
    
    def test_experiment_tracker(self):
        """Test basic experiment tracking"""
        config = PPOConfig(
            use_tensorboard=False,  # Disable for testing
            use_wandb=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override experiment directory
            tracker = ExperimentTracker("test_exp", config)
            tracker.experiment_dir = Path(tmpdir) / "test_exp"
            tracker.experiment_dir.mkdir(parents=True)
            
            # Log some metrics
            metrics = {
                'loss': 0.5,
                'reward': 10.0,
                'episode_length': 50
            }
            
            tracker.log_metrics(metrics, step=100)
            
            # Log game metrics
            game_stats = {
                'win_rate': 0.6,
                'avg_game_length': 45,
                'avg_prizes_taken': 3.5
            }
            
            tracker.log_game_metrics(game_stats, step=200)
            
            tracker.close()
            
    def test_checkpoint_saving(self):
        """Test model checkpoint saving"""
        config = PPOConfig(use_tensorboard=False, use_wandb=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("test_exp", config)
            tracker.experiment_dir = Path(tmpdir) / "test_exp"
            tracker.experiment_dir.mkdir(parents=True)
            
            # Create dummy agent
            agent = GPUOptimizedPPOAgent(config, state_dim=32, action_dim=100)
            
            # Save checkpoint
            tracker.save_checkpoint(agent, step=1000)
            
            # Check file exists
            checkpoint_path = tracker.experiment_dir / "checkpoints" / "checkpoint_1000.pt"
            assert checkpoint_path.exists()
            
            # Load and verify
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['step'] == 1000
            assert 'model_state_dict' in checkpoint
            
            tracker.close()


class TestTournament:
    """Test tournament system"""
    
    @pytest.fixture
    def tournament(self, env):
        """Create tournament"""
        return Tournament(env)
    
    def test_add_agents(self, tournament):
        """Test adding agents to tournament"""
        from pokemon_tcg_rl import RandomAgent
        
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        
        tournament.add_agent("agent1", agent1)
        tournament.add_agent("agent2", agent2)
        
        assert len(tournament.agents) == 2
        assert "agent1" in tournament.elo_ratings
        
    def test_play_match(self, tournament):
        """Test playing matches"""
        from pokemon_tcg_rl import RandomAgent
        
        tournament.add_agent("agent1", RandomAgent())
        tournament.add_agent("agent2", RandomAgent())
        
        result = tournament.play_match("agent1", "agent2", num_games=2)
        
        assert 'wins' in result
        assert 'win_rate' in result
        assert 'elo_after' in result
        
        # Check ELO ratings were updated
        assert tournament.elo_ratings["agent1"] != 1500 or tournament.elo_ratings["agent2"] != 1500
        
    def test_leaderboard(self, tournament):
        """Test leaderboard generation"""
        from pokemon_tcg_rl import RandomAgent
        
        # Add agents with different ratings
        tournament.add_agent("agent1", RandomAgent())
        tournament.add_agent("agent2", RandomAgent())
        tournament.add_agent("agent3", RandomAgent())
        
        tournament.elo_ratings["agent1"] = 1600
        tournament.elo_ratings["agent2"] = 1400
        tournament.elo_ratings["agent3"] = 1500
        
        leaderboard = tournament.get_leaderboard()
        
        assert len(leaderboard) == 3
        assert leaderboard[0][0] == "agent1"
        assert leaderboard[2][0] == "agent2"


class TestTrainingOrchestrator:
    """Test training orchestrator"""
    
    def test_orchestrator_initialization(self, env):
        """Test orchestrator setup"""
        config = PPOConfig(
            rollout_length=32,
            use_tensorboard=False,
            use_wandb=False
        )
        
        orchestrator = TrainingOrchestrator(env, config)
        
        assert orchestrator.env is env
        assert orchestrator.config is config
        assert orchestrator.tournament is not None


def test_integration_mini_training(env):
    """Test mini training run"""
    config = PPOConfig(
        rollout_length=32,
        batch_size=8,
        update_epochs=1,
        use_tensorboard=False,
        use_wandb=False,
        use_cuda=False
    )
    
    agent = GPUOptimizedPPOAgent(config, 
                                state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.n)
    
    # Run a few episodes
    for episode in range(5):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            action_mask = env.get_action_mask(env.engine.state.current_player)
            action, log_prob, value = agent.select_action(obs, action_mask)
            
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done, log_prob, value)
            
            obs = next_obs
            steps += 1
        
        if len(agent.rollout_buffer) >= 32:
            stats = agent.update()
            assert stats  # Should have training stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])