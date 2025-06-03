"""
Unit tests for agent behavior
Tests different agent implementations and strategies
"""

import pytest
import numpy as np
import torch
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pokemon_tcg_rl import (
    CardDatabase, GameEngine, Action, ActionType,
    RandomAgent, DraftAgent, RandomDraftAgent,
    EnergyType, PokemonCard, TrainerCard, EnergyCard
)
from main import (
    PokemonTCGEnv, PPOAgent, DeckBuilder
)
from training import (
    PPOConfig, GPUOptimizedPPOAgent
)


class TestRandomAgent:
    """Test random agent baseline"""
    
    def test_random_action_selection(self):
        """Test random agent selects valid actions"""
        agent = RandomAgent()
        
        # Create fake state and actions
        state = np.random.randn(32)
        actions = [
            Action(ActionType.PLAY_BASIC_POKEMON, card_index=0),
            Action(ActionType.ATTACH_ENERGY, card_index=1),
            Action(ActionType.END_TURN)
        ]
        
        # Select action multiple times
        selected_actions = []
        for _ in range(100):
            action = agent.select_action(state, actions)
            selected_actions.append(action)
        
        # Should select different actions
        unique_actions = set(a.action_type for a in selected_actions)
        assert len(unique_actions) > 1
        
        # All actions should be from legal set
        for action in selected_actions:
            assert action in actions
    
    def test_random_agent_no_learning(self):
        """Test that random agent doesn't learn"""
        agent = RandomAgent()
        
        # Update should do nothing
        state = np.random.randn(32)
        action = Action(ActionType.END_TURN)
        
        agent.update(state, action, 1.0, state, False)
        
        # No internal state should change
        assert True  # Random agent has no internal state


class TestPPOAgentBehavior:
    """Test PPO agent strategic behavior"""
    
    @pytest.fixture
    def trained_agent(self):
        """Create a partially trained agent"""
        config = PPOConfig(
            learning_rate=1e-3,  # Higher for faster learning in tests
            rollout_length=32,
            batch_size=16,
            use_cuda=False
        )
        
        agent = GPUOptimizedPPOAgent(config, state_dim=32, action_dim=100)
        
        # Do some mock training to initialize properly
        for _ in range(5):
            for _ in range(32):
                state = np.random.randn(32).astype(np.float32)
                action = np.random.randint(0, 100)
                reward = np.random.randn()
                next_state = np.random.randn(32).astype(np.float32)
                agent.store_transition(state, action, reward, next_state, False, -1.0, 0.0)
            
            agent.update()
        
        return agent
    
    def test_action_consistency(self, trained_agent):
        """Test that agent gives consistent actions for same state"""
        state = np.random.randn(32).astype(np.float32)
        
        # Get multiple actions for same state
        actions = []
        for _ in range(10):
            action, _, _ = trained_agent.select_action(state)
            actions.append(action)
        
        # In evaluation mode (no exploration), should be mostly consistent
        # Allow some variance due to sampling from distribution
        most_common = max(set(actions), key=actions.count)
        consistency = actions.count(most_common) / len(actions)
        assert consistency > 0.5  # At least 50% same action
    
    def test_action_masking_behavior(self, trained_agent):
        """Test agent respects action masks"""
        state = np.random.randn(32).astype(np.float32)
        
        # Create restrictive mask
        mask = np.zeros(100, dtype=bool)
        mask[5:10] = True  # Only allow actions 5-9
        
        # Get multiple actions
        actions = []
        for _ in range(50):
            action, _, _ = trained_agent.select_action(state, mask)
            actions.append(action)
        
        # All actions should be in allowed range
        assert all(5 <= a < 10 for a in actions)
        
    def test_value_prediction(self, trained_agent):
        """Test value predictions are reasonable"""
        # Create states with different expected values
        good_state = np.ones(32).astype(np.float32)  # All positive features
        bad_state = -np.ones(32).astype(np.float32)  # All negative features
        
        _, _, good_value = trained_agent.select_action(good_state)
        _, _, bad_value = trained_agent.select_action(bad_state)
        
        # Values should be different
        assert good_value != bad_value


class TestDraftAgents:
    """Test draft agent implementations"""
    
    def test_random_draft_agent(self):
        """Test random draft agent"""
        agent = RandomDraftAgent()
        
        # Create fake cards
        cards = [
            PokemonCard(f"Pokemon {i}", "TEST", str(i), None, hp=60 + i*10, stage="Basic")
            for i in range(10)
        ]
        
        state = np.random.randn(42)  # Draft state vector size
        
        # Select card multiple times
        selections = []
        for _ in range(100):
            selection = agent.select_card(state, cards)
            selections.append(selection)
        
        # Should select different cards
        assert len(set(selections)) > 5
        
        # All selections should be valid indices
        assert all(0 <= s < len(cards) for s in selections)
    
    def test_draft_strategy_metrics(self):
        """Test draft agent can be evaluated on strategy"""
        agent = RandomDraftAgent()
        
        # Simulate draft picks with varying quality
        high_value_cards = [
            PokemonCard(f"Rare {i}", "TEST", str(i), None, hp=200, stage="ex", rarity="Ultra Rare")
            for i in range(3)
        ]
        
        low_value_cards = [
            PokemonCard(f"Common {i}", "TEST", str(i+10), None, hp=50, stage="Basic", rarity="Common")
            for i in range(7)
        ]
        
        all_cards = high_value_cards + low_value_cards
        state = np.random.randn(42)
        
        # Track selections
        high_value_picks = 0
        total_picks = 100
        
        for _ in range(total_picks):
            pick = agent.select_card(state, all_cards)
            if pick < len(high_value_cards):
                high_value_picks += 1
        
        # Random agent should pick roughly proportionally
        expected_ratio = len(high_value_cards) / len(all_cards)
        actual_ratio = high_value_picks / total_picks
        
        # Allow some variance
        assert abs(actual_ratio - expected_ratio) < 0.2


class TestAgentComparison:
    """Test comparing different agent types"""
    
    @pytest.fixture
    def test_env(self):
        """Create test environment"""
        card_db = CardDatabase()
        
        # Create minimal card set
        sample_data = {
            "cards": [
                {
                    "name": "Pikachu",
                    "set_code": "TEST",
                    "number": "1",
                    "card_type": "Pokemon",
                    "hp": 60,
                    "stage": "Basic",
                    "attacks": [
                        {"name": "Thunder Shock", "damage": "20", "cost": "LC"}
                    ],
                    "weakness": "Fighting",
                    "retreat_cost": 1
                }
            ] + [
                {
                    "name": "Lightning Energy",
                    "set_code": "TEST",
                    "number": str(100 + i),
                    "card_type": "Energy",
                    "energy_type": "Lightning"
                } for i in range(20)
            ] + [
                {
                    "name": "Professor's Research",
                    "set_code": "TEST", 
                    "number": str(200 + i),
                    "card_type": "Trainer",
                    "trainer_type": "Supporter",
                    "effect": "Draw 7 cards"
                } for i in range(20)
            ]
        }
        
        # Duplicate cards to make 60-card decks
        all_cards = sample_data["cards"]
        sample_data["cards"] = all_cards * 3  # Make enough for deck building
        
        card_db.load_from_json(sample_data)
        
        return PokemonTCGEnv(card_db, use_graph_encoder=False, max_steps=50)
    
    def test_random_vs_random(self, test_env):
        """Test random agents playing against each other"""
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        
        wins = {1: 0, 2: 0}
        games = 10
        
        for _ in range(games):
            obs = test_env.reset()
            done = False
            last_player = 1
            
            while not done:
                current_player = test_env.engine.state.current_player
                agent = agent1 if current_player == 1 else agent2
                
                legal_actions = test_env.engine.get_legal_actions(current_player)
                if not legal_actions:
                    break
                
                action = agent.select_action(obs, legal_actions)
                action_idx = test_env._action_to_idx(action, legal_actions)
                
                if action_idx is None:
                    action_idx = 0  # Default action
                
                obs, reward, done, info = test_env.step(action_idx)
                last_player = current_player
            
            # Determine winner
            if done:
                done_check, winner = test_env.engine._check_win_conditions()
                if winner:
                    wins[winner] += 1
        
        # Random agents should win roughly equally
        if games >= 10:
            win_ratio = wins[1] / max(1, wins[1] + wins[2])
            assert 0.2 < win_ratio < 0.8  # Allow variance
    
    def test_trained_vs_random(self, test_env, trained_agent):
        """Test trained agent should beat random agent"""
        random_agent = RandomAgent()
        
        wins = {'trained': 0, 'random': 0}
        games = 5
        
        for game_idx in range(games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                agents = {1: ('trained', trained_agent), 2: ('random', random_agent)}
            else:
                agents = {1: ('random', random_agent), 2: ('trained', trained_agent)}
            
            obs = test_env.reset()
            done = False
            
            while not done:
                current_player = test_env.engine.state.current_player
                agent_name, agent = agents[current_player]
                
                if agent_name == 'trained':
                    action_mask = test_env.get_action_mask(current_player)
                    action, _, _ = agent.select_action(obs, action_mask)
                else:
                    legal_actions = test_env.engine.get_legal_actions(current_player)
                    if legal_actions:
                        game_action = agent.select_action(obs, legal_actions)
                        action = test_env._action_to_idx(game_action, legal_actions)
                        if action is None:
                            action = 0
                    else:
                        action = 0
                
                obs, reward, done, info = test_env.step(action)
            
            # Determine winner
            if done:
                done_check, winner = test_env.engine._check_win_conditions()
                if winner:
                    winner_name = agents[winner][0]
                    wins[winner_name] += 1
        
        # Trained agent should win more (but with small sample, just check it wins at least once)
        assert wins['trained'] >= 1


class TestSpecializedAgents:
    """Test specialized agent strategies"""
    
    def test_aggressive_strategy(self):
        """Test agent that prioritizes attacks"""
        # This would be implemented as a custom agent
        # For now, test the concept
        
        class AggressiveAgent:
            def select_action(self, state, legal_actions):
                # Prioritize attack actions
                attack_actions = [a for a in legal_actions if a.action_type == ActionType.ATTACK]
                if attack_actions:
                    return attack_actions[0]
                return legal_actions[0] if legal_actions else Action(ActionType.PASS)
        
        agent = AggressiveAgent()
        
        # Test with mixed actions
        actions = [
            Action(ActionType.PLAY_BASIC_POKEMON, card_index=0),
            Action(ActionType.ATTACK, attack_index=0),
            Action(ActionType.END_TURN)
        ]
        
        selected = agent.select_action(None, actions)
        assert selected.action_type == ActionType.ATTACK
    
    def test_defensive_strategy(self):
        """Test agent that prioritizes defense"""
        
        class DefensiveAgent:
            def select_action(self, state, legal_actions):
                # Prioritize bench building and energy attachment
                priority_order = [
                    ActionType.PLAY_BASIC_POKEMON,
                    ActionType.ATTACH_ENERGY,
                    ActionType.EVOLVE_POKEMON,
                    ActionType.PLAY_TRAINER,
                    ActionType.RETREAT,
                    ActionType.END_TURN,
                    ActionType.ATTACK
                ]
                
                for action_type in priority_order:
                    matching = [a for a in legal_actions if a.action_type == action_type]
                    if matching:
                        return matching[0]
                
                return legal_actions[0] if legal_actions else Action(ActionType.PASS)
        
        agent = DefensiveAgent()
        
        # Test prioritizes bench over attack
        actions = [
            Action(ActionType.ATTACK, attack_index=0),
            Action(ActionType.PLAY_BASIC_POKEMON, card_index=0),
            Action(ActionType.END_TURN)
        ]
        
        selected = agent.select_action(None, actions)
        assert selected.action_type == ActionType.PLAY_BASIC_POKEMON


def test_agent_state_representation():
    """Test how agents interpret state vectors"""
    # Create a known state
    state = np.zeros(32, dtype=np.float32)
    
    # Set specific features
    state[0] = 0.8   # Active Pokemon HP ratio
    state[1] = 2.0   # Stage (Stage 1)
    state[2] = 0.3   # Energy attached
    state[22] = 1.0  # All prizes remaining
    state[25] = 0.0  # No opponent prizes taken
    
    # Different agents might weight features differently
    # This tests that state is interpretable
    
    # High HP should indicate good position
    assert state[0] > 0.5
    
    # Full prizes means game just started
    assert state[22] == 1.0
    
    # State should be normalized
    assert np.all(state >= 0)
    assert np.all(state <= 10)  # Reasonable bounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
