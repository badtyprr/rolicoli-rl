"""
Pokemon Trading Card Game OpenAI Gym Environment
Wraps the existing game engine in a Gym-compatible interface for use with Stable Baselines3
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from pokemon_tcg_rl import (
    CardDatabase, GameEngine, GameState, Action, ActionType,
    Card, PokemonCard, TrainerCard, EnergyCard, EnergyType,
    DeckBuilder, GamePhase
)

logger = logging.getLogger(__name__)


class PokemonTCGGymEnv(gym.Env):
    """
    OpenAI Gym wrapper for Pokemon TCG
    Compatible with Stable Baselines3 and supports partial observability
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 card_database: CardDatabase,
                 deck1_type: EnergyType = EnergyType.LIGHTNING,
                 deck2_type: EnergyType = EnergyType.WATER,
                 max_steps: int = 200,
                 reward_shaping: bool = True,
                 self_play: bool = False,
                 opponent_policy: Optional[Any] = None):
        """
        Initialize Pokemon TCG Gym Environment

        Args:
            card_database: Database of available cards
            deck1_type: Energy type for player 1 deck
            deck2_type: Energy type for player 2 deck
            max_steps: Maximum steps per episode
            reward_shaping: Whether to use reward shaping
            self_play: Whether to use self-play (both players controlled by agent)
            opponent_policy: Optional opponent policy for player 2
        """
        super().__init__()

        self.card_db = card_database
        self.deck_builder = DeckBuilder(card_database)
        self.engine = GameEngine(card_database)

        self.deck1_type = deck1_type
        self.deck2_type = deck2_type
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.self_play = self_play
        self.opponent_policy = opponent_policy

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # Define observation space
        # Using the same 32-dimensional state vector as original
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(32,),
            dtype=np.float32
        )

        # Define action space
        # Maximum possible actions in Pokemon TCG
        self.action_space = spaces.Discrete(200)

        # Store last game state for reward shaping
        self._last_game_metrics = None

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0

        # Build decks
        deck1 = self.deck_builder.build_basic_deck(self.deck1_type)
        deck2 = self.deck_builder.build_basic_deck(self.deck2_type)

        # Setup new game
        self.engine.setup_game(deck1, deck2)

        # Get initial observation
        obs = self._get_observation()

        # Initialize metrics for reward shaping
        if self.reward_shaping:
            self._last_game_metrics = self._get_game_metrics()

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment

        Args:
            action: Action index to execute

        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        self.current_step += 1

        # Handle opponent moves if not self-play
        if not self.self_play and self.engine.state.current_player == 2:
            self._execute_opponent_move()
            # After opponent move, check if game is over
            done, winner = self.engine._check_win_conditions()
            if done:
                reward = 10.0 if winner == 1 else -10.0
                obs = self._get_observation()
                return obs, reward, done, {'winner': winner}

        # Get legal actions for current player
        legal_actions = self.engine.get_legal_actions(self.engine.state.current_player)

        # Convert action index to game action
        if action >= len(legal_actions):
            # Invalid action - use first legal action
            game_action = legal_actions[0] if legal_actions else Action(ActionType.PASS)
            reward = -0.1  # Small penalty for invalid action
        else:
            game_action = legal_actions[action]
            reward = 0.0

        # Store old metrics if using reward shaping
        if self.reward_shaping:
            old_metrics = self._get_game_metrics()

        # Execute action
        new_state, action_reward, done = self.engine.apply_action(game_action)
        reward += action_reward

        # Apply reward shaping
        if self.reward_shaping and not done:
            new_metrics = self._get_game_metrics()
            shaped_reward = self._compute_shaped_reward(old_metrics, new_metrics, game_action)
            reward += shaped_reward
            self._last_game_metrics = new_metrics

        # Check max steps
        if self.current_step >= self.max_steps:
            done = True

        # Get next observation
        obs = self._get_observation()

        # Build info dict
        info = {
            'current_player': self.engine.state.current_player,
            'turn_count': self.engine.state.turn_count,
            'legal_actions_count': len(legal_actions),
            'action_type': game_action.action_type.value if game_action else None
        }

        # Add winner info if game is done
        if done:
            _, winner = self.engine._check_win_conditions()
            info['winner'] = winner

        self.episode_reward += reward

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from game state"""
        # Always observe from player 1's perspective for single-agent training
        return self.engine.get_state_vector(1)

    def _get_game_metrics(self) -> Dict[str, Any]:
        """Get current game metrics for reward shaping"""
        player_state = self.engine.state.get_player_state(1)
        opponent_state = self.engine.state.get_opponent_state(1)

        return {
            "bench_size": len(player_state["bench"]),
            "total_energy": sum(
                sum(p.attached_energy.values())
                for p in [player_state["active"]] + player_state["bench"]
                if p
            ),
            "prizes_left": len(player_state["prizes"]),
            "opp_prizes_left": len(opponent_state["prizes"]),
            "hand_size": len(player_state["hand"]),
            "total_damage": sum(
                p.damage_counters
                for p in [opponent_state["active"]] + opponent_state["bench"]
                if p
            )
        }

    def _compute_shaped_reward(self, old_state: Dict, new_state: Dict,
                              action: Action) -> float:
        """Compute shaped reward based on state changes"""
        reward = 0.0

        # Reward for playing Pokemon to bench
        if new_state["bench_size"] > old_state["bench_size"]:
            reward += 0.1

        # Reward for attaching energy
        if new_state["total_energy"] > old_state["total_energy"]:
            reward += 0.1

        # Reward for dealing damage
        damage_dealt = new_state["total_damage"] - old_state["total_damage"]
        reward += damage_dealt * 0.01

        # Reward for taking prizes
        prizes_taken = old_state["opp_prizes_left"] - new_state["opp_prizes_left"]
        reward += prizes_taken * 1.0

        # Penalty for losing prizes
        prizes_lost = old_state["prizes_left"] - new_state["prizes_left"]
        reward -= prizes_lost * 1.0

        # Small reward for evolving
        if action.action_type == ActionType.EVOLVE_POKEMON:
            reward += 0.2

        return reward

    def _execute_opponent_move(self):
        """Execute opponent move (player 2)"""
        if self.opponent_policy is not None:
            # Use provided policy
            obs = self.engine.get_state_vector(2)
            action = self.opponent_policy.predict(obs, deterministic=True)[0]
            legal_actions = self.engine.get_legal_actions(2)
            if action < len(legal_actions):
                self.engine.apply_action(legal_actions[action])
            else:
                self.engine.apply_action(legal_actions[0] if legal_actions else Action(ActionType.PASS))
        else:
            # Random opponent
            legal_actions = self.engine.get_legal_actions(2)
            if legal_actions:
                import random
                action = random.choice(legal_actions)
                self.engine.apply_action(action)

    def render(self, mode='human'):
        """Render the current game state"""
        if mode == 'ansi':
            return self._render_ansi()
        elif mode == 'human':
            print(self._render_ansi())

    def _render_ansi(self) -> str:
        """Render game state as ASCII text"""
        state = self.engine.state
        output = []
        output.append("=" * 50)
        output.append(f"Turn {state.turn_count} - Player {state.current_player}'s turn")
        output.append("=" * 50)

        # Player 1 state
        output.append("\nPlayer 1:")
        output.append(f"  Prizes: {len(state.player1_prizes)}")
        output.append(f"  Hand: {len(state.player1_hand)} cards")
        output.append(f"  Deck: {len(state.player1_deck)} cards")

        if state.player1_active:
            active = state.player1_active
            output.append(f"  Active: {active.name} ({active.current_hp}/{active.hp} HP)")

        output.append(f"  Bench: {len(state.player1_bench)} Pokemon")

        # Player 2 state
        output.append("\nPlayer 2:")
        output.append(f"  Prizes: {len(state.player2_prizes)}")
        output.append(f"  Hand: {len(state.player2_hand)} cards")
        output.append(f"  Deck: {len(state.player2_deck)} cards")

        if state.player2_active:
            active = state.player2_active
            output.append(f"  Active: {active.name} ({active.current_hp}/{active.hp} HP)")

        output.append(f"  Bench: {len(state.player2_bench)} Pokemon")

        return "\n".join(output)

    def close(self):
        """Clean up resources"""
        pass

    def seed(self, seed=None):
        """Set random seed"""
        import random
        import numpy as np

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        return [seed]

    def get_legal_action_mask(self) -> np.ndarray:
        """
        Get mask of legal actions for current player
        Used for action masking in PPO
        """
        legal_actions = self.engine.get_legal_actions(self.engine.state.current_player)
        mask = np.zeros(self.action_space.n, dtype=bool)

        for i in range(min(len(legal_actions), self.action_space.n)):
            mask[i] = True

        return mask