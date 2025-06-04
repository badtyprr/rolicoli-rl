"""
Pokemon TCG Draft Environment as OpenAI Gym
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from pokemon_tcg_rl import (
    CardDatabase, DraftEnvironment, DraftAction, DraftState,
    Card, PokemonCard, TrainerCard, EnergyCard
)


class PokemonTCGDraftEnv(gym.Env):
    """
    OpenAI Gym wrapper for Pokemon TCG Draft
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 card_database: CardDatabase,
                 draft_format: str = "booster",
                 total_packs: int = 3,
                 cards_per_pack: int = 10):
        """
        Initialize Draft Environment

        Args:
            card_database: Database of available cards
            draft_format: Type of draft ("booster" supported)
            total_packs: Number of packs to draft
            cards_per_pack: Cards per pack
        """
        super().__init__()

        self.card_db = card_database
        self.draft_env = DraftEnvironment(card_database, draft_format)
        self.draft_env.state.total_packs = total_packs
        self.draft_env.state.cards_per_pack = cards_per_pack

        # Define observation space (42-dimensional draft state vector)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(42,),
            dtype=np.float32
        )

        # Define action space (max cards per pack)
        self.action_space = spaces.Discrete(cards_per_pack)

    def reset(self) -> np.ndarray:
        """Reset draft to beginning"""
        self.draft_state, obs = self.draft_env.reset()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Make a draft pick

        Args:
            action: Index of card to pick

        Returns:
            observation: Next draft state
            reward: Terminal reward based on deck quality
            done: Whether draft is complete
            info: Additional information
        """
        # Execute draft action for player 1
        draft_action = DraftAction(action)
        self.draft_state, reward, done, info = self.draft_env.step(1, draft_action)

        if not done:
            # Simulate player 2 pick (random for now)
            if self.draft_state.player2_choices:
                import random
                opponent_pick = random.randint(0, len(self.draft_state.player2_choices) - 1)
                self.draft_env.step(2, DraftAction(opponent_pick))

        # Get next observation
        obs = self.draft_env._get_state_vector(1)

        return obs, reward, done, info

    def render(self, mode='human'):
        """Render current draft state"""
        if mode == 'ansi':
            return self._render_ansi()
        elif mode == 'human':
            print(self._render_ansi())

    def _render_ansi(self) -> str:
        """Render draft state as text"""
        output = []
        output.append("=" * 50)
        output.append(f"Pack {self.draft_state.current_pack + 1}/{self.draft_state.total_packs}")
        output.append(f"Pick {self.draft_state.current_pick + 1}/{self.draft_state.cards_per_pack}")
        output.append("=" * 50)

        output.append("\nCurrent choices:")
        for i, card in enumerate(self.draft_state.player1_choices):
            if isinstance(card, PokemonCard):
                output.append(f"  {i}: {card.name} ({card.hp} HP, {card.rarity or 'Common'})")
            else:
                output.append(f"  {i}: {card.name}")

        output.append(f"\nCards drafted: {len(self.draft_state.player1_pool)}")

        return "\n".join(output)

    def close(self):
        """Clean up resources"""
        pass

    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            self.draft_env.rng.seed(seed)
        return [seed]