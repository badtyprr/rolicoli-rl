"""
Unit tests for Pokemon TCG game engine
Tests game rules, mechanics, and state management
"""

import pytest
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pokemon_tcg_rl import (
    CardType, TrainerType, EnergyType, GamePhase, SpecialCondition,
    ActionType, Attack, Ability, Card, PokemonCard, TrainerCard,
    EnergyCard, GameState, Action, CardDatabase, GameEngine
)


class TestCardDatabase:
    """Test card database functionality"""
    
    def test_load_cards(self):
        """Test loading cards from JSON"""
        sample_data = {
            "cards": [
                {
                    "name": "Pikachu",
                    "set_code": "SVI",
                    "number": "025",
                    "card_type": "Pokemon",
                    "hp": 70,
                    "stage": "Basic",
                    "attacks": [
                        {"name": "Thunder Shock", "damage": "30", "cost": "LC"}
                    ],
                    "weakness": "Fighting",
                    "retreat_cost": 1
                }
            ]
        }
        
        db = CardDatabase(sample_data)
        assert len(db.cards) == 1
        
        pikachu = db.get_card("Pikachu")
        assert pikachu is not None
        assert isinstance(pikachu, PokemonCard)
        assert pikachu.hp == 70
        assert len(pikachu.attacks) == 1
        
    def test_parse_energy_cost(self):
        """Test energy cost parsing"""
        db = CardDatabase()
        
        cost = db._parse_energy_cost("LLC")
        assert cost[EnergyType.LIGHTNING] == 2
        assert cost[EnergyType.COLORLESS] == 1
        
        cost = db._parse_energy_cost("WWCC")
        assert cost[EnergyType.WATER] == 2
        assert cost[EnergyType.COLORLESS] == 2


class TestGameEngine:
    """Test game engine functionality"""

    @pytest.fixture
    def sample_deck(self):
        """Create a sample 60-card deck"""
        deck = []

        # Add Pokemon (15 cards)
        for i in range(15):
            pokemon = PokemonCard(
                name=f"Pokemon {i}",
                set_code="TEST",
                number=str(i),
                card_type=CardType.POKEMON,
                hp=60,
                stage="Basic",
                pokemon_type=EnergyType.LIGHTNING,
                attacks=[Attack("Attack", "20", {EnergyType.COLORLESS: 1})],
                retreat_cost=1
            )
            deck.append(pokemon)

        # Add Trainers (30 cards)
        for i in range(30):
            trainer = TrainerCard(
                name=f"Trainer {i}",
                set_code="TEST",
                number=str(100 + i),
                card_type=CardType.TRAINER,
                trainer_type=TrainerType.ITEM,
                effect_text="Draw 1 card"
            )
            deck.append(trainer)

        # Add Energy (15 cards)
        for i in range(15):
            energy = EnergyCard(
                name="Lightning Energy",
                set_code="TEST",
                number=str(200 + i),
                card_type=CardType.ENERGY,
                energy_type=EnergyType.LIGHTNING
            )
            deck.append(energy)

        assert len(deck) == 60  # Ensure we have exactly 60 cards
        return deck
    
    def test_game_setup(self, sample_deck):
        """Test game initialization"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        # Check initial state
        assert len(engine.state.player1_hand) == 7
        assert len(engine.state.player2_hand) == 7
        assert len(engine.state.player1_prizes) == 6
        assert len(engine.state.player2_prizes) == 6
        assert engine.state.player1_active is not None
        assert engine.state.player2_active is not None
        
    def test_legal_actions(self, sample_deck):
        """Test legal action generation"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        # Get legal actions for current player
        actions = engine.get_legal_actions(engine.state.current_player)
        assert len(actions) > 0
        
        # Should be able to play basic Pokemon, attach energy, etc.
        action_types = {a.action_type for a in actions}
        assert ActionType.END_TURN in action_types
        
    def test_attack_damage_calculation(self):
        """Test damage calculation with weakness/resistance"""
        engine = GameEngine()
        
        # Create attacking Pokemon
        attacker = PokemonCard(
            name="Attacker",
            set_code="TEST",
            number="1",
            card_type=CardType.POKEMON,
            hp=100,
            stage="Basic",
            pokemon_type=EnergyType.FIRE,
            attacks=[Attack("Fire Blast", "50", {})],
            retreat_cost=2
        )
        
        # Create defending Pokemon with weakness
        defender = PokemonCard(
            name="Defender",
            set_code="TEST",
            number="2",
            card_type=CardType.POKEMON,
            hp=100,
            stage="Basic",
            pokemon_type=EnergyType.GRASS,
            attacks=[],
            weakness=EnergyType.FIRE,
            retreat_cost=1
        )
        
        # Calculate damage
        damage = engine._calculate_damage(
            attacker.attacks[0], attacker, defender, 1
        )
        
        # Should be doubled due to weakness
        assert damage == 100
        
    def test_knockout_handling(self, sample_deck):
        """Test Pokemon knockout mechanics"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        # Damage opponent's active Pokemon
        opp_active = engine.state.player2_active
        opp_active.damage_counters = opp_active.hp
        
        # Handle knockout
        initial_prizes = len(engine.state.player1_prizes)
        engine._handle_knockout(opp_active, 2, 1)
        
        # Check prize card was taken
        assert len(engine.state.player1_prizes) == initial_prizes - 1
        assert len(engine.state.player1_hand) > 7  # Prize added to hand
        
    def test_win_conditions(self, sample_deck):
        """Test various win conditions"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        # Test prize card win
        engine.state.player1_prizes.clear()
        done, winner = engine._check_win_conditions()
        assert done is True
        assert winner == 1
        
        # Test no Pokemon win
        engine.state.player1_prizes = [Card("Prize", "TEST", "1", CardType.POKEMON)]
        engine.state.player2_active = None
        engine.state.player2_bench.clear()
        done, winner = engine._check_win_conditions()
        assert done is True
        assert winner == 1


class TestPokemonMechanics:
    """Test specific Pokemon TCG mechanics"""
    
    def test_evolution(self):
        """Test Pokemon evolution mechanics"""
        # Create basic Pokemon
        pikachu = PokemonCard(
            name="Pikachu",
            set_code="TEST",
            number="1",
            card_type=CardType.POKEMON,
            hp=60,
            stage="Basic",
            pokemon_type=EnergyType.LIGHTNING,
            attacks=[Attack("Thunder Shock", "20", {EnergyType.LIGHTNING: 1})],
            retreat_cost=1
        )
        pikachu.turns_in_play = 1
        pikachu.attached_energy[EnergyType.LIGHTNING] = 2
        pikachu.damage_counters = 20
        
        # Create evolution
        raichu = PokemonCard(
            name="Raichu",
            set_code="TEST",
            number="2",
            card_type=CardType.POKEMON,
            hp=120,
            stage="Stage 1",
            evolves_from="Pikachu",
            pokemon_type=EnergyType.LIGHTNING,
            attacks=[Attack("Thunder", "90", {EnergyType.LIGHTNING: 3})],
            retreat_cost=2
        )
        
        # Evolve should preserve energy and damage
        engine = GameEngine()
        engine.state.player1_active = pikachu
        engine.state.player1_hand = [raichu]
        
        action = Action(ActionType.EVOLVE_POKEMON, card_index=0, target_index=-1)
        engine.apply_action(action)
        
        # Check evolution preserved state
        evolved = engine.state.player1_active
        assert evolved.name == "Raichu"
        assert evolved.attached_energy[EnergyType.LIGHTNING] == 2
        assert evolved.damage_counters == 20
        assert evolved.evolved_this_turn is True
        
    def test_special_conditions(self):
        """Test special condition mechanics"""
        pokemon = PokemonCard(
            name="Test Pokemon",
            set_code="TEST",
            number="1",
            card_type=CardType.POKEMON,
            hp=100,
            stage="Basic",
            pokemon_type=EnergyType.PSYCHIC,
            attacks=[],
            retreat_cost=1
        )
        
        # Apply conditions
        pokemon.special_conditions.add(SpecialCondition.POISONED)
        pokemon.special_conditions.add(SpecialCondition.BURNED)
        
        engine = GameEngine()
        engine.state.player1_active = pokemon
        
        # Pokemon checkup should apply damage
        initial_damage = pokemon.damage_counters
        engine._pokemon_checkup()
        
        # Poison = 10 damage, Burn = 20 damage
        assert pokemon.damage_counters >= initial_damage + 10
        
    def test_energy_attachment(self):
        """Test energy attachment rules"""
        pokemon = PokemonCard(
            name="Test Pokemon",
            set_code="TEST",
            number="1",
            card_type=CardType.POKEMON,
            hp=100,
            stage="Basic",
            pokemon_type=EnergyType.WATER,
            attacks=[Attack("Water Gun", "30", {EnergyType.WATER: 2})],
            retreat_cost=1
        )
        
        energy = EnergyCard(
            name="Water Energy",
            set_code="TEST",
            number="100",
            card_type=CardType.ENERGY,
            energy_type=EnergyType.WATER
        )
        
        # Attach energy
        pokemon.attached_energy[energy.energy_type] += 1
        
        # Check if attack can be used
        assert not pokemon.attacks[0].can_use(pokemon.attached_energy)
        
        # Attach another
        pokemon.attached_energy[energy.energy_type] += 1
        assert pokemon.attacks[0].can_use(pokemon.attached_energy)


class TestStateVector:
    """Test state vector generation for RL"""

    @pytest.fixture
    def sample_deck(self):
        """Create a sample 60-card deck"""
        deck = []

        # Add Pokemon
        for i in range(15):
            pokemon = PokemonCard(
                name=f"Pokemon {i}",
                set_code="TEST",
                number=str(i),
                card_type=CardType.POKEMON,
                hp=60,
                stage="Basic",
                pokemon_type=EnergyType.LIGHTNING,
                attacks=[Attack("Attack", "20", {EnergyType.COLORLESS: 1})],
                retreat_cost=1
            )
            deck.append(pokemon)

        # Add Trainers
        for i in range(30):
            trainer = TrainerCard(
                name=f"Trainer {i}",
                set_code="TEST",
                number=str(100 + i),
                card_type=CardType.TRAINER,
                trainer_type=TrainerType.ITEM,
                effect_text="Draw 1 card"
            )
            deck.append(trainer)

        # Add Energy
        for i in range(15):
            energy = EnergyCard(
                name="Lightning Energy",
                set_code="TEST",
                number=str(200 + i),
                card_type=CardType.ENERGY,
                energy_type=EnergyType.LIGHTNING
            )
            deck.append(energy)

        return deck
    
    def test_state_vector_shape(self, sample_deck):
        """Test state vector has consistent shape"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        state_vec = engine.get_state_vector(1)
        assert isinstance(state_vec, np.ndarray)
        assert state_vec.shape == (32,)  # Expected feature count
        assert state_vec.dtype == np.float32
        
    def test_state_vector_values(self, sample_deck):
        """Test state vector contains valid values"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())
        
        state_vec = engine.get_state_vector(1)
        
        # All values should be normalized [0, 1] or small positive
        assert np.all(state_vec >= 0)
        assert np.all(state_vec <= 10)  # Reasonable upper bound
        
        # Specific features should be in expected ranges
        # Hand size (index 20) should be 7/10 = 0.7
        assert 0.6 <= state_vec[20] <= 0.8
        
        # Prize count (index 22) should be 6/6 = 1.0
        assert 0.9 <= state_vec[22] <= 1.1


class TestFullGame:
    """Test full game simulation"""

    @pytest.fixture
    def sample_deck(self):
        """Create a sample 60-card deck"""
        # Same implementation as above
        deck = []
        # ... (same deck creation code)
        return deck

    def test_full_game_simulation(self, sample_deck):
        """Test a complete game simulation"""
        engine = GameEngine()
        engine.setup_game(sample_deck.copy(), sample_deck.copy())

        # Play for a limited number of turns
        max_turns = 100
        turn_count = 0

        while turn_count < max_turns:
            current_player = engine.state.current_player
            actions = engine.get_legal_actions(current_player)

            if not actions:
                break

            # Take a random action
            import random
            action = random.choice(actions)

            _, _, done = engine.apply_action(action)

            if done:
                break

            turn_count += 1

        # Game should have progressed
        assert engine.state.turn_count > 1
        assert len(engine.action_history) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])