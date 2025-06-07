"""
Pokemon TCG Limitless Scraper for 2025 Standard Meta
Author: Professional Implementation
Description: Scrapes all Pokemon TCG cards legal in the 2025 Standard format from limitlesstcg.com
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys
import io

# Set up proper encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pokemon_tcg_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PokemonCard:
    """Data class representing a Pokemon TCG card with competitive play information"""
    name: str
    set_code: str
    number: str
    card_type: str  # Pokemon, Trainer, Energy
    hp: Optional[int] = None
    pokemon_type: Optional[str] = None  # Fire, Water, Grass, etc.
    stage: Optional[str] = None  # Basic, Stage 1, Stage 2
    suffix: Optional[str] = None  # ex, V, VMAX, VSTAR, etc.
    evolves_from: Optional[str] = None
    abilities: List[Dict[str, str]] = None
    attacks: List[Dict[str, str]] = None
    weakness: Optional[str] = None
    resistance: Optional[str] = None
    retreat_cost: Optional[int] = None
    retreat_cost_array: List[str] = None  # Energy type array representation
    regulation_mark: Optional[str] = None
    rarity: Optional[str] = None
    artist: Optional[str] = None
    card_text: Optional[str] = None  # For trainers and rule boxes
    rules: List[str] = None  # Rule box text for special cards
    trainer_type: Optional[str] = None  # Item, Supporter, Stadium, Tool
    energy_type: Optional[str] = None  # Basic, Special

    def __post_init__(self):
        if self.abilities is None:
            self.abilities = []
        if self.attacks is None:
            self.attacks = []
        if self.rules is None:
            self.rules = []
        if self.retreat_cost_array is None:
            self.retreat_cost_array = []


class LimitlessTCGScraper:
    """Professional scraper for Limitless TCG website"""

    BASE_URL = "https://limitlesstcg.com"
    CARDS_URL = f"{BASE_URL}/cards"

    # Energy type mappings
    ENERGY_SYMBOLS = {
        'G': 'Grass', 'R': 'Fire', 'W': 'Water', 'L': 'Lightning',
        'P': 'Psychic', 'F': 'Fighting', 'D': 'Darkness', 'M': 'Metal',
        'N': 'Dragon', 'Y': 'Fairy', 'C': 'Colorless'
    }

    # Rule box texts for special Pokemon
    RULE_BOX_TEXTS = {
        'ex': 'Pokémon ex rule — When your Pokémon ex is Knocked Out, your opponent takes 2 Prize cards.',
        'V': 'V rule — When your Pokémon V is Knocked Out, your opponent takes 2 Prize cards.',
        'VMAX': 'VMAX rule — When your Pokémon VMAX is Knocked Out, your opponent takes 3 Prize cards.',
        'VSTAR': 'VSTAR rule — When your Pokémon VSTAR is Knocked Out, your opponent takes 2 Prize cards.',
        'GX': 'Pokémon-GX rule — When your Pokémon-GX is Knocked Out, your opponent takes 2 Prize cards.',
        'EX': 'Pokémon-EX rule — When a Pokémon-EX has been Knocked Out, your opponent takes 2 Prize cards.',
    }

    def __init__(self, delay: float = 0.5, use_api_reference: bool = False):
        """
        Initialize the scraper

        Args:
            delay: Delay between requests in seconds (be respectful to the server)
            use_api_reference: Whether to download and use Pokemon TCG API data for validation
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.delay = delay
        self.cards_data: List[PokemonCard] = []
        self.standard_sets: Set[str] = set()
        self.api_reference_data = {}

        if use_api_reference:
            self._load_api_reference_data()

    def _load_api_reference_data(self):
        """Load reference data from Pokemon TCG API for validation"""
        logger.info("Loading Pokemon TCG API reference data...")
        try:
            # Try to get data from the API
            api_sets = ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'sv7', 'sv8']  # Scarlet & Violet sets

            for set_id in api_sets:
                try:
                    response = self._make_request(f"https://api.pokemontcg.io/v2/cards?q=set.id:{set_id}")
                    if response:
                        data = response.json()
                        if 'data' in data:
                            for card in data['data']:
                                # Create a key based on name and number
                                key = f"{card.get('name', '')}_{card.get('number', '')}"
                                self.api_reference_data[key] = card
                except Exception as e:
                    logger.debug(f"Could not load set {set_id}: {e}")

            logger.info(f"Loaded {len(self.api_reference_data)} reference cards")
        except Exception as e:
            logger.warning(f"Could not load API reference data: {e}")

    def _validate_with_reference(self, card: PokemonCard) -> PokemonCard:
        """Validate and enrich card data using API reference if available"""
        if not self.api_reference_data:
            return card

        # Look up card in reference data
        key = f"{card.name}_{card.number}"
        ref_card = self.api_reference_data.get(key)

        if ref_card:
            # Validate and fix card type if needed
            if ref_card.get('supertype') == 'Pokémon' and card.card_type != 'Pokemon':
                logger.warning(f"Fixing card type for {card.name}: {card.card_type} -> Pokemon")
                card.card_type = 'Pokemon'
                # Re-parse as Pokemon if type was wrong
                return card

            # Enrich missing data
            if card.card_type == 'Pokemon' and not card.hp:
                card.hp = ref_card.get('hp')

            # Add any missing attacks
            if not card.attacks and ref_card.get('attacks'):
                for ref_attack in ref_card['attacks']:
                    attack_cost = ''.join(ref_attack.get('cost', []))
                    card.attacks.append({
                        'cost': attack_cost,
                        'name': ref_attack.get('name', ''),
                        'damage': ref_attack.get('damage', ''),
                        'text': ref_attack.get('text', ''),
                        'energy_types': ref_attack.get('cost', [])
                    })

        return card

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make a request with error handling and rate limiting

        Args:
            url: URL to request

        Returns:
            Response object or None if failed
        """
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def _get_current_standard_sets(self) -> Set[str]:
        """
        Dynamically fetch all sets currently legal in Standard format
        Based on regulation marks G, H, and newer
        """
        logger.info("Fetching current Standard format sets...")
        standard_sets = set()

        try:
            # Method 1: Get the cards database page and extract sets
            response = self._make_request(self.CARDS_URL)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for set selector dropdown or set list
                # Try multiple possible selectors
                set_selectors = [
                    soup.find('select', {'name': 'set'}),
                    soup.find('select', {'id': 'set'}),
                    soup.find('select', {'class': re.compile('set')}),
                ]

                for selector in set_selectors:
                    if selector:
                        options = selector.find_all('option')
                        for option in options:
                            set_code = option.get('value', '').strip()
                            set_text = option.get_text().strip()

                            # Extract set code from either value or text
                            if set_code and 2 <= len(set_code) <= 4 and set_code.isalpha():
                                standard_sets.add(set_code.upper())
                            elif set_text:
                                # Try to extract code from text like "SVI - Scarlet & Violet"
                                match = re.match(r'^([A-Z]{2,4})\s*[-–]', set_text)
                                if match:
                                    standard_sets.add(match.group(1))
                        break

                # Also look for set links in the page
                set_links = soup.find_all('a', href=re.compile(r'/cards\?[^"]*set[:=]([A-Z]{2,4})'))
                for link in set_links:
                    href = link.get('href', '')
                    match = re.search(r'set[:=]([A-Z]{2,4})', href, re.IGNORECASE)
                    if match:
                        standard_sets.add(match.group(1).upper())

            # Method 2: Search for Standard format cards and extract sets
            if len(standard_sets) < 10:
                logger.info("Checking Standard format cards for set codes...")
                search_url = f"{self.CARDS_URL}?q=format:standard&cpp=100"
                response = self._make_request(search_url)

                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Find all card links
                    card_links = soup.find_all('a', href=re.compile(r'/cards/([A-Z]{2,4})/\d+'))
                    for link in card_links:
                        match = re.search(r'/cards/([A-Z]{2,4})/\d+', link.get('href', ''))
                        if match:
                            standard_sets.add(match.group(1).upper())

                    # Also check for set info in card listings
                    card_infos = soup.find_all(['span', 'div'], class_=re.compile('set|card-set'))
                    for info in card_infos:
                        text = info.get_text()
                        match = re.match(r'^([A-Z]{2,4})\b', text)
                        if match:
                            standard_sets.add(match.group(1))

            # Method 3: Check known Standard sets
            if len(standard_sets) < 10:
                logger.info("Verifying known Standard sets...")
                # Current Standard sets (Regulation G, H, and newer)
                known_sets = {
                    # Scarlet & Violet main sets
                    'SVI', 'PAL', 'OBF', 'MEW', 'PAR', 'PAF', 'TEF', 'TWM',
                    'SFA', 'SCR', 'SSP', 'PRE', 'JTG',
                    # Promos and special sets
                    'SVP', 'SVE',
                    # Possible future sets
                    'DRI', 'BTL'
                }

                # Verify each set exists
                verified_sets = set()
                for set_code in known_sets:
                    check_url = f"{self.CARDS_URL}?q=set:{set_code}"
                    response = self._make_request(check_url)
                    if response and 'No cards found' not in response.text and '0 cards found' not in response.text:
                        verified_sets.add(set_code)
                        logger.debug(f"Verified set: {set_code}")

                standard_sets.update(verified_sets)

            # Clean up set codes
            standard_sets = {s.upper() for s in standard_sets if s and 2 <= len(s) <= 4 and s.isalpha()}

        except Exception as e:
            logger.error(f"Error fetching Standard sets: {e}")

        # Fallback if nothing found
        if not standard_sets:
            logger.warning("No sets found dynamically, using comprehensive fallback list")
            standard_sets = {
                'JTG', 'PRE', 'SSP', 'SCR', 'SFA', 'TWM', 'TEF', 'PAF',
                'PAR', 'MEW', 'OBF', 'PAL', 'SVI', 'SVE', 'SVP'
            }

        logger.info(f"Found {len(standard_sets)} Standard legal sets: {sorted(standard_sets)}")
        return standard_sets

    def _is_standard_legal_set(self, page_text: str) -> bool:
        """
        Check if a set contains Standard legal cards (regulation marks G, H, I, or newer)
        """
        # Look for regulation marks in the page
        reg_marks = re.findall(r'([G-Z])\s+Regulation Mark', page_text)

        # G and later marks are Standard legal
        for mark in reg_marks:
            if mark >= 'G':
                return True

        # Also check for specific Standard format mentions
        if 'format:standard' in page_text.lower():
            return True

        return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_energy_cost(self, cost_text: str) -> Tuple[str, List[str]]:
        """Parse energy cost string into formatted string and list of types"""
        if not cost_text:
            return "", []

        cost_text = cost_text.strip()
        types = []

        for char in cost_text:
            if char in self.ENERGY_SYMBOLS:
                types.append(self.ENERGY_SYMBOLS[char])

        return cost_text, types

    def _determine_card_type(self, name: str, text: str, soup: BeautifulSoup) -> str:
        """More accurately determine card type"""
        # Clean the text for better matching
        clean_text = re.sub(r'\s+', ' ', text)

        # Check for Trainer cards first - they have specific pattern
        if re.search(r'Trainer\s*[-–—]\s*(Item|Supporter|Stadium|Tool|Pokémon Tool)', clean_text):
            return 'Trainer'

        # Check for Basic Energy
        if 'Basic' in name and 'Energy' in name:
            return 'Energy'

        # Check for HP - strong indicator of Pokemon
        if re.search(r'\d+\s*HP', text):
            return 'Pokemon'

        # Check for Pokemon-specific keywords
        if any(keyword in text for keyword in ['Stage 1', 'Stage 2', 'Basic Pokémon', 'Evolves from', 'Weakness:', 'Resistance:', 'Retreat:']):
            return 'Pokemon'

        # Check if name contains Pokemon suffixes
        if any(suffix in name for suffix in [' ex', ' V', ' VMAX', ' VSTAR', ' GX', ' EX']):
            return 'Pokemon'

        # Check for Energy keywords
        if 'Energy' in text and 'HP' not in text:
            # Check specifically for energy card patterns
            if 'Special Energy' in text or 'Basic Energy' in text:
                return 'Energy'
            # Energy cards usually have very short text
            if len(text) < 500 and ('provides' in text.lower() or 'Energy' in name):
                return 'Energy'

        # Check for Trainer keywords without HP
        if 'HP' not in text:
            # Check for trainer action keywords
            trainer_keywords = ['draw', 'search your deck', 'discard', 'shuffle', 'attach',
                              'your turn', 'opponent', 'prize card', 'bench', 'switch']
            if any(keyword in text.lower() for keyword in trainer_keywords):
                return 'Trainer'

        # Check HTML structure for more clues
        if soup:
            # Look for Pokemon-specific elements
            if soup.find(string=re.compile(r'\d+\s*HP')):
                return 'Pokemon'

            # Look for attack cost patterns (energy symbols followed by attack names)
            if soup.find(string=re.compile(r'[GRWLPFDMNYC]+\s+[A-Z][a-z]')):
                return 'Pokemon'

        # Default based on content patterns
        if len(text) > 800 or 'HP' in text:
            return 'Pokemon'
        elif 'Energy' in name or 'Energy' in text:
            return 'Energy'
        else:
            # Default to Trainer for cards without HP
            return 'Trainer'

    def _parse_pokemon_card(self, soup: BeautifulSoup, url: str, actual_set_code: str = None) -> Optional[PokemonCard]:
        """Parse a Pokemon card page"""
        try:
            # Extract basic information from the page structure
            card_container = soup.find('div', {'class': 'card-detail'}) or soup.find('main') or soup

            # Get card name - clean it properly
            name_elem = soup.find('h1') or soup.find('title')
            if name_elem:
                name = name_elem.get_text().strip()
                # Remove set info if in title
                name = name.split(' - ')[0].strip()
            else:
                return None

            # Extract set and number from URL
            url_match = re.search(r'/cards/([^/]+)/(\d+)', url)
            if url_match:
                set_code = actual_set_code or url_match.group(1)
                number = url_match.group(2)
            else:
                return None

            # Skip if not in Standard format
            if set_code not in self.standard_sets:
                return None

            # Get all text content for parsing
            page_text = card_container.get_text()

            # Determine card type more accurately
            card_type = self._determine_card_type(name, page_text, soup)

            card = PokemonCard(
                name=name,
                set_code=set_code,
                number=number,
                card_type=card_type
            )

            # Parse based on card type
            if card_type == 'Pokemon':
                self._parse_pokemon_details(card, page_text, soup)
            elif card_type == 'Trainer':
                self._parse_trainer_details(card, page_text, soup)
            elif card_type == 'Energy':
                self._parse_energy_details(card, page_text, soup)

            # Parse common attributes
            self._parse_common_attributes(card, page_text)

            return card

        except Exception as e:
            logger.error(f"Error parsing card at {url}: {e}")
            return None

    def _parse_pokemon_details(self, card: PokemonCard, text: str, soup: BeautifulSoup):
        """Parse Pokemon-specific details"""
        # Pokemon Type - this is the energy type for the Pokemon
        type_match = re.search(r'([A-Za-z]+)\s+(?:-|–)\s+\d+\s*HP', text)
        if type_match:
            pokemon_type_name = type_match.group(1).strip()
            card.pokemon_type = pokemon_type_name
            # Also set energy_type for Pokemon cards
            card.energy_type = pokemon_type_name

        # HP
        hp_match = re.search(r'(\d+)\s*HP', text)
        if hp_match:
            card.hp = int(hp_match.group(1))

        # Stage and suffix - separate them properly
        card.suffix = None
        if re.search(r'\bTera\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'Tera'
        elif 'VMAX' in card.name or re.search(r'\bVMAX\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'VMAX'
        elif 'VSTAR' in card.name or re.search(r'\bVSTAR\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'VSTAR'
        elif ' V' in card.name or re.search(r'\bPokémon V\b', text):
            card.stage = 'Basic'  # V Pokemon are always Basic
            card.suffix = 'V'
        elif ' ex' in card.name or re.search(r'\bPokémon ex\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'ex'
        elif ' GX' in card.name or re.search(r'\bGX\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'GX'
        elif ' EX' in card.name or re.search(r'\bEX\b', text):
            card.stage = self._determine_base_stage(text)
            card.suffix = 'EX'
        else:
            card.stage = self._determine_base_stage(text)

        # Add rule box text based on suffix
        if card.suffix and card.suffix in self.RULE_BOX_TEXTS:
            card.rules.append(self.RULE_BOX_TEXTS[card.suffix])

        # Evolves from
        evolves_match = re.search(r'Evolves from\s+([A-Za-z\s]+?)(?:\n|$|[A-Z])', text)
        if evolves_match:
            card.evolves_from = evolves_match.group(1).strip()

        # Parse abilities - with better text extraction
        self._parse_abilities(card, text, soup)

        # Parse attacks
        self._parse_attacks(card, text)

        # Weakness, Resistance, Retreat
        weakness_match = re.search(r'Weakness:\s*([^\n]+)', text)
        if weakness_match:
            weakness = weakness_match.group(1).strip()
            card.weakness = weakness if weakness.lower() != 'none' else None

        resistance_match = re.search(r'Resistance:\s*([^\n]+)', text)
        if resistance_match:
            resistance = resistance_match.group(1).strip()
            card.resistance = resistance if resistance.lower() != 'none' else None

        # Retreat cost - both numeric and array representation
        retreat_match = re.search(r'Retreat:\s*(\d+)', text)
        if retreat_match:
            card.retreat_cost = int(retreat_match.group(1))
            # Create array representation
            card.retreat_cost_array = ['Colorless'] * card.retreat_cost

        # Parse flavor text (appears after main card info, before illustrator)
        self._parse_flavor_text(card, text, soup)

        # Look for any additional rules in the text
        self._parse_rules(card, text)

    def _determine_base_stage(self, text: str) -> str:
        """Determine the base stage without suffix"""
        if re.search(r'\bStage 2\b', text):
            return 'Stage 2'
        elif re.search(r'\bStage 1\b', text):
            return 'Stage 1'
        elif re.search(r'\bBasic\b', text):
            return 'Basic'
        else:
            return 'Basic'  # Default if not found

    def _parse_rules(self, card: PokemonCard, text: str):
        """Parse any additional rule box text"""
        # Look for other common rule patterns
        rule_patterns = [
            r'(VSTAR Power[^.]+\.)',
            r'(You can\'t have more than \d+ [^.]+in your deck\.)',
            r'(Prism Star Rule[^.]+\.)',
            r'(Tag Team rule[^.]+\.)',
            r'(Ancient Trait[^.]+\.)',
        ]

        for pattern in rule_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and match not in card.rules:
                    card.rules.append(match)

    def _parse_abilities(self, card: PokemonCard, text: str, soup: BeautifulSoup = None):
        """Parse Pokemon abilities with improved text extraction"""
        # Method 1: Try to find abilities in structured HTML first
        if soup:
            ability_divs = soup.find_all('div', class_=re.compile('ability|power|body'))
            for div in ability_divs:
                ability_name_elem = div.find(class_=re.compile('ability-name|power-name'))
                ability_text_elem = div.find(class_=re.compile('ability-text|power-text'))

                if ability_name_elem and ability_text_elem:
                    ability_name = self._clean_text(ability_name_elem.get_text())
                    ability_text = self._clean_text(ability_text_elem.get_text())

                    if ability_name or ability_text:
                        card.abilities.append({
                            'name': ability_name,
                            'text': ability_text
                        })

        # Method 2: Pattern-based extraction from text
        # More comprehensive pattern to capture full ability text
        ability_pattern = re.compile(
            r'(?:Ability|Poké-POWER|Poké-BODY)\s*[:\.]?\s*([^\n]+?)[\n\s]+([^⚫]+?)(?=(?:Ability|Poké-POWER|Poké-BODY|[GRWLPFDMNYC]{1,3}\s+[A-Z][a-z]|Weakness|Resistance|Retreat|Pokémon ex rule|V rule|$))',
            re.MULTILINE | re.DOTALL
        )

        abilities = ability_pattern.findall(text)
        for ability_name, ability_text in abilities:
            ability_name = self._clean_text(ability_name)
            ability_text = self._clean_text(ability_text)

            # Skip if both are empty
            if not ability_name and not ability_text:
                continue

            # Don't add duplicate abilities
            duplicate = False
            for existing in card.abilities:
                if existing['name'] == ability_name:
                    # Update text if it's longer/better
                    if len(ability_text) > len(existing['text']):
                        existing['text'] = ability_text
                    duplicate = True
                    break

            if not duplicate and (ability_name or ability_text):
                card.abilities.append({
                    'name': ability_name,
                    'text': ability_text
                })

        # Clean up: Remove any empty abilities
        card.abilities = [a for a in card.abilities if a.get('name') or a.get('text')]

    def _parse_attacks(self, card: PokemonCard, text: str):
        """Parse Pokemon attacks with improved effect text capture"""
        # Try multiple patterns to capture attacks with their full effect text

        # Pattern 1: More comprehensive pattern that captures attack effects
        attack_pattern = re.compile(
            r'([GRWLPFDMNYC]+)\s+([A-Z][A-Za-z\s]+?)\s+(\d+\+?)\s*\n?([^⚫\n]+(?:\n[^⚫\n]+)*?)(?=(?:[GRWLPFDMNYC]+\s+[A-Z]|Weakness|Resistance|Retreat|Illustrated|$))',
            re.MULTILINE
        )

        attacks = attack_pattern.findall(text)

        # If no attacks found, try alternative pattern
        if not attacks:
            # Pattern 2: Simpler pattern for basic attacks
            attack_pattern2 = re.compile(
                r'([GRWLPFDMNYC]{1,4})\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(\d+\+?)',
                re.MULTILINE
            )
            simple_attacks = attack_pattern2.findall(text)

            # For simple attacks, try to find effect text after damage
            for cost, name, damage in simple_attacks:
                # Look for text after this attack
                effect_pattern = re.compile(
                    rf'{re.escape(name)}\s+{re.escape(damage)}\s*\n?([^⚫\n]+(?:\n[^⚫\n]+)*?)(?=(?:[GRWLPFDMNYC]+\s+[A-Z]|Weakness|Resistance|Retreat|Illustrated|$))',
                    re.MULTILINE | re.DOTALL
                )
                effect_match = effect_pattern.search(text)
                effect = effect_match.group(1).strip() if effect_match else ""
                attacks.append((cost, name, damage, effect))

        for cost, name, damage, effect in attacks:
            cost = cost.strip()
            name = self._clean_text(name)
            damage = damage.strip()
            effect = self._clean_text(effect)

            # Validate attack name
            if len(name) < 50 and not re.search(r'Pokémon|Stage|HP|Weakness|Resistance', name):
                # Parse energy cost into array format for consistency
                cost_array = list(cost)  # Split into individual characters
                energy_types = []
                for energy_char in cost_array:
                    if energy_char in self.ENERGY_SYMBOLS:
                        energy_types.append(self.ENERGY_SYMBOLS[energy_char])
                    else:
                        energy_types.append('Colorless')

                # Clean up effect text - remove any trailing metadata
                if effect:
                    # Remove common ending patterns
                    effect = re.sub(r'\s*(Weakness|Resistance|Retreat|Illustrated by).*$', '', effect, flags=re.IGNORECASE)
                    effect = effect.strip()

                card.attacks.append({
                    'cost': cost_array,  # Now always an array
                    'name': name,
                    'damage': damage,
                    'text': effect,
                    'energy_types': energy_types
                })

    def _parse_flavor_text(self, card: PokemonCard, text: str, soup: BeautifulSoup):
        """Parse flavor text for Pokemon cards"""
        # Flavor text usually appears between the main card info and "Illustrated by"
        # It's typically in italics or a separate div

        # Method 1: Look for flavor text in HTML structure
        if soup:
            flavor_divs = soup.find_all(['div', 'p', 'span'], class_=re.compile('flavor|pokedex|description'))
            for div in flavor_divs:
                flavor = self._clean_text(div.get_text())
                if flavor and len(flavor) > 20 and 'Illustrated' not in flavor:
                    card.card_text = flavor
                    return

        # Method 2: Extract from text pattern
        # Flavor text is usually after all game text but before metadata
        flavor_pattern = re.compile(
            r'(?:Retreat:\s*\d+|resistance:\s*none)\s*\n+([^⚫\n]+(?:\n[^⚫\n]+)*?)(?=(?:Illustrated by|Regulation Mark|[A-Z]\s+Regulation|$))',
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        match = flavor_pattern.search(text)
        if match:
            potential_flavor = self._clean_text(match.group(1))
            # Validate it's likely flavor text (not game text)
            if (potential_flavor and
                len(potential_flavor) > 20 and
                not re.search(r'damage|attack|energy|pokemon|weakness|resistance', potential_flavor, re.IGNORECASE) and
                not re.search(r'^[GRWLPFDMNYC]+\s+', potential_flavor)):
                card.card_text = potential_flavor

    def _parse_trainer_details(self, card: PokemonCard, text: str, soup: BeautifulSoup):
        """Parse Trainer card details"""
        # Look for trainer type in the format "Trainer - Subtype"
        trainer_type_match = re.search(r'Trainer\s*[-–—]\s*(Item|Supporter|Stadium|Tool|Pokémon Tool)', text)
        if trainer_type_match:
            card.trainer_type = trainer_type_match.group(1).strip()
            # Normalize "Pokémon Tool" to "Tool"
            if card.trainer_type == 'Pokémon Tool':
                card.trainer_type = 'Tool'
        else:
            # Fallback detection
            if 'Supporter' in text:
                card.trainer_type = 'Supporter'
            elif 'Stadium' in text:
                card.trainer_type = 'Stadium'
            elif 'Tool' in text:
                card.trainer_type = 'Tool'
            else:
                card.trainer_type = 'Item'

        # Extract trainer effect text
        # The effect text appears after "Trainer - Type" and before "Illustrated by"
        # Pattern to capture everything between trainer type and illustrated by
        effect_pattern = re.compile(
            r'Trainer\s*[-–—]\s*(?:Item|Supporter|Stadium|Tool|Pokémon Tool)\s*\n+(.+?)(?=Illustrated by|Regulation Mark|©)',
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        match = effect_pattern.search(text)
        if match:
            effect_text = match.group(1).strip()
            # Clean up the text - remove extra whitespace and newlines
            effect_text = re.sub(r'\s+', ' ', effect_text)
            card.card_text = effect_text
        else:
            # Alternative pattern - look for text after trainer type
            lines = text.split('\n')
            effect_lines = []
            found_trainer = False

            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue

                # Check if we found the trainer type line
                if re.search(r'Trainer\s*[-–—]\s*(Item|Supporter|Stadium|Tool)', line):
                    found_trainer = True
                    continue

                # If we found trainer and this line contains effect text
                if found_trainer:
                    # Stop at metadata
                    if any(stop in line for stop in ['Illustrated by', 'Regulation Mark', '©', 'legal']):
                        break
                    # Add non-metadata lines
                    if line and not line.startswith('!'):  # Skip image links
                        effect_lines.append(line)

            if effect_lines:
                card.card_text = ' '.join(effect_lines)

        # Special rule handling
        if 'ACE SPEC' in text:
            card.rules.append("ACE SPEC rule — You can't have more than 1 ACE SPEC card in your deck.")

        # For Supporter cards, add the supporter rule if not already present
        if card.trainer_type == 'Supporter' and not any('Supporter rule' in rule for rule in card.rules):
            card.rules.append("Supporter rule — You may play only 1 Supporter card during your turn.")

        # For Stadium cards, add the stadium rule
        if card.trainer_type == 'Stadium' and not any('Stadium rule' in rule for rule in card.rules):
            card.rules.append("Stadium rule — This Stadium stays in play when you play it. Discard it if another Stadium comes into play.")

    def _parse_energy_details(self, card: PokemonCard, text: str, soup: BeautifulSoup):
        """Parse Energy card details"""
        # Determine if Basic or Special Energy
        if 'Basic' in card.name and 'Energy' in card.name:
            card.energy_type = 'Basic'
            # Extract the energy type from the name
            for energy_name, energy_type in [('Fire', 'Fire'), ('Water', 'Water'), ('Grass', 'Grass'),
                                            ('Lightning', 'Lightning'), ('Psychic', 'Psychic'),
                                            ('Fighting', 'Fighting'), ('Darkness', 'Darkness'),
                                            ('Metal', 'Metal'), ('Fairy', 'Fairy')]:
                if energy_name in card.name:
                    card.pokemon_type = energy_type  # Store the energy color type here
                    break
        elif 'Special Energy' in text:
            card.energy_type = 'Special'
        else:
            # Default to Special for non-basic energy
            card.energy_type = 'Special'

        # Extract energy effect for special energy
        if card.energy_type == 'Special':
            # Try to find the effect text
            effect_pattern = re.compile(
                r'Special Energy\s*\n+([^⚫]+?)(?=(?:Illustrated by|Regulation Mark|$))',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )

            match = effect_pattern.search(text)
            if match:
                card.card_text = self._clean_text(match.group(1))
            else:
                # Fallback: extract main text
                text_lines = text.split('\n')
                effect_lines = []
                capture = False

                for line in text_lines:
                    line = line.strip()
                    if 'Energy' in line and card.name in line:
                        capture = True
                        continue
                    if capture and line:
                        if any(x in line for x in ['Illustrated by', 'Regulation Mark', '©']):
                            break
                        effect_lines.append(line)

                card.card_text = ' '.join(effect_lines)

    def _parse_common_attributes(self, card: PokemonCard, text: str):
        """Parse attributes common to all card types"""
        # Regulation mark
        reg_mark_match = re.search(r'([A-Z])\s+Regulation Mark', text)
        if reg_mark_match:
            card.regulation_mark = reg_mark_match.group(1)

        # Rarity - updated patterns to match Limitless format
        rarity_patterns = [
            'Secret Rare', 'Ultra Rare', 'Hyper Rare', 'Full Art',
            'Rare Holo', 'Holo Rare', 'Rare', 'Uncommon', 'Common',
            'Special Illustration Rare', 'Illustration Rare'
        ]
        for rarity in rarity_patterns:
            if rarity in text:
                card.rarity = rarity
                break

        # Artist
        artist_match = re.search(r'Illustrated by\s+([^\n]+?)(?:\s*[A-Z]\s+Regulation Mark|$)', text)
        if artist_match:
            artist_name = artist_match.group(1).strip()
            # Clean up artist name - remove any trailing metadata
            artist_name = re.sub(r'\s*(•|More formats|legal|Standard).*$', '', artist_name)
            card.artist = artist_name

    def _get_set_cards(self, set_code: str) -> List[Tuple[str, str]]:
        """Get all card URLs and numbers from a specific set"""
        card_data = []
        page = 1
        cards_per_page = 50  # Limitless default seems to be 50

        while True:
            # Include cards per page parameter
            url = f"{self.CARDS_URL}?q=set:{set_code}&page={page}&cpp={cards_per_page}"
            response = self._make_request(url)

            if not response:
                break

            soup = BeautifulSoup(response.text, 'html.parser')

            # Check for "No cards found" message
            if 'No cards found' in response.text or '0 cards found' in response.text:
                break

            # Find all card links on the page
            # Limitless uses simple <a> tags with href="/cards/SET/NUMBER"
            all_links = soup.find_all('a', href=True)
            card_links = []

            for link in all_links:
                href = link.get('href', '')
                # Match pattern /cards/SET/NUMBER
                if re.match(r'^/cards/[A-Z0-9]+/\d+$', href):
                    card_links.append(link)

            # Remove duplicates based on href
            unique_hrefs = set()
            unique_card_links = []
            for link in card_links:
                href = link.get('href')
                if href not in unique_hrefs:
                    unique_hrefs.add(href)
                    unique_card_links.append(link)

            if not unique_card_links:
                # No cards found on this page
                if page == 1:
                    logger.warning(f"No cards found for set {set_code}")
                break

            # Process each unique card link
            page_cards = 0
            for link in unique_card_links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.BASE_URL, href)
                    # Extract the actual set code from the URL
                    url_match = re.search(r'/cards/([^/]+)/(\d+)', href)
                    if url_match:
                        actual_set_code = url_match.group(1).upper()
                        card_data.append((full_url, actual_set_code))
                        page_cards += 1

            logger.info(f"Found {page_cards} unique cards on page {page} of {set_code}")

            # Check if we should continue to next page
            # If we found less than cards_per_page, we're likely on the last page
            if page_cards < cards_per_page:
                # Double check by trying next page
                next_url = f"{self.CARDS_URL}?q=set:{set_code}&page={page + 1}&cpp={cards_per_page}"
                next_response = self._make_request(next_url)
                if next_response and 'No cards found' not in next_response.text:
                    # There are more pages
                    page += 1
                    continue
                else:
                    # No more pages
                    break
            else:
                # Full page, likely more pages exist
                page += 1

            # Safety check
            if page > 20:  # Most sets don't have more than 20 pages at 50 cards per page
                logger.warning(f"Reached page limit for set {set_code}")
                break

        logger.info(f"Total cards found in {set_code}: {len(card_data)}")
        return card_data

    def _scrape_card(self, url: str, actual_set_code: str = None) -> Optional[PokemonCard]:
        """Scrape a single card"""
        response = self._make_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        card = self._parse_pokemon_card(soup, url, actual_set_code)

        # Validate with reference data if available
        if card and self.api_reference_data:
            card = self._validate_with_reference(card)

            # If card type was wrong and it's a Pokemon, re-parse
            if card.card_type == 'Pokemon' and not card.hp:
                # Re-parse with correct type
                page_text = soup.get_text()
                self._parse_pokemon_details(card, page_text, soup)

        return card

    def scrape_standard_2025(self, max_workers: int = 5) -> List[PokemonCard]:
        """
        Scrape all cards legal in current Standard format

        Args:
            max_workers: Number of concurrent threads for scraping

        Returns:
            List of PokemonCard objects
        """
        # First, get the current Standard format sets dynamically
        self.standard_sets = self._get_current_standard_sets()

        if not self.standard_sets:
            logger.error("No Standard format sets found!")
            return []

        logger.info(f"Starting to scrape Standard format cards from {len(self.standard_sets)} sets")
        all_card_data = []

        # Get all card URLs and actual set codes from Standard sets
        for set_code in sorted(self.standard_sets):
            logger.info(f"Getting cards from set: {set_code}")
            card_data = self._get_set_cards(set_code)
            all_card_data.extend(card_data)
            logger.info(f"Found {len(card_data)} cards in {set_code}")

        logger.info(f"Total cards to scrape: {len(all_card_data)}")

        # Scrape cards concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {executor.submit(self._scrape_card, url, actual_set): (url, actual_set)
                            for url, actual_set in all_card_data}

            for future in as_completed(future_to_data):
                url, actual_set = future_to_data[future]
                try:
                    card = future.result()
                    if card:
                        self.cards_data.append(card)
                        # Safe logging with Unicode handling
                        try:
                            logger.info(f"Scraped: {card.name} ({card.set_code} {card.number})")
                        except UnicodeEncodeError:
                            # Fallback for problematic characters
                            safe_name = card.name.encode('ascii', 'replace').decode('ascii')
                            logger.info(f"Scraped: {safe_name} ({card.set_code} {card.number})")
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")

        logger.info(f"Scraping complete. Total cards: {len(self.cards_data)}")
        return self.cards_data

    def save_to_json(self, filename: str = 'pokemon_tcg_standard_2025.json'):
        """Save scraped data to JSON file"""
        data = {
            'metadata': {
                'format': 'Standard',
                'scrape_date': datetime.now().isoformat(),
                'total_cards': len(self.cards_data),
                'sets': sorted(list(self.standard_sets)),
                'regulation_marks': 'G, H, and newer'
            },
            'cards': [asdict(card) for card in self.cards_data]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Data saved to {filename}")

    def save_to_csv(self, filename: str = 'pokemon_tcg_standard_2025.csv'):
        """Save scraped data to CSV file"""
        import csv

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'name', 'set_code', 'number', 'card_type', 'pokemon_type', 'hp', 'stage', 'suffix',
                'evolves_from', 'weakness', 'resistance', 'retreat_cost', 'retreat_cost_array',
                'regulation_mark', 'rarity', 'artist', 'trainer_type', 'energy_type', 'rules'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for card in self.cards_data:
                row = {
                    'name': card.name,
                    'set_code': card.set_code,
                    'number': card.number,
                    'card_type': card.card_type,
                    'pokemon_type': card.pokemon_type,
                    'hp': card.hp,
                    'stage': card.stage,
                    'suffix': card.suffix,
                    'evolves_from': card.evolves_from,
                    'weakness': card.weakness,
                    'resistance': card.resistance,
                    'retreat_cost': card.retreat_cost,
                    'retreat_cost_array': ','.join(card.retreat_cost_array) if card.retreat_cost_array else '',
                    'regulation_mark': card.regulation_mark,
                    'rarity': card.rarity,
                    'artist': card.artist,
                    'trainer_type': card.trainer_type,
                    'energy_type': card.energy_type,
                    'rules': ' | '.join(card.rules) if card.rules else ''
                }
                writer.writerow(row)

        logger.info(f"Data saved to {filename}")


def main():
    """Main execution function"""
    scraper = LimitlessTCGScraper(delay=0.5, use_api_reference=False)  # Set to True to use API validation

    try:
        # Scrape all Standard format cards
        cards = scraper.scrape_standard_2025(max_workers=5)

        # Save data in multiple formats
        scraper.save_to_json()
        scraper.save_to_csv()

        # Print summary statistics
        print("\n=== Scraping Summary ===")
        print(f"Total cards scraped: {len(cards)}")

        # Count by card type
        type_counts = {}
        for card in cards:
            type_counts[card.card_type] = type_counts.get(card.card_type, 0) + 1

        print("\nCards by type:")
        for card_type, count in type_counts.items():
            print(f"  {card_type}: {count}")

        # Count by stage and suffix (for Pokemon)
        stage_counts = {}
        suffix_counts = {}
        for card in cards:
            if card.card_type == 'Pokemon':
                if card.stage:
                    stage_counts[card.stage] = stage_counts.get(card.stage, 0) + 1
                if card.suffix:
                    suffix_counts[card.suffix] = suffix_counts.get(card.suffix, 0) + 1

        print("\nPokemon by stage:")
        for stage, count in sorted(stage_counts.items()):
            print(f"  {stage}: {count}")

        print("\nPokemon by suffix:")
        for suffix, count in sorted(suffix_counts.items()):
            print(f"  {suffix}: {count}")

        # Count by set
        set_counts = {}
        for card in cards:
            set_counts[card.set_code] = set_counts.get(card.set_code, 0) + 1

        print("\nCards by set:")
        for set_code, count in sorted(set_counts.items()):
            print(f"  {set_code}: {count}")

        # Data quality check
        issues = []
        for card in cards:
            if card.card_type == 'Pokemon':
                if not card.hp:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing HP")
                if not card.stage:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing stage")
                if not card.pokemon_type:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing pokemon_type")
            elif card.card_type == 'Trainer':
                if not card.trainer_type:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing trainer_type")
                if not card.card_text:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing card_text")
            elif card.card_type == 'Energy':
                if not card.energy_type:
                    issues.append(f"{card.name} ({card.set_code} {card.number}): Missing energy_type")

        if issues:
            print(f"\n=== Data Quality Issues Found ({len(issues)}) ===")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")

    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()