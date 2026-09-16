# scripts/poker_gui.py
import sys
import os
import random
import glob
import torch
import pokers as pkrs
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFrame, QSizePolicy, QSlider, QComboBox,
                            QSpinBox, QDoubleSpinBox, QMessageBox, QGridLayout, QGroupBox,
                            QFileDialog, QFormLayout, QScrollArea, QTextBrowser)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QPalette

# Import the DeepCFR agent
from src.core.deep_cfr import DeepCFRAgent
from src.core.action_space import resolve_action
from src.core.checkpoint_kinds import (
    HU_STRATEGY_ONLY_CHECKPOINT_KIND,
    STRATEGY_ONLY_CHECKPOINT_KIND,
)
from src.core.game_contract import FIXED_HU_GAME_CONTRACT
from src.core.model import set_verbose
from src.agents.random_agent import RandomAgent
from policy_runtime.core import PolicyRuntimeAgent

class PolicyRuntimeAdapter:
    """Адаптер PolicyRuntimeAgent → интерфейс pkrs.Action для GUI."""

    def __init__(self, checkpoint_path, player_id, device):
        self.player_id = player_id
        self.agent = PolicyRuntimeAgent(checkpoint_path, device, min_action_prob=0.0)

    def choose_action(self, state):
        action_type = self.agent.choose_action(
            state, player_id=self.player_id, deterministic=False
        )
        return resolve_action(action_type, state).action


def create_playing_agent(model_path, player_id, device):
    """Загружает модель в адаптер, совместимый с игровым движком GUI."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if is_policy_runtime_checkpoint(checkpoint):
        return PolicyRuntimeAdapter(model_path, player_id=player_id, device=str(device))

    agent = DeepCFRAgent(player_id=player_id, device=device)
    agent.load_model(model_path)
    return agent


def is_policy_runtime_checkpoint(checkpoint):
    """Определяет light-checkpoint, который должен загружаться через policy runtime."""
    return (
        isinstance(checkpoint, dict)
        and checkpoint.get("checkpoint_kind")
        in {STRATEGY_ONLY_CHECKPOINT_KIND, HU_STRATEGY_ONLY_CHECKPOINT_KIND}
        and isinstance(checkpoint.get("strategy_net"), dict)
    )


_POSITION_NAMES = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"}


def table_player_ids(num_players):
    """Возвращает порядок мест за столом для поддерживаемого формата игры."""
    if num_players == 2:
        return (1, 0)
    if num_players == 6:
        return (1, 2, 3, 0, 5, 4)
    raise ValueError("GUI поддерживает только HU или 6-max")


def create_game_state(num_players, seed, stake=200.0, sb=1.0, bb=2.0):
    """Создаёт состояние раздачи, сохраняя неизменяемый контракт HU."""
    table_player_ids(num_players)
    if num_players == 2:
        stake = FIXED_HU_GAME_CONTRACT.starting_stack
        sb = FIXED_HU_GAME_CONTRACT.small_blind
        bb = FIXED_HU_GAME_CONTRACT.big_blind
    return pkrs.State.from_seed(
        n_players=num_players,
        button=seed % num_players,
        sb=sb,
        bb=bb,
        stake=stake,
        seed=seed,
    )


def position_label(player_id, button, num_players=6):
    rel = (player_id - button) % num_players
    if num_players == 2:
        return "BTN/SB" if rel == 0 else "BB"
    return _POSITION_NAMES.get(rel, f"P{player_id}")


class CardWidget(QLabel):
    """Widget to display a playing card"""
    def __init__(self, card=None, hidden=False, parent=None):
        super().__init__(parent)
        self.card = card
        self.hidden = hidden
        self.setMinimumSize(80, 120)
        self.setMaximumSize(80, 120)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
            }
        """)
        self.update_display()
    
    def update_display(self):
        """Update the card display"""
        if self.hidden:
            self.setText("🂠")
            return
            
        if self.card is None:
            self.setText("")
            self.setStyleSheet("background-color: transparent; border: none;")
            return
            
        # Convert rank to text representation
        rank_map = {
            0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
            7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A",
        }
        
        suit_map = {
            0: ("♣", "black"),
            1: ("♦", "red"),
            2: ("♥", "red"),
            3: ("♠", "black"),
        }
        
        rank_text = rank_map[int(self.card.rank)]
        suit_text, color = suit_map[int(self.card.suit)]
        
        self.setText(f"{rank_text}\n{suit_text}")
        self.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
                font-weight: bold;
                color: {color};
            }}
        """)
    
    def set_card(self, card, hidden=False):
        """Set the card to display"""
        self.card = card
        self.hidden = hidden
        self.update_display()


class PlayerWidget(QGroupBox):
    """Widget to display a player's information"""
    def __init__(self, player_id, is_human=False, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self.is_human = is_human
        self.name = "YOU" if is_human else f"AI Player {player_id}"
        self.setTitle(self.name)
        
        # Add some styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #999;
                border-radius: 5px;
                margin-top: 0.5em;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Card display
        card_layout = QHBoxLayout()
        self.card1 = CardWidget(hidden=not is_human)
        self.card2 = CardWidget(hidden=not is_human)
        card_layout.addWidget(self.card1)
        self.action_label = QLabel("")
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.hide()
        card_layout.addWidget(self.action_label)
        card_layout.addWidget(self.card2)
        layout.addLayout(card_layout)
        
        self.position_label = QLabel("")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setStyleSheet("font-size: 9pt; color: #666;")
        layout.addWidget(self.position_label)
        
        # Player info
        info_layout = QFormLayout()
        self.chips_label = QLabel("Chips: 200.0")
        self.bet_label = QLabel("Bet: 0.0")
        self.status_label = QLabel("Status: Active")
        
        info_layout.addRow("Chips:", self.chips_label)
        info_layout.addRow("Bet:", self.bet_label)
        info_layout.addRow("Status:", self.status_label)
        
        layout.addLayout(info_layout)
        self.setLayout(layout)
        
        # Highlight current player with a border
        self.active_style = """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #5c85d6;
                background-color: #e6f0ff;
                border-radius: 5px;
                margin-top: 0.5em;
                padding: 5px;
            }
        """
        self.inactive_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #999;
                border-radius: 5px;
                margin-top: 0.5em;
                padding: 5px;
            }
        """
        
        self.button_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #999;
                background-color: #f7f7c9;
                border-radius: 5px;
                margin-top: 0.5em;
                padding: 5px;
            }
        """

    _ACTION_COLORS = {
        "Fold": "#c0392b", "Check": "#27ae60", "Call": "#27ae60", "Raise": "#2980b9",
    }

    def set_position(self, pos_name):
        self.position_label.setText(pos_name)
        if pos_name:
            self.setTitle(f"{self.name} [{pos_name}]")
        else:
            self.setTitle(self.name)

    def set_action(self, action_text):
        if action_text:
            color = "#c0392b"
            for prefix, c in self._ACTION_COLORS.items():
                if action_text.startswith(prefix):
                    color = c
                    break
            self.action_label.setText(action_text)
            self.action_label.setStyleSheet(
                f"font-size: 14pt; font-weight: bold; color: {color};"
            )
            self.action_label.show()
        else:
            self.action_label.hide()

    def update_hand(self, hand, show_all=False):
        """Update the player's hand display"""
        if hand is None:
            self.card1.set_card(None)
            self.card2.set_card(None)
            return
            
        self.card1.set_card(hand[0], hidden=not (self.is_human or show_all))
        self.card2.set_card(hand[1], hidden=not (self.is_human or show_all))
    
    def update_info(self, chips, bet, active):
        """Update the player's information"""
        self.chips_label.setText(f"{chips:.1f}bb")
        self.bet_label.setText(f"{bet:.1f}bb")
        status_text = "Active" if active else "Folded"
        self.status_label.setText(status_text)
        
        if not active:
            self.set_action("Fold")
            self.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #999;
                    color: #999;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    margin-top: 0.5em;
                    padding: 5px;
                }
            """)
    
    def highlight_current(self, is_current=False, is_button=False):
        """Highlight the current player"""
        if is_current:
            self.setStyleSheet(self.active_style)
        elif is_button:
            self.setStyleSheet(self.button_style)
        else:
            self.setStyleSheet(self.inactive_style)


class PokerTable(QWidget):
    """Widget that displays the poker table and game information"""
    def __init__(self, num_players=6, human_player_id=0, parent=None):
        super().__init__(parent)
        if num_players not in (2, 6):
            raise ValueError("GUI поддерживает только HU или 6-max")
        if not 0 <= human_player_id < num_players:
            raise ValueError("Позиция человека выходит за пределы стола")
        self.num_players = num_players
        self.human_player_id = human_player_id
        self.setMinimumSize(800, 600)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Game info
        info_layout = QHBoxLayout()
        self.stage_label = QLabel("Stage: Preflop")
        self.pot_label = QLabel("Pot: $0.00")
        info_layout.addWidget(self.stage_label)
        info_layout.addWidget(self.pot_label)
        main_layout.addLayout(info_layout)
        
        # Community cards
        self.community_frame = QFrame()
        self.community_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.community_frame.setLineWidth(1)
        community_layout = QHBoxLayout()
        self.community_cards = [CardWidget() for _ in range(5)]
        for card in self.community_cards:
            community_layout.addWidget(card)
        self.community_frame.setLayout(community_layout)
        main_layout.addWidget(self.community_frame)
        
        # Player layouts
        self.players = []
        
        # Верхний ряд игроков.
        top_layout = QHBoxLayout()
        player_ids = table_player_ids(num_players)
        top_player_ids = player_ids[:num_players // 2]
        for i in top_player_ids:
            player = PlayerWidget(i, is_human=(i == human_player_id))
            self.players.append(player)
            top_layout.addWidget(player)
        main_layout.addLayout(top_layout)
        
        # Нижний ряд игроков.
        bottom_layout = QHBoxLayout()
        bottom_player_ids = player_ids[num_players // 2:]
        for i in bottom_player_ids:
            is_human = (i == human_player_id)
            player = PlayerWidget(i, is_human=is_human)
            self.players.append(player)
            bottom_layout.addWidget(player)
        main_layout.addLayout(bottom_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.fold_button = QPushButton("Fold")
        self.check_call_button = QPushButton("Check/Call")
        self.raise_button = QPushButton("Raise")
        
        # Create raise amount controls
        raise_layout = QVBoxLayout()
        raise_controls = QHBoxLayout()
        self.bb = 2.0
        self.raise_amount = QDoubleSpinBox()
        self.raise_amount.setRange(0, 1000)
        self.raise_amount.setValue(1.0)
        self.raise_amount.setSingleStep(0.5)
        self.raise_amount.setSuffix("bb")
        self.half_pot_button = QPushButton("½ Pot")
        self.pot_button = QPushButton("Pot")
        raise_controls.addWidget(self.raise_amount)
        raise_controls.addWidget(self.half_pot_button)
        raise_controls.addWidget(self.pot_button)
        raise_layout.addWidget(self.raise_button)
        raise_layout.addLayout(raise_controls)
        
        action_layout.addWidget(self.fold_button)
        action_layout.addWidget(self.check_call_button)
        action_layout.addLayout(raise_layout)
        main_layout.addLayout(action_layout)
        
        # Game controls
        controls_layout = QHBoxLayout()
        self.new_hand_button = QPushButton("New Hand")
        self.show_cards_button = QPushButton("Show All Cards")
        controls_layout.addWidget(self.new_hand_button)
        controls_layout.addWidget(self.show_cards_button)
        main_layout.addLayout(controls_layout)
        
        self.setLayout(main_layout)
        
        # Style the buttons
        for button in [self.fold_button, self.check_call_button, self.raise_button,
                      self.half_pot_button, self.pot_button, self.new_hand_button,
                      self.show_cards_button]:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4c72b0;
                    border: none;
                    color: white;
                    padding: 10px 15px;
                    font-size: 14px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #5c85d6;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
        
        # Give fold button a reddish color
        self.fold_button.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                border: none;
                color: white;
                padding: 10px 15px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        # Disable action buttons initially
        self.set_action_buttons_enabled(False)
    
    def update_community_cards(self, cards):
        """Update the community cards display"""
        for i, card_widget in enumerate(self.community_cards):
            if i < len(cards):
                card_widget.set_card(cards[i])
            else:
                card_widget.set_card(None)
    
    def update_pot(self, amount):
        """Update the pot display"""
        self.pot_label.setText(f"Pot: {amount:.1f}bb")
    
    def update_stage(self, stage):
        """Update the game stage display"""
        stage_names = {
            0: "Preflop",
            1: "Flop",
            2: "Turn",
            3: "River",
            4: "Showdown"
        }
        self.stage_label.setText(f"Stage: {stage_names.get(int(stage), str(stage))}")
    
    def update_players(self, player_states, current_player, button_position, show_all_cards=False):
        """Update all player displays"""
        bb = max(1.0, self.bb)
        for i, player in enumerate(self.players):
            # Find the corresponding player state
            player_state = next((ps for ps in player_states if ps.player == player.player_id), None)
            
            if player_state:
                player.update_hand(player_state.hand, show_all_cards)
                player.update_info(
                    chips=player_state.stake / bb,
                    bet=player_state.bet_chips / bb,
                    active=player_state.active
                )
                
                # Highlight current player and button position
                is_current = player.player_id == current_player
                is_button = player.player_id == button_position
                player.highlight_current(is_current, is_button)
                pos_name = position_label(player.player_id, button_position, len(player_states))
                player.set_position(pos_name)
    
    def set_action_buttons_enabled(self, enabled, legal_actions=None):
        """Enable or disable action buttons based on legal actions"""
        if not enabled:
            self.fold_button.setEnabled(False)
            self.check_call_button.setEnabled(False)
            self.raise_button.setEnabled(False)
            self.raise_amount.setEnabled(False)
            self.half_pot_button.setEnabled(False)
            self.pot_button.setEnabled(False)
            return
            
        if legal_actions:
            self.fold_button.setEnabled(pkrs.ActionEnum.Fold in legal_actions)
            
            # Update check/call button text and state
            if pkrs.ActionEnum.Check in legal_actions:
                self.check_call_button.setText("Check")
                self.check_call_button.setEnabled(True)
            elif pkrs.ActionEnum.Call in legal_actions:
                self.check_call_button.setText("Call")
                self.check_call_button.setEnabled(True)
            else:
                self.check_call_button.setEnabled(False)
            
            # Enable raise controls if raise is legal
            can_raise = pkrs.ActionEnum.Raise in legal_actions
            self.raise_button.setEnabled(can_raise)
            self.raise_amount.setEnabled(can_raise)
            self.half_pot_button.setEnabled(can_raise)
            self.pot_button.setEnabled(can_raise)
        else:
            # Default to all enabled
            self.fold_button.setEnabled(True)
            self.check_call_button.setEnabled(True)
            self.raise_button.setEnabled(True)
            self.raise_amount.setEnabled(True)
            self.half_pot_button.setEnabled(True)
            self.pot_button.setEnabled(True)
    
    def update_raise_limits(self, min_bet, max_bet):
        """Update the raise amount limits (доллары → bb)"""
        bb = max(1.0, self.bb)
        self.raise_amount.setRange(min_bet / bb, max_bet / bb)
        self.raise_amount.setValue(min_bet / bb)


class ModelSelectionDialog(QWidget):
    """Dialog for selecting model checkpoints"""
    def __init__(self, num_players=6, parent=None):
        super().__init__(parent)
        self.num_players = num_players
        self.setWindowTitle("Select AI Models")
        
        layout = QVBoxLayout()
        
        # Model directory selection
        dir_layout = QHBoxLayout()
        self.models_dir_input = QComboBox()
        self.models_dir_input.setEditable(True)
        self.models_dir_input.addItems(["models", "models_om", "models_mixed_om", "models_mixed_om_v2"])
        dir_layout.addWidget(QLabel("Models Directory:"))
        dir_layout.addWidget(self.models_dir_input, 1)
        self.browse_button = QPushButton("Browse...")
        dir_layout.addWidget(self.browse_button)
        layout.addLayout(dir_layout)
        
        # Number of models to load
        num_models_layout = QHBoxLayout()
        self.num_models_spinner = QSpinBox()
        self.num_models_spinner.setRange(1, num_players - 1)
        self.num_models_spinner.setValue(num_players - 1)
        num_models_layout.addWidget(QLabel("Number of AI opponents:"))
        num_models_layout.addWidget(self.num_models_spinner)
        layout.addLayout(num_models_layout)
        
        # Player position
        position_layout = QHBoxLayout()
        self.position_combo = QComboBox()
        if num_players == 2:
            self.position_combo.addItem("Hero (BTN/SB)")
            self.position_combo.setEnabled(False)
        else:
            self.position_combo.addItems(["BTN", "SB", "BB", "UTG", "MP", "CO"])
        self.position_combo.setCurrentIndex(0)
        position_layout.addWidget(QLabel("Your position:"))
        position_layout.addWidget(self.position_combo)
        layout.addLayout(position_layout)
        
        # Game settings
        game_layout = QFormLayout()
        self.stake_spinner = QDoubleSpinBox()
        self.stake_spinner.setRange(10, 1000)
        self.stake_spinner.setValue(200)
        self.stake_spinner.setSingleStep(10)
        
        self.sb_spinner = QDoubleSpinBox()
        self.sb_spinner.setRange(0.5, 10)
        self.sb_spinner.setValue(1)
        self.sb_spinner.setSingleStep(0.5)
        
        self.bb_spinner = QDoubleSpinBox()
        self.bb_spinner.setRange(1, 20)
        self.bb_spinner.setValue(2)
        self.bb_spinner.setSingleStep(1)

        if num_players == 2:
            self.stake_spinner.setValue(FIXED_HU_GAME_CONTRACT.starting_stack)
            self.sb_spinner.setValue(FIXED_HU_GAME_CONTRACT.small_blind)
            self.bb_spinner.setValue(FIXED_HU_GAME_CONTRACT.big_blind)
            self.stake_spinner.setEnabled(False)
            self.sb_spinner.setEnabled(False)
            self.bb_spinner.setEnabled(False)
        
        game_layout.addRow("Starting Chips:", self.stake_spinner)
        game_layout.addRow("Small Blind:", self.sb_spinner)
        game_layout.addRow("Big Blind:", self.bb_spinner)
        layout.addLayout(game_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Game")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.browse_button.clicked.connect(self.browse_models_dir)
        
        # Style
        self.setStyleSheet("""
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #999;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4c72b0;
                border: none;
                color: white;
                padding: 8px 12px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5c85d6;
            }
        """)
        
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 12px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
    
    def browse_models_dir(self):
        """Open file dialog to select models directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Models Directory")
        if dir_path:
            self.models_dir_input.setEditText(dir_path)


class PokerGUI(QMainWindow):
    """Main window for the poker GUI application"""
    def __init__(self, num_players=6):
        super().__init__()
        if num_players not in (2, 6):
            raise ValueError("GUI поддерживает только HU или 6-max")
        
        # Initialize variables
        self.num_players = num_players
        self.agents = None  # Initialize to None instead of empty list
        self.state = None
        self.human_player_id = 0
        self.show_all_cards = False
        self.game_in_progress = False
        self.last_player_actions = {}
        self.current_stage = -1
        
        # Setup UI
        self.init_ui()
        
        # Show model selection on startup
        self.show_model_selection()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("DeepCFR Poker AI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Poker table
        self.table = PokerTable(num_players=self.num_players, human_player_id=self.human_player_id)
        main_layout.addWidget(self.table)
        
        # Game history
        history_layout = QVBoxLayout()
        history_label = QLabel("Game History")
        history_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.history_text = QTextBrowser()
        self.history_text.setReadOnly(True)
        self.history_text.setPlainText("Welcome to DeepCFR Poker AI! Models will be loaded automatically.")
        self.history_text.setStyleSheet("background-color: #f5f5f5; padding: 8px; border-radius: 5px;")
        self.history_text.setMaximumHeight(150)
        self.history_text.setMinimumHeight(80)
        
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_text)
        main_layout.addLayout(history_layout)
        
        self.central_widget.setLayout(main_layout)
        
        # Connect signals
        self.table.fold_button.clicked.connect(lambda: self.human_action(pkrs.ActionEnum.Fold))
        self.table.check_call_button.clicked.connect(self.handle_check_call)
        self.table.raise_button.clicked.connect(lambda: self.human_action(pkrs.ActionEnum.Raise, self.table.raise_amount.value()))
        self.table.half_pot_button.clicked.connect(self.handle_half_pot)
        self.table.pot_button.clicked.connect(self.handle_pot)
        self.table.new_hand_button.clicked.connect(lambda: self.start_new_hand())
        self.table.show_cards_button.clicked.connect(self.toggle_show_cards)
        
        # Setup the model selection dialog
        self.model_dialog = ModelSelectionDialog(num_players=self.num_players, parent=self)
        self.model_dialog.start_button.clicked.connect(self.load_models_and_start)
        self.model_dialog.cancel_button.clicked.connect(lambda: self.close())
    
    def show_model_selection(self):
        """Show the model selection dialog"""
        self.model_dialog.show()
    
    def load_models_and_start(self):
        """Load selected models and start the game"""
        # Get dialog values
        models_dir = self.model_dialog.models_dir_input.currentText()
        num_models = self.model_dialog.num_models_spinner.value()
        self.human_player_id = self.model_dialog.position_combo.currentIndex()
        stake = self.model_dialog.stake_spinner.value()
        sb = self.model_dialog.sb_spinner.value()
        bb = self.model_dialog.bb_spinner.value()
        
        # Hide dialog
        self.model_dialog.hide()
        
        # Check if models directory exists
        if not os.path.isdir(models_dir):
            QMessageBox.warning(self, "Warning", f"Directory not found: {models_dir}")
            self.show_model_selection()
            return
        
        # Try to load AI models
        try:
            self.load_ai_models(models_dir, num_models)
            
            # Start the first hand
            self.start_new_hand(stake=stake, sb=sb, bb=bb)
            
            # Log startup info
            self.log_message(f"Game started with {num_models} AI opponents")
            self.log_message(f"You are Player {self.human_player_id}")
            self.log_message(f"Starting chips: ${stake}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            self.show_model_selection()
    
    def load_ai_models(self, models_dir, num_models):
        """Load AI models from the specified directory"""
        # Clear existing agents
        self.agents = [None] * self.num_players
        
        # Find model checkpoint files (exclude _light.pt which lacks advantage_net)
        model_files = [f for f in glob.glob(os.path.join(models_dir, "*.pt"))
                       if not f.endswith("_light.pt")]
        
        if not model_files:
            model_files = glob.glob(os.path.join(models_dir, "*_light.pt"))
            if model_files:
                self.log_message("Strategy-only mode: _light.pt (no advantage_net, ok for play)")
        
        if not model_files:
            self.log_message(f"No model files found in {models_dir}, using random agents")
            
            # Create random agents for all positions except human
            for i in range(self.num_players):
                if i != self.human_player_id:
                    self.agents[i] = RandomAgent(i)
            
            return
        
        # Select random models
        selected_models = random.sample(model_files, min(num_models, len(model_files)))
        
        # Log the selected models
        self.log_message("Selected AI models:")
        for i, model_path in enumerate(selected_models):
            model_name = os.path.basename(model_path)
            self.log_message(f"  {i+1}. {model_name}")
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models for positions other than human player's
        model_idx = 0
        for pos in range(self.num_players):
            if pos == self.human_player_id:
                continue
                
            if model_idx < len(selected_models):
                model_path = selected_models[model_idx]
                
                try:
                    agent = create_playing_agent(model_path, player_id=pos, device=device)
                    self.agents[pos] = agent
                    self.log_message(f"Loaded model for Player {pos}")
                    model_idx += 1
                    
                except Exception as e:
                    self.log_message(f"Error loading model for Player {pos}: {e}")
                    self.log_message("Using random agent instead")
                    self.agents[pos] = RandomAgent(pos)
            else:
                # Use random agent if no more models available
                self.agents[pos] = RandomAgent(pos)
                self.log_message(f"Using random agent for Player {pos}")
    
    def start_new_hand(self, stake=200.0, sb=1.0, bb=2.0):
        """Start a new poker hand"""
        # Make sure we have agents loaded
        if self.agents is None or not any(x is not None for x in self.agents):
            QMessageBox.warning(self, "Warning", "No AI agents loaded. Please select models first.")
            self.show_model_selection()
            return
        
        # Generate random seed for the hand
        seed = random.randint(0, 10000)
        
        # Create a new poker game
        self.state = create_game_state(
            num_players=self.num_players,
            seed=seed,
            stake=stake,
            sb=sb,
            bb=bb,
        )
        
        # Reset UI
        self.show_all_cards = False
        self.table.show_cards_button.setText("Show All Cards")
        self.last_player_actions = {}
        self.current_stage = -1
        for pw in self.table.players:
            pw.set_action("")
        self.update_ui()
        
        # Start the game loop if it's the human's turn first
        self.game_in_progress = True
        if self.state.current_player == self.human_player_id:
            self.process_human_turn()
        else:
            # Start AI turns
            QTimer.singleShot(1000, self.process_ai_turn)
    
    def update_ui(self):
        """Update the UI to match the current game state"""
        if not self.state:
            return

        if self.state.stage != self.current_stage:
            self.current_stage = int(self.state.stage)
            if self.current_stage > 0 and self.current_stage != 4:  # не сбрасываем на префлопе и showdown
                self.last_player_actions = {}
                for pw in self.table.players:
                    pw.set_action("")
            
        # Update community cards
        self.table.update_community_cards(self.state.public_cards)
        
        # Update game info
        bb = max(1.0, float(self.state.bb))
        self.table.bb = bb
        self.table.update_pot(self.state.pot / bb)
        self.table.update_stage(self.state.stage)
        
        # Update player displays
        self.table.update_players(
            self.state.players_state,
            self.state.current_player,
            self.state.button,
            self.show_all_cards
        )
        
        # Update action buttons
        is_human_turn = (self.state.current_player == self.human_player_id)
        self.table.set_action_buttons_enabled(
            is_human_turn and not self.state.final_state,
            self.state.legal_actions if is_human_turn else None
        )
        
        # Update raise amount limits
        if is_human_turn and pkrs.ActionEnum.Raise in self.state.legal_actions:
            player_state = self.state.players_state[self.human_player_id]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake
            
            # Calculate call amount and remaining after call
            call_amount = max(0, self.state.min_bet - current_bet)
            remaining_after_call = max(0, available_stake - call_amount)
            max_total = current_bet + available_stake
            
            # Minimum raise increment: max(bb, last_raise_increment)
            min_increment = max(1.0, float(self.state.bb))
            if hasattr(self.state, 'last_raise_increment') and self.state.last_raise_increment and float(self.state.last_raise_increment) > 0:
                min_increment = max(min_increment, float(self.state.last_raise_increment))
            
            # Spin box shows TOTAL bet amount (call + increment)
            if remaining_after_call >= min_increment:
                min_total = current_bet + call_amount + min_increment
                self.table.update_raise_limits(min_total, max_total)
            else:
                # Can only go all-in (remaining < min_increment, pokers will clamp)
                self.table.update_raise_limits(max_total, max_total)
            
            # Update buttons for raising half pot and pot
            half_pot_increment = max(self.state.pot * 0.5, min_increment)
            full_pot_increment = max(self.state.pot, min_increment)
            half_pot_total = current_bet + call_amount + min(half_pot_increment, remaining_after_call)
            full_pot_total = current_bet + call_amount + min(full_pot_increment, remaining_after_call)
            
            # Only enable if these amounts are within our stake
            self.table.half_pot_button.setEnabled(half_pot_total <= max_total)
            self.table.pot_button.setEnabled(full_pot_total <= max_total)
    
    def process_human_turn(self):
        """Process human player's turn"""
        # Enable action buttons based on legal actions
        self.table.set_action_buttons_enabled(True, self.state.legal_actions)
        
        # Log whose turn it is
        self.log_message("Your turn:")
        
        # Log legal actions
        action_text = []
        if pkrs.ActionEnum.Fold in self.state.legal_actions:
            action_text.append("Fold")
        if pkrs.ActionEnum.Check in self.state.legal_actions:
            action_text.append("Check")
        if pkrs.ActionEnum.Call in self.state.legal_actions:
            action_text.append("Call")
        if pkrs.ActionEnum.Raise in self.state.legal_actions:
            action_text.append("Raise")
        
        self.log_message(f"Available actions: {', '.join(action_text)}")
    
    def process_ai_turn(self):
        """Process the next AI player's turn"""
        if not self.state or self.state.final_state:
            return
            
        current_player = self.state.current_player
        
        # If it's the human's turn, let them play
        if current_player == self.human_player_id:
            self.process_human_turn()
            return
            
        # Let the AI choose an action
        agent = self.agents[current_player]
        
        if agent is None:
            # This shouldn't happen, but just in case
            self.log_message(f"Error: No agent for Player {current_player}")
            # Create a random agent
            agent = RandomAgent(current_player)
            self.agents[current_player] = agent
        
        # Choose action
        try:
            action = agent.choose_action(self.state)

            action_name = str(action.action).replace("ActionEnum.", "")
            if action.action == pkrs.ActionEnum.Raise:
                bb_act = max(1.0, float(self.state.bb))
                ps = self.state.players_state[current_player]
                call_amount = max(0.0, float(self.state.min_bet) - float(ps.bet_chips))
                total_street = (call_amount + action.amount) / bb_act
                action_name = f"Raise {total_street:.1f}bb"
            self.last_player_actions[current_player] = action_name
                
            # Log the action
            if action.action == pkrs.ActionEnum.Raise:
                ps = self.state.players_state[current_player]
                total_bet = ps.bet_chips + max(0, self.state.min_bet - ps.bet_chips) + action.amount
                bb = max(1.0, float(self.state.bb))
                self.log_message(f"Player {current_player} raises +{action.amount / bb:.1f}bb (total: {total_bet / bb:.1f}bb)")
            else:
                self.log_message(f"Player {current_player} {action.action}")
                
            # Apply the action with a delay
            self.state = self.state.apply_action(action)
            self.update_ui()

            action_name = self.last_player_actions.get(current_player, "")
            for pw in self.table.players:
                if pw.player_id == current_player:
                    pw.set_action(action_name)
                    break
            
            # Process end of hand or continue with next player
            if self.state.final_state:
                self.handle_end_of_hand()
            else:
                # Schedule next AI turn or human turn
                QTimer.singleShot(1000, self.process_ai_turn)
                
        except Exception as e:
            self.log_message(f"Error in AI turn: {str(e)}")
            # Try to recover by ending the hand
            self.game_in_progress = False
            self.table.set_action_buttons_enabled(False)
    
    def human_action(self, action_enum, amount=0):
        """Process human player's action"""
        if not self.state or self.state.final_state:
            return
            
        if self.state.current_player != self.human_player_id:
            return
            
        # Check if the action is legal
        if action_enum not in self.state.legal_actions:
            self.log_message(f"Illegal action: {action_enum}")
            return

        # Спиннер в bb → конвертировать в доллары
        bb = max(1.0, float(self.state.bb))
        amount = amount * bb
            
        # Convert absolute raise amount to increment over min_bet
        if action_enum == pkrs.ActionEnum.Raise and amount > 0:
            player_state = self.state.players_state[self.human_player_id]
            call_amount = max(0, self.state.min_bet - player_state.bet_chips)
            increment = max(0, amount - player_state.bet_chips - call_amount)
            amount = increment
            bb_raise = max(1.0, float(self.state.bb))
            self.log_message(f"You raise {increment / bb_raise:.1f}bb (total bet: {(player_state.bet_chips + call_amount + increment) / bb_raise:.1f}bb)")
        else:
            if action_enum == pkrs.ActionEnum.Raise:
                self.log_message(f"You raise")
            else:
                self.log_message(f"You {action_enum}")
        
        # Create the action
        action = pkrs.Action(action=action_enum, amount=amount)

        action_name = str(action_enum).replace("ActionEnum.", "")
        if action_enum == pkrs.ActionEnum.Raise and amount > 0:
            bb_raise = max(1.0, float(self.state.bb))
            total_street = (call_amount + increment) / bb_raise
            action_name = f"Raise {total_street:.1f}bb"
        self.last_player_actions[self.human_player_id] = action_name
            
        # Apply the action
        self.state = self.state.apply_action(action)
        self.update_ui()

        action_name = self.last_player_actions.get(self.human_player_id, "")
        for pw in self.table.players:
            if pw.player_id == self.human_player_id:
                pw.set_action(action_name)
                break
        
        # Process end of hand or continue with AI turns
        if self.state.final_state:
            self.handle_end_of_hand()
        else:
            # Schedule next AI turn
            QTimer.singleShot(1000, self.process_ai_turn)
    
    def handle_check_call(self):
        """Handle check or call button press"""
        if pkrs.ActionEnum.Check in self.state.legal_actions:
            self.human_action(pkrs.ActionEnum.Check)
        else:
            self.human_action(pkrs.ActionEnum.Call)
    
    def handle_half_pot(self):
        """Handle half pot button press"""
        player_state = self.state.players_state[self.human_player_id]
        call_amount = max(0, self.state.min_bet - player_state.bet_chips)
        remaining_after_call = max(0, player_state.stake - call_amount)
        half_pot_increment = max(self.state.pot * 0.5, 1.0)
        half_pot_increment = min(half_pot_increment, remaining_after_call)
        desired_total = player_state.bet_chips + call_amount + half_pot_increment
        bb = max(1.0, float(self.state.bb))
        self.table.raise_amount.setValue(min(desired_total / bb, self.table.raise_amount.maximum()))
        self.human_action(pkrs.ActionEnum.Raise, self.table.raise_amount.value())
    
    def handle_pot(self):
        """Handle pot button press"""
        player_state = self.state.players_state[self.human_player_id]
        call_amount = max(0, self.state.min_bet - player_state.bet_chips)
        remaining_after_call = max(0, player_state.stake - call_amount)
        full_pot_increment = max(self.state.pot, 1.0)
        full_pot_increment = min(full_pot_increment, remaining_after_call)
        desired_total = player_state.bet_chips + call_amount + full_pot_increment
        bb = max(1.0, float(self.state.bb))
        self.table.raise_amount.setValue(min(desired_total / bb, self.table.raise_amount.maximum()))
        self.human_action(pkrs.ActionEnum.Raise, self.table.raise_amount.value())
    
    def handle_end_of_hand(self):
        """Handle the end of a hand"""
        self.game_in_progress = False
        self.table.set_action_buttons_enabled(False)
        
        # Show all cards
        self.show_all_cards = True
        self.table.update_players(
            self.state.players_state,
            self.state.current_player,
            self.state.button,
            self.show_all_cards
        )

        for player_id, action_name in self.last_player_actions.items():
            for pw in self.table.players:
                if pw.player_id == player_id:
                    pw.set_action(action_name)
                    break
        
        # Log the results
        self.log_message("--- Hand Complete ---")
        
        # Show all players' hands and results
        for player in self.state.players_state:
            player_type = "You" if player.player == self.human_player_id else f"Player {player.player}"
            result_str = f"won ${player.reward:.2f}" if player.reward > 0 else f"lost ${-player.reward:.2f}"
            
            hand_str = ""
            if player.hand:
                card1 = card_to_string(player.hand[0])
                card2 = card_to_string(player.hand[1])
                hand_str = f" with {card1} {card2}"
                
            self.log_message(f"{player_type} {result_str}{hand_str}")
    
    def toggle_show_cards(self):
        """Toggle showing all cards"""
        self.show_all_cards = not self.show_all_cards
        
        if self.show_all_cards:
            self.table.show_cards_button.setText("Hide Cards")
        else:
            self.table.show_cards_button.setText("Show All Cards")
            
        # Update the UI
        if self.state:
            self.table.update_players(
                self.state.players_state,
                self.state.current_player,
                self.state.button,
                self.show_all_cards
            )
    
    def log_message(self, message):
        """Add a message to the history display"""
        current_text = self.history_text.toPlainText()
        new_text = f"{message}\n{current_text}"
        
        # Limit length to prevent memory issues
        max_lines = 100
        lines = new_text.split('\n')
        if len(lines) > max_lines:
            new_text = '\n'.join(lines[:max_lines])
            
        self.history_text.setPlainText(new_text)
        self.history_text.verticalScrollBar().setValue(0)


def card_to_string(card):
    """Convert a poker card to a readable string"""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 
             7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
    
    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DeepCFR Poker GUI')
    
    # Model loading options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--models', nargs='+', help='Specific model paths to use as opponents')
    model_group.add_argument('--models_folder', type=str, help='Folder containing model checkpoints')
    
    # Game settings
    parser.add_argument('--hu', action='store_true', help='Запустить двухместный HU-стол')
    parser.add_argument('--position', type=int, default=0, help='Your position at the table (0-5)')
    parser.add_argument('--stake', type=float, default=200.0, help='Initial chip stack')
    parser.add_argument('--sb', type=float, default=1.0, help='Small blind amount')
    parser.add_argument('--bb', type=float, default=2.0, help='Big blind amount')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up the application
    app = QApplication(sys.argv)
    num_players = 2 if args.hu else 6
    window = PokerGUI(num_players=num_players)
    
    # Initialize with command line arguments if provided
    if args.models or args.models_folder:
        window.model_dialog.hide()
        window.agents = [None] * num_players
        
        # Load specified models
        if args.models:
            window.log_message(f"Loading specified models: {args.models}")
            window.human_player_id = args.position
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model_idx = 0
            for pos in range(num_players):
                if pos == window.human_player_id:
                    continue
                    
                if model_idx < len(args.models):
                    model_path = args.models[model_idx]
                    
                    try:
                        agent = create_playing_agent(model_path, player_id=pos, device=device)
                        window.agents[pos] = agent
                        window.log_message(f"Loaded model for Player {pos}: {os.path.basename(model_path)}")
                        model_idx += 1
                        
                    except Exception as e:
                        window.log_message(f"Error loading model for Player {pos}: {e}")
                        window.log_message("Using random agent instead")
                        window.agents[pos] = RandomAgent(pos)
                else:
                    # Use random agent if no more models available
                    window.agents[pos] = RandomAgent(pos)
                    window.log_message(f"Using random agent for Player {pos}")
        
        # Load models from folder
        elif args.models_folder:
            window.log_message(f"Loading models from folder: {args.models_folder}")
            window.human_player_id = args.position
            
            # Find model files
            model_files = glob.glob(os.path.join(args.models_folder, "*.pt"))
            
            if not model_files:
                window.log_message(f"No model files found in {args.models_folder}")
                # Create random agents
                for pos in range(6):
                    if pos != window.human_player_id:
                        window.agents[pos] = RandomAgent(pos)
            else:
                # Use up to 5 random models from the folder
                num_models = min(num_players - 1, len(model_files))
                selected_models = random.sample(model_files, num_models)
                window.log_message(f"Selected {num_models} models from folder")
                
                # Load selected models
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_idx = 0
                
                for pos in range(num_players):
                    if pos == window.human_player_id:
                        continue
                        
                    if model_idx < len(selected_models):
                        model_path = selected_models[model_idx]
                        
                        try:
                            agent = create_playing_agent(model_path, player_id=pos, device=device)
                            window.agents[pos] = agent
                            window.log_message(f"Loaded model for Player {pos}: {os.path.basename(model_path)}")
                            model_idx += 1
                            
                        except Exception as e:
                            window.log_message(f"Error loading model for Player {pos}: {e}")
                            window.log_message("Using random agent instead")
                            window.agents[pos] = RandomAgent(pos)
                    else:
                        # Use random agent if no more models available
                        window.agents[pos] = RandomAgent(pos)
                        window.log_message(f"Using random agent for Player {pos}")
        
        # Start a new hand with the loaded models
        window.start_new_hand(stake=args.stake, sb=args.sb, bb=args.bb)
    
    window.show()
    sys.exit(app.exec_())
