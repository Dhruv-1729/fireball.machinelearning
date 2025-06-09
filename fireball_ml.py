import random
import numpy as np
import pickle
from colorama import Fore # Make sure colorama is installed: pip install colorama
from collections import defaultdict
import copy


class FireballQLearning:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, lambda_val=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.eligibility_traces = defaultdict(lambda: defaultdict(float))
        self.move_history = []
        self.opp_move_history = []  # NEW: Add a list to track opponent moves

    def update_histories(self, my_action, opp_action):
        """NEW: A helper method to update both move histories."""
        self.move_history.append(my_action)
        if len(self.move_history) > 4: # Keep history to a reasonable size
            self.move_history.pop(0)

        self.opp_move_history.append(opp_action)
        if len(self.opp_move_history) > 4:
            self.opp_move_history.pop(0)

    def get_state(self, my_charges, opp_charges):
        """NEW: Creates a richer state that includes both players' move patterns."""
        my_charges_c = min(max(my_charges, 0), 10)
        opp_charges_c = min(max(opp_charges, 0), 10)
        
        # Use the last 3 moves of each player for the pattern to detect strategies
        my_recent_pattern = "_".join(self.move_history[-3:]) if len(self.move_history) > 0 else "start"
        opp_recent_pattern = "_".join(self.opp_move_history[-3:]) if len(self.opp_move_history) > 0 else "start"
        
        # The new state is much more descriptive
        return f"mc_{my_charges_c}_oc_{opp_charges_c}_mypatt_{my_recent_pattern}_opppatt_{opp_recent_pattern}"
    
    @staticmethod
    def get_legal_moves(charges):
        if charges == 0:
            return ["charge", "shield"]
        elif charges == 1:
            return ["charge", "fireball", "shield"]
        elif 2 <= charges < 5:
            return ["charge", "fireball", "iceball", "shield"]
        else: # charges >= 5
            return ["charge", "fireball", "iceball", "shield", "megaball"]
    
    def choose_action(self, state, legal_moves, training=True):
        if len(self.move_history) >= 6:
            recent_moves = self.move_history[-6:]
            if len(set(recent_moves)) <= 2:
                if training:
                    self.epsilon = min(0.5, self.epsilon * 1.5) 
        
        # Removed strategy_variant specific logic
        
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves) if legal_moves else "charge"
        else:
            if not legal_moves: 
                return "charge" 
            if state not in self.q_table: 
                return random.choice(legal_moves)
            
            legal_q_values = {move: self.q_table[state][move] for move in legal_moves}
            for move in legal_q_values:
                legal_q_values[move] += random.uniform(-0.01, 0.01) 
            best_move = max(legal_q_values, key=legal_q_values.get)
            return best_move
    

    def save_model(self, filename="fireball_ai_model.pkl"): # Default filename
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f) 
    
    def load_model(self, filename="fireball_ai_model.pkl"): # Default filename
        try:
            with open(filename, 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded_table)
            return True
        except FileNotFoundError:
            print(f"Warning: Model file '{filename}' not found. Starting with a fresh Q-table.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a fresh Q-table.")
            return False

class FireballGame:
    def __init__(self):
        self.moves = ["charge", "fireball", "iceball", "shield", "megaball"]
        self.reset_game()
    
    def reset_game(self):
        self.player_charges = 0
        self.comp_charges = 0
        self.game_over = False
        self.winner = None
    
    def get_move_cost(self, move):
        costs = {"charge": -1, "fireball": 1, "iceball": 2, "shield": 0, "megaball": 5}
        return costs.get(move, 0) 
    
    def execute_turn(self, player_move, comp_move):
        self.player_charges -= self.get_move_cost(player_move)
        self.comp_charges -= self.get_move_cost(comp_move)
        self.player_charges = max(0, self.player_charges)
        self.comp_charges = max(0, self.comp_charges)
        result = self.determine_winner(player_move, comp_move)
        if result != "continue":
            self.game_over = True
            self.winner = result
        return result
    
    def determine_winner(self, p1_move, p2_move):
        """Determines the winner based on standard hierarchical game rules.
           p1 is player1 (can be human or AI1), p2 is player2 (can be AI or AI2).
           Returns 'player1', 'player2', or 'continue'.
        """
        if p1_move == p2_move and p1_move != "megaball":
            return "continue"

        if p1_move == "megaball":
            return "player1" if p2_move != "megaball" else "continue"
        if p2_move == "megaball": # p1_move is not megaball here
            return "player2"

        if p1_move == "shield": # and p2_move is not megaball
            return "continue" 
        if p2_move == "shield": # and p1_move is not megaball
            return "continue"

        # Charge, Fireball, Iceball interactions (no shields, no megaballs)
        if p1_move == "charge":
            if p2_move in ["fireball", "iceball"]: return "player2"
        elif p1_move == "fireball":
            if p2_move == "charge": return "player1"
            if p2_move == "iceball": return "player2" 
        elif p1_move == "iceball":
            if p2_move == "charge": return "player1"
            if p2_move == "fireball": return "player1"
            
        return "continue" # Should ideally not be reached if all defined pairs are covered
    
def train_ai(episodes=100000, self_play_ratio=0.4, verbose=True):
    """
    Trains an AI model using Q-Learning with Eligibility Traces (Q(λ))
    and Self-Play for more robust opponent training.
    """
    # Initialize the AI with the lambda parameter for eligibility traces
    ai = FireballQLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.5, lambda_val=0.9)
    frozen_opponent_ai = None  # This will hold the frozen copy of the AI for self-play

    print(f"{Fore.BLUE}Starting training with Self-Play (Ratio: {self_play_ratio*100}%) and Q(λ)...{Fore.RESET}")

    for episode in range(episodes):
        # Periodically update the frozen opponent to a newer, smarter version of the AI
        if episode > 0 and episode % 10000 == 0:
            if verbose:
                print(f"{Fore.MAGENTA}  Updating self-play opponent at episode {episode}...{Fore.RESET}")
            # Create a new AI instance for the opponent with a little exploratory behavior
            frozen_opponent_ai = FireballQLearning(epsilon=0.05) 
            # Deep copy the main AI's "brain" (Q-table) to the frozen opponent
            frozen_opponent_ai.q_table = copy.deepcopy(ai.q_table)

        # --- Episode Initialization ---
        game = FireballGame()
        ai.eligibility_traces.clear()
        ai.move_history = []

        # Get the initial state and choose the first action
        state = ai.get_state(game.comp_charges, game.player_charges)
        legal_moves = FireballQLearning.get_legal_moves(game.comp_charges)
        action = ai.choose_action(state, legal_moves, training=True)

        # --- Main Game Loop for a Single Episode ---
        while not game.game_over:
            ai_charges_at_decision = game.comp_charges
            player_charges_at_decision = game.player_charges

            # --- OPPONENT SELECTION: Self-Play or Scripted Bot ---
            use_self_play = frozen_opponent_ai is not None and random.random() < self_play_ratio

            if use_self_play:
                # Opponent is the frozen copy of the AI
                opp_state = frozen_opponent_ai.get_state(game.player_charges, game.comp_charges)
                opp_legal_moves = FireballQLearning.get_legal_moves(game.player_charges)
                player_move = frozen_opponent_ai.choose_action(opp_state, opp_legal_moves, training=False)
            else:
                # Opponent is one of the simple scripted bots
                player_legal_moves = FireballQLearning.get_legal_moves(player_charges_at_decision)
                if not player_legal_moves:
                    player_move = "charge"
                else:
                    if episode % 3 == 0: player_move = random.choice(player_legal_moves)
                    elif episode % 3 == 1:
                        if "megaball" in player_legal_moves: player_move = "megaball"
                        elif "iceball" in player_legal_moves: player_move = "iceball"
                        else: player_move = "fireball" if "fireball" in player_legal_moves else random.choice(player_legal_moves)
                    else:
                        if "shield" in player_legal_moves and random.random() < 0.7: player_move = "shield"
                        else: player_move = random.choice(player_legal_moves)

            # AI's chosen 'action' becomes the 'comp_move' for this turn
            comp_move = action
            raw_game_result = game.execute_turn(player_move, comp_move)

            ai.update_histories(comp_move, player_move)
            if use_self_play:
                frozen_opponent_ai.update_histories(player_move, comp_move)

# --- Detailed Reward Shaping Calculation (with your tweaks) ---
# --- Detailed Reward Shaping Calculation (with your tweaks) ---
            reward = 0.0
            if raw_game_result == "player2":  # AI won
                # Your dynamic win reward
                final_charge_diff = game.comp_charges - game.player_charges
                reward = 15.0 + (final_charge_diff * 0.5) 
            elif raw_game_result == "player1":  # AI lost
                # Your dynamic loss reward
                final_charge_diff = game.comp_charges - game.player_charges
                reward = -15.0 + (final_charge_diff * 0.5)
            else:  # "continue"
                reward = -0.10  # Your base penalty for non-decisive moves
                # First, check for High-Alert Endgame Scenarios
                if player_charges_at_decision == 4: # CRITICAL THREAT: Opponent is 1 charge away from winning.
                    if comp_move in ["fireball", "iceball", "megaball"]:
                        # AI *must* attack to force a shield or win. This is the only good play.
                        reward += 5.0 
                    elif comp_move == 'shield':
                        # The worst possible move. Guarantees opponent can win next turn.
                        reward -= 8.0 
                    elif comp_move == 'charge':
                        # Also a terrible move. The AI can't win a charging race from here.
                        reward -= 7.0 

                elif player_charges_at_decision == 3: # HIGH THREAT: Opponent is 2 charges away.
                    if comp_move in ["fireball", "iceball", "megaball"]:
                        # Good move. Puts pressure on the opponent.
                        reward += 2.0
                    elif comp_move == 'shield':
                        # A very passive move. Cedes control to the opponent.
                        reward -= 3.0 
                    elif comp_move == 'charge':
                         # Charging to keep pace is risky but better than shielding.
                        reward -= 1.0

                # If not an endgame scenario, use your standard logic
                else: # This block now handles player_charges_at_decision of 0, 1, or 2
                    # Your charge advantage logic
                    charge_difference = ai_charges_at_decision - player_charges_at_decision
                    reward += charge_difference * 0.20

                    if comp_move == "charge":
                        if ai_charges_at_decision < 3: reward += 1.0
                        elif 3 <= ai_charges_at_decision < 6:
                            reward += (0.5 + 0.1 * (ai_charges_at_decision + 0.5))
                        elif ai_charges_at_decision >= 6:
                            reward -= 10
                        # Note: The penalty for charging when opp has 3+ charges is now handled by the endgame logic above.
                            
                    elif comp_move == "shield":
                        if player_charges_at_decision == 0: reward -= 8.0
                        elif player_charges_at_decision == 1:
                            if player_move == "fireball": reward += 1.0
                            elif player_move == "charge": reward -= 2.5
                            else: reward -= 0.25
                        elif player_charges_at_decision == 2:
                            if player_move in ["fireball", "iceball"]: reward += 1.0
                            elif player_move == "charge": reward -= 4.5
                            # I've corrected your typo from "sheild" to "shield" here
                            elif player_move == "shield": reward += 0.30

                    elif comp_move == "fireball":
                        if player_move == "shield": reward -= 0.41
                        elif player_move == "fireball": reward += 0.1
                    
                    elif comp_move == "iceball":
                        if player_move == "shield": reward -= 0.425
                        elif player_move == "iceball": reward += 0.20
                        
                    elif comp_move == "megaball":
                        # Your reward for using megaball in a standard situation
                        reward += 14.0
                    elif player_charges_at_decision == 0:
                        if comp_move == "shield": reward -= 7.0


            # --- Q(λ) Update Logic (remains the same) ---
            next_state = ai.get_state(game.comp_charges, game.player_charges)
            next_legal_moves = FireballQLearning.get_legal_moves(game.comp_charges)
            next_action = ai.choose_action(next_state, next_legal_moves, training=True)

            if game.game_over: max_next_q = 0.0
            else: max_next_q = max([ai.q_table[next_state][m] for m in next_legal_moves], default=0.0)

            delta = reward + (ai.discount_factor * max_next_q) - ai.q_table[state][action]
            ai.eligibility_traces[state][action] += 1

            states_to_update = list(ai.eligibility_traces.keys())
            for s_trace in states_to_update:
                actions_to_update = list(ai.eligibility_traces[s_trace].keys())
                for a_trace in actions_to_update:
                    ai.q_table[s_trace][a_trace] += ai.learning_rate * delta * ai.eligibility_traces[s_trace][a_trace]
                    ai.eligibility_traces[s_trace][a_trace] *= ai.discount_factor * ai.lambda_val
            
            state = next_state
            action = next_action

        # --- Epsilon Decay and Logging ---
        if (episode + 1) % 1000 == 0:
            ai.epsilon = max(0.01, ai.epsilon * 0.96)
        if verbose and ((episode + 1) % 5000 == 0 or episode == episodes - 1):
            print(f"  Episode {episode + 1}/{episodes}, Epsilon: {ai.epsilon:.4f}")

    if verbose:
        print(f"{Fore.GREEN}\nTraining completed!{Fore.RESET}")
    return ai
def play_vs_ai():
    ai = FireballQLearning(epsilon=0) 
    if not ai.load_model():
        print("Could not load AI model. Please train the AI first (option 1).")
        return
    
    game = FireballGame()
    print(f'{Fore.GREEN}Game started! You vs AI{Fore.RESET}')
    print(f"Available moves: {', '.join(game.moves)}") 
    print("Type 'end' to quit")
    
    while not game.game_over:
        print(f"\nYour charges: {Fore.CYAN}{game.player_charges}{Fore.RESET}, AI charges: {Fore.RED}{game.comp_charges}{Fore.RESET}")
        
        # AI is comp (p2), Player is p1
        player_legal_moves = ai.get_legal_moves(game.player_charges) 
        comp_legal_moves = ai.get_legal_moves(game.comp_charges)
        print(f"Your legal moves: {Fore.YELLOW}{player_legal_moves}{Fore.RESET}")
        
        player_move = ""
        while True:
            player_move = input("Enter your move: ").lower()
            if player_move == "end": print("Game ended by player."); return
            if player_move not in game.moves: # Corrected: check against game.moves
                print(f"{Fore.RED}Invalid move name! Choose from {game.moves}{Fore.RESET}")
                continue
            if player_move not in player_legal_moves:
                print(f"{Fore.RED}Not a legal move with your current charges! Choose from {player_legal_moves}{Fore.RESET}")
                continue
            break
        
        ai_state = ai.get_state(game.comp_charges, game.player_charges) # AI's perspective
        comp_move = ai.choose_action(ai_state, comp_legal_moves, training=False) if comp_legal_moves else "charge"
        
        # game.execute_turn expects (player1_move, player2_move)
        # Human is player1, AI (comp) is player2
        result = game.execute_turn(player_move, comp_move)
        ai.update_histories(comp_move, player_move)

        print(f'{Fore.BLUE}You chose: {player_move}{Fore.RESET}')
        print(f'{Fore.MAGENTA}AI chose: {comp_move}{Fore.RESET}')
        
        if result == "player1": # Human won
            print(f'{Fore.GREEN}You won! :){Fore.RESET}')
        elif result == "player2": # AI won
            print(f'{Fore.RED}AI won! :({Fore.RESET}')
        elif result == "continue":
            pass # Game continues
        else: # Should not happen if logic is correct
            print(f"{Fore.YELLOW}Game over, but result unclear: {result}{Fore.RESET}")
        
        if game.game_over: break


def ai_vs_ai_demo():
    # Load the single trained AI model for both players in the demo
    ai1 = FireballQLearning(epsilon=0.05) # Small exploration for ai1
    ai2 = FireballQLearning(epsilon=0.05) # Small exploration for ai2
    
    loaded_ai1 = ai1.load_model()
    loaded_ai2 = ai2.load_model() # Both load the same "fireball_ai_model.pkl"

    if not loaded_ai1 or not loaded_ai2:
        print("Could not load the AI model for the demo. Please train the AI first (option 1).")
        return
            
    game = FireballGame()
    print(f'\n{Fore.GREEN}AI vs AI Demo - Strategic Battle!{Fore.RESET}')
    
    turn = 0
    max_turns = 50 # Increased max turns for demo
    while not game.game_over and turn < max_turns:
        # AI1 plays as "player1" in game logic
        ai1_state = ai1.get_state(game.player_charges, game.comp_charges) 
        ai1_legal_moves = ai1.get_legal_moves(game.player_charges)
        ai1_move = ai1.choose_action(ai1_state, ai1_legal_moves, training=False) if ai1_legal_moves else "charge"
        
        # AI2 plays as "player2" (comp) in game logic
        ai2_state = ai2.get_state(game.comp_charges, game.player_charges) 
        ai2_legal_moves = ai2.get_legal_moves(game.comp_charges)
        ai2_move = ai2.choose_action(ai2_state, ai2_legal_moves, training=False) if ai2_legal_moves else "charge"
        
        # AI objects track their own move history internally via update_q_value or choose_action
        # No need to manually update ai.move_history here for non-training.
        
        result = game.execute_turn(ai1_move, ai2_move) # ai1 is p1, ai2 is p2
        ai1.update_histories(ai1_move, ai2_move)
        ai2.update_histories(ai2_move, ai1_move)
        
        # Display actual charges after the move for clarity
        # game.player_charges is AI1's charges, game.comp_charges is AI2's charges
        print(f"\nTurn {turn + 1}:")
        print(f"  AI1 ({Fore.CYAN}charges: {game.player_charges}{Fore.RESET}) chose: {Fore.BLUE}{ai1_move}{Fore.RESET}")
        print(f"  AI2 ({Fore.RED}charges: {game.comp_charges}{Fore.RESET}) chose: {Fore.MAGENTA}{ai2_move}{Fore.RESET}")
        
        if result == "player1": 
            print(f'{Fore.GREEN}AI1 wins!{Fore.RESET}')
        elif result == "player2": 
            print(f'{Fore.RED}AI2 wins!{Fore.RESET}')
        elif result == "continue":
            print(f"  Result: {Fore.YELLOW}Continue{Fore.RESET}")
        
        if game.game_over: break
        turn += 1
    
    if not game.game_over and turn >= max_turns:
        print(f"\nGame ended after {max_turns} turns: {Fore.YELLOW}Strategic Stalemate or Max Turns Reached{Fore.RESET}")

class RandomBotController:
    """
    A simple controller that chooses a random legal move.
    """
    def choose_move(self, my_charges, legal_moves_list): # opponent_charges is not needed for this simple random bot
        if not legal_moves_list:
            # This case should ideally not be hit if game.get_legal_moves always returns valid moves
            return "charge" 
        return random.choice(legal_moves_list)

def simulate_one_game(player1_controller, player1_is_ai, 
                      player2_controller, player2_is_ai, 
                      game_instance, max_turns=50):
    game_instance.reset_game()

    if player1_is_ai: player1_controller.move_history = []
    if player2_is_ai: player2_controller.move_history = []

    for _turn_num in range(max_turns):
        # Player 1's (p1) move
        p1_current_charges = game_instance.player_charges
        # CORRECTED CALL: Use FireballQLearning's static method
        p1_legal_moves = FireballQLearning.get_legal_moves(p1_current_charges) 

        if player1_is_ai:
            p1_opponent_charges = game_instance.comp_charges
            p1_state = player1_controller.get_state(p1_current_charges, p1_opponent_charges)
            p1_move = player1_controller.choose_action(p1_state, p1_legal_moves, training=False)
        else: 
            p1_move = player1_controller.choose_move(p1_current_charges, p1_legal_moves)

        # Player 2's (p2) move
        p2_current_charges = game_instance.comp_charges
        # CORRECTED CALL: Use FireballQLearning's static method
        p2_legal_moves = FireballQLearning.get_legal_moves(p2_current_charges) 

        if player2_is_ai:
            p2_opponent_charges = game_instance.player_charges 
            p2_state = player2_controller.get_state(p2_current_charges, p2_opponent_charges)
            p2_move = player2_controller.choose_action(p2_state, p2_legal_moves, training=False)
        else: 
            p2_move = player2_controller.choose_move(p2_current_charges, p2_legal_moves)

        result = game_instance.execute_turn(p1_move, p2_move)

        if game_instance.game_over:
            return game_instance.winner # 'player1' or 'player2'

    return "draw"

def run_evaluation_simulations(num_games=75000):
    """
    Runs two sets of simulations: RandomBot vs RandomBot, and AI vs RandomBot.
    Prints the win percentages.
    """
    print(f"{Fore.YELLOW}\n--- Starting Evaluation Simulations ({num_games} games per set) ---{Fore.RESET}")
    print(f"{Fore.CYAN}Note: Using standardized game logic from FireballGame for all simulations.{Fore.RESET}")
    
    game_instance = FireballGame() # Use one game instance, reset for each game
    
    # --- Simulation 1: Random Bot vs. Random Bot ---
    print(f"{Fore.GREEN}\nRunning: Random Bot 1 vs. Random Bot 2...{Fore.RESET}")
    rb1 = RandomBotController()
    rb2 = RandomBotController()
    
    rb1_wins_count = 0
    rb2_wins_count = 0
    draws_rb_vs_rb_count = 0
    
    for i in range(num_games):
        if (i + 1) % (num_games // 20 if num_games >= 20 else 1) == 0:
            print(f"  Game {i + 1}/{num_games} (Random vs Random)")
        winner = simulate_one_game(rb1, False, rb2, False, game_instance)
        if winner == "player1":
            rb1_wins_count += 1
        elif winner == "player2":
            rb2_wins_count += 1
        else: # draw
            draws_rb_vs_rb_count += 1
            
    print(f"{Fore.CYAN}\nResults: Random Bot 1 vs. Random Bot 2 ({num_games} games){Fore.RESET}")
    print(f"  Random Bot 1 Wins: {rb1_wins_count} ({rb1_wins_count/num_games*100:.2f}%)")
    print(f"  Random Bot 2 Wins: {rb2_wins_count} ({rb2_wins_count/num_games*100:.2f}%)")
    print(f"  Draws:             {draws_rb_vs_rb_count} ({draws_rb_vs_rb_count/num_games*100:.2f}%)")

    # --- Simulation 2: AI Bot vs. Random Bot ---
    print(f"{Fore.GREEN}\nRunning: AI Bot vs. Random Bot...{Fore.RESET}")
    
    ai_bot = FireballQLearning(epsilon=0) # AI plays in pure exploitation mode
    if not ai_bot.load_model("fireball_ai_model.pkl"):
        print(f"{Fore.RED}ERROR: Could not load trained AI model 'fireball_ai_model.pkl'.{Fore.RESET}")
        print(f"{Fore.RED}Please ensure the AI has been trained and the model file exists.{Fore.RESET}")
        print(f"{Fore.YELLOW}--- Evaluation Simulations Aborted ---{Fore.RESET}")
        return
    
    random_opponent = RandomBotController()
    
    ai_wins_count = 0
    rb_wins_vs_ai_count = 0
    draws_ai_vs_rb_count = 0
    
    for i in range(num_games):
        if (i + 1) % (num_games // 20 if num_games >= 20 else 1) == 0:
            print(f"  Game {i + 1}/{num_games} (AI vs Random)")
        
        # AI is player1, Random Bot is player2
        winner = simulate_one_game(ai_bot, True, random_opponent, False, game_instance)
        if winner == "player1": # AI won
            ai_wins_count += 1
        elif winner == "player2": # Random Bot won
            rb_wins_vs_ai_count += 1
        else: # draw
            draws_ai_vs_rb_count += 1

    print(f"{Fore.CYAN}\nResults: AI Bot vs. Random Bot ({num_games} games){Fore.RESET}")
    print(f"  AI Bot Wins:         {ai_wins_count} ({ai_wins_count/num_games*100:.2f}%)")
    print(f"  Random Bot Wins:     {rb_wins_vs_ai_count} ({rb_wins_vs_ai_count/num_games*100:.2f}%)")
    print(f"  Draws:               {draws_ai_vs_rb_count} ({draws_ai_vs_rb_count/num_games*100:.2f}%)")
    
    print(f"{Fore.YELLOW}\n--- Evaluation Simulations Finished ---{Fore.RESET}")


def evaluate_model(ai_bot_instance, num_games, game_instance, random_bot_controller, verbose=True):
    """Evaluates a given AI model instance against a random bot."""
    if verbose: print(f"  Evaluating model ({num_games} games vs RandomBot)...")
    ai_wins = 0
    
    # Ensure AI is in exploitation mode for evaluation
    original_epsilon = ai_bot_instance.epsilon # Save current epsilon
    ai_bot_instance.epsilon = 0.0 # Set to exploitation mode

    for i in range(num_games):
        if verbose and (i + 1) % (num_games // 10 if num_games >=10 else 1) == 0:
            print(f"    Eval game {i+1}/{num_games}")
        # AI is player1, Random Bot is player2
        ai_bot_instance.move_history = [] # Clear AI's short-term move history before each game
        winner = simulate_one_game(ai_bot_instance, True, random_bot_controller, False, game_instance)
        if winner == "player1": # AI won
            ai_wins += 1
    
    ai_bot_instance.epsilon = original_epsilon # Restore original epsilon
    
    win_rate = ai_wins / num_games
    if verbose: print(f"  Evaluation complete. Win Rate: {win_rate*100:.2f}%")
    return win_rate
def find_best_model(num_training_attempts=5, episodes_per_attempt=75000, num_eval_games_per_attempt=200):
    print(f"{Fore.YELLOW}\n--- Starting search for the best model over {num_training_attempts} training attempts ---{Fore.RESET}")
    print(f"Each attempt trains for {episodes_per_attempt} episodes.")
    print(f"Each trained model then evaluated for {num_eval_games_per_attempt} games against a random bot.")
    
    best_ai_so_far = None
    best_win_rate_so_far = -1.0
    best_attempt_info = {}

    game_instance = FireballGame() # For evaluations
    random_opponent = RandomBotController() # For evaluations

    for attempt in range(num_training_attempts):
        print(f"{Fore.CYAN}\nTraining Attempt {attempt + 1}/{num_training_attempts}:{Fore.RESET}")
        # Train a new AI model instance (verbose=False for less console spam during automated runs)
        current_ai = train_ai(episodes=episodes_per_attempt, verbose=False) 
        
        # Evaluate the newly trained model
        current_win_rate = evaluate_model(current_ai, num_eval_games_per_attempt, game_instance, random_opponent, verbose=True)
        
        if current_win_rate > best_win_rate_so_far:
            best_win_rate_so_far = current_win_rate
            best_ai_so_far = current_ai # Store the actual AI object
            best_attempt_info = {"attempt_number": attempt + 1, "win_rate": current_win_rate}
            print(f"{Fore.GREEN}  Attempt {attempt + 1} is the new best! Win Rate: {best_win_rate_so_far*100:.2f}%{Fore.RESET}")
        else:
            print(f"  Attempt {attempt + 1} Win Rate: {current_win_rate*100:.2f}%. Not better than current best ({best_win_rate_so_far*100:.2f}%).")


    if best_ai_so_far:
        print(f"{Fore.YELLOW}\n--- Best Model Search Complete ---{Fore.RESET}")
        print(f"Best model found from Attempt {best_attempt_info['attempt_number']} with a win rate of {best_attempt_info['win_rate']*100:.2f}% (vs RandomBot).")
        print(f"{Fore.GREEN}Saving this best model to 'fireball_ai_model.pkl'...{Fore.RESET}")
        best_ai_so_far.save_model("fireball_ai_model.pkl") # Save the Q-table of the best AI
        print("Best model saved successfully.")
    else:
        print(f"{Fore.RED}\nNo models were successfully trained and evaluated to find a best one.{Fore.RESET}")
            
    return best_ai_so_far, best_win_rate_so_far

if __name__ == "__main__":
    print("Fireball Game with ML AI")
    print("1. Train AI (Standard Single Run)")
    print("2. Play vs AI")
    print("3. AI vs AI Demo")
    print("4. Evaluate Current AI Performance")
    print("5. FInd the best AI Model Training (Deep training)") # New Option
    
    choice = input("Choose option (1/2/3/4/5): ")
    
    if choice == "1":
        trained_ai = train_ai(episodes=100000) # Default episodes from previous context
        if trained_ai:
            print("Standard training complete. Saving model...")
            trained_ai.save_model("fireball_ai_model.pkl")
            print("Model saved as fireball_ai_model.pkl")
    elif choice == "2":
        play_vs_ai()
    elif choice == "3":
        ai_vs_ai_demo()
    elif choice == "4": 
        run_evaluation_simulations(num_games=100000) # Default from previous context
    elif choice == "5": # New evaluation call
        try:
            num_attempts = int(input("Enter number of training attempts (e.g., 3): "))
            # episodes_per = int(input("Enter episodes per training attempt (e.g., 50000): ")) # You can make this configurable too
            eval_games = int(input(f"Enter evaluation games per attempt (e.g., 10000): "))
            if num_attempts <=0 or eval_games <=0:
                print("Number of attempts and games must be positive.")
            else:
                # Using default episodes_per_attempt from find_best_model's signature for now
                find_best_model(num_training_attempts=num_attempts, num_eval_games_per_attempt=eval_games)
        except ValueError:
            print("Invalid input. Please enter numbers.")
    else:
        print("Invalid choice!")