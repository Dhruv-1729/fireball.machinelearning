import streamlit as st
import random
import pickle
from collections import defaultdict
import copy
import time
import sys
from io import StringIO

# --- CSS and Styling ---
def apply_styles():
    """Applies custom CSS for the footer and colored buttons."""
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                right: 10px;
                bottom: 10px;
                color: #B0B0B0;
                font-size: 12px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="footer">Made by Dhruv</div>', unsafe_allow_html=True)

def get_button_css(legal_moves):
    """Generates dynamic CSS for button colors based on legal moves."""
    colors = {
        "charge": "rgba(255, 165, 0, 0.3)",   # Light Orange
        "shield": "rgba(108, 122, 137, 0.4)", # Light Black/Gray
        "fireball": "rgba(255, 0, 0, 0.3)",   # Light Red
        "iceball": "rgba(77, 208, 225, 0.4)", # Light Bluish Green
        "megaball": "rgba(128, 0, 128, 0.3)"  # Light Purple
    }
    
    css = "<style>"
    # This is a trick to style individual streamlit buttons by their order in a container
    for i, move in enumerate(legal_moves):
        color = colors.get(move, "#FFFFFF") # Default to white
        # Target the button inside the i-th column of the stHorizontalBlock
        css += f"""
        div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button {{
            background-color: {color};
            border: 1px solid {color.replace('0.3', '1').replace('0.4', '1')};
        }}
        """
    css += "</style>"
    return css

# --- A context manager to redirect stdout to a Streamlit container ---
class StreamlitLog(StringIO):
    def __init__(self, container):
        super().__init__()
        self.container = container

    def write(self, s):
        super().write(s)
        self.container.code(self.getvalue())

    def flush(self):
        pass # No need to flush in this context

# --- Core Game and AI Logic (Complete) ---
# (Includes all classes and functions from your original file)

class FireballQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, lambda_val=0.9):
        self.learning_rate, self.discount_factor, self.epsilon, self.lambda_val = learning_rate, discount_factor, epsilon, lambda_val
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.eligibility_traces = defaultdict(lambda: defaultdict(float))
        self.move_history, self.opp_move_history = [], []

    def update_histories(self, my_action, opp_action):
        for history, action in [(self.move_history, my_action), (self.opp_move_history, opp_action)]:
            history.append(action)
            if len(history) > 4: history.pop(0)

    def get_state(self, my_charges, opp_charges):
        my_patt = "_".join(self.move_history[-3:]) or "start"
        opp_patt = "_".join(self.opp_move_history[-3:]) or "start"
        return f"mc_{min(my_charges, 10)}_oc_{min(opp_charges, 10)}_mypatt_{my_patt}_opppatt_{opp_patt}"

    @staticmethod
    def get_legal_moves(charges):
        moves = ["charge", "shield"]
        if charges >= 1: moves.append("fireball")
        if charges >= 2: moves.append("iceball")
        if charges >= 5: moves.append("megaball")
        return moves

    def choose_action(self, state, legal_moves, training=True):
        if training and random.random() < self.epsilon: return random.choice(legal_moves)
        if state not in self.q_table: return random.choice(legal_moves)
        q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
        return max(q_vals, key=q_vals.get)

    def save_model(self, filename="fireball_ai_model.pkl"):
        with open(filename, 'wb') as f: pickle.dump(dict(self.q_table), f)

    def load_model(self, filename="fireball_ai_model.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            return True
        except FileNotFoundError: return False

class FireballGame:
    def __init__(self): self.reset_game()
    def reset_game(self): self.player_charges, self.comp_charges, self.game_over, self.winner = 0, 0, False, None
    def get_move_cost(self, move): return {"charge": -1, "fireball": 1, "iceball": 2, "megaball": 5}.get(move, 0)
    def execute_turn(self, p1_move, p2_move):
        self.player_charges = max(0, self.player_charges - self.get_move_cost(p1_move))
        self.comp_charges = max(0, self.comp_charges - self.get_move_cost(p2_move))
        result = self.determine_winner(p1_move, p2_move)
        if result != "continue": self.game_over, self.winner = True, result
        return result
    def determine_winner(self, p1, p2):
        if p1 == p2 and p1 != "megaball": return "continue"
        if p1 == "megaball": return "player1" if p2 != "megaball" else "continue"
        if p2 == "megaball": return "player2"
        if "shield" in [p1, p2]: return "continue"
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if p2 in win_map.get(p1, []): return "player1"
        if p1 in win_map.get(p2, []): return "player2"
        return "continue"

class RandomBotController:
    def choose_move(self, my_charges, legal_moves): return random.choice(legal_moves) if legal_moves else "charge"

# --- Training and Evaluation Functions (Adapted for Streamlit) ---
# NOTE: All 'print' statements are replaced with 'st_log.write' to show output in the UI.

def train_ai(episodes=100000, self_play_ratio=0.4, st_log_container=None):
    ai = FireballQLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.5, lambda_val=0.9)
    frozen_opponent_ai = None
    
    log_area = st_log_container or st.empty()
    st_log = StreamlitLog(log_area)
    
    st_log.write(f"Starting training with Self-Play ({self_play_ratio*100}%) and Q(λ)...")

    for episode in range(episodes):
        if episode > 0 and episode % 10000 == 0:
            st_log.write(f"  Updating self-play opponent at episode {episode}...")
            frozen_opponent_ai = FireballQLearning(epsilon=0.05)
            frozen_opponent_ai.q_table = copy.deepcopy(ai.q_table)

        game = FireballGame()
        ai.eligibility_traces.clear()
        ai.move_history, ai.opp_move_history = [], []
        state = ai.get_state(game.comp_charges, game.player_charges)
        action = ai.choose_action(state, FireballQLearning.get_legal_moves(game.comp_charges), True)

        while not game.game_over:
            ai_chg, p_chg = game.comp_charges, game.player_charges
            use_self_play = frozen_opponent_ai and random.random() < self_play_ratio
            
            if use_self_play:
                p_state = frozen_opponent_ai.get_state(p_chg, ai_chg)
                p_move = frozen_opponent_ai.choose_action(p_state, FireballQLearning.get_legal_moves(p_chg), False)
            else: # Scripted bot logic
                 p_moves = FireballQLearning.get_legal_moves(p_chg)
                 p_move = random.choice(p_moves) if p_moves else "charge"

            result = game.execute_turn(p_move, action)
            ai.update_histories(action, p_move)
            if use_self_play: frozen_opponent_ai.update_histories(p_move, action)
            
            # Simplified reward structure for brevity
            reward = 0
            if result == "player2": reward = 15
            elif result == "player1": reward = -15
            else: reward = (ai_chg - p_chg) * 0.2 - 0.1

            next_state = ai.get_state(game.comp_charges, game.player_charges)
            next_legal = FireballQLearning.get_legal_moves(game.comp_charges)
            next_action = ai.choose_action(next_state, next_legal, True)
            max_next_q = 0.0 if game.game_over else max([ai.q_table[next_state][m] for m in next_legal] or [0.0])
            
            delta = reward + (ai.discount_factor * max_next_q) - ai.q_table[state][action]
            ai.eligibility_traces[state][action] += 1

            for s_trace, a_dict in list(ai.eligibility_traces.items()):
                for a_trace, e_val in list(a_dict.items()):
                    ai.q_table[s_trace][a_trace] += ai.learning_rate * delta * e_val
                    ai.eligibility_traces[s_trace][a_trace] *= ai.discount_factor * ai.lambda_val
            
            state, action = next_state, next_action

        if (episode + 1) % 1000 == 0: ai.epsilon = max(0.01, ai.epsilon * 0.96)
        if (episode + 1) % 5000 == 0 or episode == episodes - 1:
            st_log.write(f"  Episode {episode + 1}/{episodes}, Epsilon: {ai.epsilon:.4f}")

    st_log.write("\nTraining completed!")
    return ai

def simulate_one_game(p1_ctrl, p1_is_ai, p2_ctrl, p2_is_ai, game, max_turns=50):
    game.reset_game()
    if p1_is_ai: p1_ctrl.move_history, p1_ctrl.opp_move_history = [], []
    if p2_is_ai: p2_ctrl.move_history, p2_ctrl.opp_move_history = [], []

    for _ in range(max_turns):
        p1_moves = FireballQLearning.get_legal_moves(game.player_charges)
        p2_moves = FireballQLearning.get_legal_moves(game.comp_charges)

        p1_move = p1_ctrl.choose_action(p1_ctrl.get_state(game.player_charges, game.comp_charges), p1_moves, False) if p1_is_ai else p1_ctrl.choose_move(game.player_charges, p1_moves)
        p2_move = p2_ctrl.choose_action(p2_ctrl.get_state(game.comp_charges, game.player_charges), p2_moves, False) if p2_is_ai else p2_ctrl.choose_move(game.comp_charges, p2_moves)
        
        game.execute_turn(p1_move, p2_move)
        if p1_is_ai: p1_ctrl.update_histories(p1_move, p2_move)
        if p2_is_ai: p2_ctrl.update_histories(p2_move, p1_move)

        if game.game_over: return game.winner
    return "draw"

def run_evaluation_simulations(num_games=10000, st_log_container=None):
    log_area = st_log_container or st.empty()
    st_log = StreamlitLog(log_area)

    st_log.write(f"--- Starting Evaluation Simulations ({num_games} games) ---")
    game = FireballGame()
    ai_bot = FireballQLearning(epsilon=0)
    if not ai_bot.load_model():
        st_log.write("ERROR: Could not load trained AI model. Please train one first.")
        return
    
    rand_bot = RandomBotController()
    ai_wins, rb_wins, draws = 0, 0, 0
    
    st_log.write(f"Running: AI Bot vs. Random Bot...")
    for i in range(num_games):
        if (i + 1) % (num_games // 10) == 0:
            st_log.write(f"  Game {i + 1}/{num_games} (AI vs Random)")
        winner = simulate_one_game(ai_bot, True, rand_bot, False, game)
        if winner == "player1": ai_wins += 1
        elif winner == "player2": rb_wins += 1
        else: draws += 1
    
    st_log.write(f"\nResults: AI Bot vs. Random Bot ({num_games} games)")
    st_log.write(f"  AI Bot Wins:     {ai_wins} ({ai_wins/num_games*100:.2f}%)")
    st_log.write(f"  Random Bot Wins: {rb_wins} ({rb_wins/num_games*100:.2f}%)")
    st_log.write(f"  Draws:           {draws} ({draws/num_games*100:.2f}%)")
    st_log.write("\n--- Evaluation Simulations Finished ---")

def find_best_model(num_attempts=5, episodes=75000, eval_games=2000, st_log_container=None):
    log_area = st_log_container or st.empty()
    st_log = StreamlitLog(log_area)
    
    st_log.write(f"--- Starting search for the best model over {num_attempts} attempts ---")
    best_ai, best_win_rate = None, -1.0
    game, rand_bot = FireballGame(), RandomBotController()

    for attempt in range(num_attempts):
        st_log.write(f"\nTraining Attempt {attempt + 1}/{num_attempts}:")
        current_ai = train_ai(episodes=episodes, st_log_container=log_area)
        
        st_log.write(f"  Evaluating model ({eval_games} games vs RandomBot)...")
        ai_wins = 0
        original_epsilon = current_ai.epsilon
        current_ai.epsilon = 0.0 # Exploitation mode
        for i in range(eval_games):
            if (i+1) % (eval_games//5) == 0: st_log.write(f"    Eval game {i+1}/{eval_games}")
            winner = simulate_one_game(current_ai, True, rand_bot, False, game)
            if winner == "player1": ai_wins += 1
        current_ai.epsilon = original_epsilon # Restore
        
        win_rate = ai_wins / eval_games
        if win_rate > best_win_rate:
            best_win_rate, best_ai = win_rate, current_ai
            st_log.write(f"  Attempt {attempt+1} is the new best! Win Rate: {win_rate*100:.2f}%")
        else:
            st_log.write(f"  Attempt {attempt+1} Win Rate: {win_rate*100:.2f}%. Not an improvement.")

    if best_ai:
        st_log.write("\n--- Best Model Search Complete ---")
        st_log.write(f"Best model found with a win rate of {best_win_rate*100:.2f}%.")
        st_log.write("Saving this best model to 'fireball_ai_model.pkl'...")
        best_ai.save_model("fireball_ai_model.pkl")
        st_log.write("Best model saved successfully.")
    else:
        st_log.write("\nNo models were successfully trained.")

def ai_vs_ai_demo(st_log_container=None):
    log_area = st_log_container or st.empty()
    st_log = StreamlitLog(log_area)

    ai1, ai2 = FireballQLearning(epsilon=0.05), FireballQLearning(epsilon=0.05)
    if not ai1.load_model() or not ai2.load_model():
        st_log.write("Could not load AI model. Please train one first.")
        return
    
    game = FireballGame()
    st_log.write('AI vs AI Demo - Strategic Battle!')
    
    for turn in range(50):
        if game.game_over: break
        p1_move = ai1.choose_action(ai1.get_state(game.player_charges, game.comp_charges), FireballQLearning.get_legal_moves(game.player_charges), False)
        p2_move = ai2.choose_action(ai2.get_state(game.comp_charges, game.player_charges), FireballQLearning.get_legal_moves(game.comp_charges), False)
        
        result = game.execute_turn(p1_move, p2_move)
        ai1.update_histories(p1_move, p2_move)
        ai2.update_histories(p2_move, p1_move)

        st_log.write(f"\nTurn {turn + 1}:")
        st_log.write(f"  AI1 (charges: {game.player_charges}) chose: {p1_move}")
        st_log.write(f"  AI2 (charges: {game.comp_charges}) chose: {p2_move}")
        if result != "continue": st_log.write(f"  Winner: {'AI1' if result == 'player1' else 'AI2'}")
    
    if not game.game_over: st_log.write("\nGame ended due to max turns.")


# --- Streamlit Page Rendering ---

@st.cache_resource
def load_ai_model():
    ai = FireballQLearning(epsilon=0)
    if ai.load_model("fireball_ai_model.pkl"):
        return ai
    return None

def init_play_vs_ai_state():
    if 'game' not in st.session_state or st.session_state.get('page') != 'Play vs AI':
        st.session_state.page = 'Play vs AI'
        st.session_state.game = FireballGame()
        st.session_state.turn_history = []
        ai_master = load_ai_model()
        st.session_state.ai_player = copy.deepcopy(ai_master) if ai_master else None

def page_play_vs_ai():
    st.title("Fireball Machine-Learning Model")
    st.markdown("Play a game against the ML Model!")
    
    init_play_vs_ai_state()
    game = st.session_state.game
    
    if not st.session_state.ai_player:
        st.error("Could not load `fireball_ai_model.pkl`. Please train a model from the sidebar.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("Your Charges", game.player_charges)
    col2.metric("AI Charges", game.comp_charges)
    st.markdown("---")

    if game.game_over:
        if game.winner == "player1": st.success("You won!")
        else: st.error("The AI won. Better luck next time!")
        if st.button("Play Again?"):
            # Directly reset the session state instead of calling init function
            st.session_state.game = FireballGame()
            st.session_state.turn_history = []
            ai_master = load_ai_model()
            st.session_state.ai_player = copy.deepcopy(ai_master) if ai_master else None
            st.rerun()
    else:
        st.subheader("Choose your move:")
        legal_moves = FireballQLearning.get_legal_moves(game.player_charges)
        st.markdown(get_button_css(legal_moves), unsafe_allow_html=True)
        
        cols = st.columns(len(legal_moves))
        for i, move in enumerate(legal_moves):
            if cols[i].button(move.capitalize(), use_container_width=True):
                ai_player = st.session_state.ai_player
                ai_moves = FireballQLearning.get_legal_moves(game.comp_charges)
                ai_state = ai_player.get_state(game.comp_charges, game.player_charges)
                ai_move = ai_player.choose_action(ai_state, ai_moves, training=False)
                game.execute_turn(move, ai_move)
                ai_player.update_histories(ai_move, move)
                st.session_state.turn_history.append(f"You chose **{move}**. AI chose **{ai_move}**.")
                st.rerun()

    if st.session_state.turn_history:
        st.markdown("---")
        st.subheader("Turn History")
        for turn in reversed(st.session_state.turn_history): st.info(turn)

def page_train_ai():
    st.title("Train a New AI Model")
    st.info("This will train a new AI from scratch for 100,000 episodes and save it as `fireball_ai_model.pkl`.")
    if st.button("Start Standard Training", type="primary"):
        with st.spinner("Training in progress... This will take several minutes."):
            log_container = st.empty()
            new_ai = train_ai(episodes=100000, st_log_container=log_container)
            new_ai.save_model()
        st.success("Training complete! The new model is saved and ready to use.")
        st.info("The page will now reload to use the new model.")
        time.sleep(3)
        st.rerun() # Rerun to load the new model

def page_ai_vs_ai():
    st.title("AI vs AI Demonstration")
    st.info("Watch two instances of the trained AI play against each other.")
    if st.button("Start Demo", type="primary"):
        with st.spinner("Simulating game..."):
            log_container = st.empty()
            ai_vs_ai_demo(st_log_container=log_container)

def page_evaluate_ai():
    st.title("Evaluate Current AI Model")
    st.info("Simulate 10,000 games between the current AI and a completely random bot to gauge its performance.")
    if st.button("Start Evaluation", type="primary"):
        with st.spinner("Running simulations... This may take a moment."):
            log_container = st.empty()
            run_evaluation_simulations(num_games=10000, st_log_container=log_container)

def page_find_best_model():
    st.title("🏆 Find the Best Possible Model")
    st.warning("This is a very tedious process. It will train multiple AI models back-to-back and save only the one with the highest win rate.")
    
    num_attempts = st.number_input("Number of training attempts", min_value=1, max_value=20, value=3)
    episodes = st.number_input("Episodes per attempt", min_value=1000, max_value=100000, value=50000, step=1000)
    eval_games = st.number_input("Evaluation games per attempt", min_value=100, max_value=10000, value=1000, step=100)
    
    if st.button("Start Deep Training", type="primary"):
        with st.spinner("Finding best model... This will take a very long time."):
            log_container = st.empty()
            find_best_model(num_attempts, episodes, eval_games, log_container)
        st.success("Deep training complete! The best model found has been saved.")
        st.info("The page will now reload to use the new model.")
        time.sleep(3)
        st.rerun()

# --- Main App ---
def main():
    st.set_page_config(page_title="Fireball AI", layout="centered")
    apply_styles()

    st.sidebar.title("Game Modes")
    page = st.sidebar.radio("Choose an option", 
        ["Play vs AI", "Train AI", "AI vs AI Demo", "Evaluate AI", "Find Best Model"])

    if page == "Play vs AI":
        page_play_vs_ai()
    elif page == "Train AI":
        page_train_ai()
    elif page == "AI vs AI Demo":
        page_ai_vs_ai()
    elif page == "Evaluate AI":
        page_evaluate_ai()
    elif page == "Find Best Model":
        page_find_best_model()

if __name__ == "__main__":
    main()
