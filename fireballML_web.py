import streamlit as st
import random
import pickle
from collections import defaultdict
import copy
import time
import sys
from io import StringIO
import hashlib
import uuid
import json
import ast

# Firebase Admin SDK for Firestore
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase Initialization ---
# IMPORTANT: This part requires a serviceAccountKey.json file to connect to your Firestore database.
# You need to create a Firebase project, enable Firestore, and generate a private key.
# For security, we use Streamlit's secrets management.
# 1. Create a file .streamlit/secrets.toml
# 2. Add your Firebase service account key JSON content to it like this:
#
# [firebase_service_account]
# type = "service_account"
# project_id = "your-project-id"
# private_key_id = "your-private-key-id"
# private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email = "your-client-email"
# client_id = "your-client-id"
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_x509_cert_url = "your-client-x509-cert-url"

@st.cache_resource
def init_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        # Check if the app is already initialized
        if not firebase_admin._apps:
            # Use Streamlit's secrets to get the credentials
            creds_input = st.secrets["firebase_service_account"]
            creds_dict = None

            # The secret can be a dict-like object or a string. We need a dictionary.
            if isinstance(creds_input, str):
                # If it's a string, it might be a JSON string or a string representation of a dict
                try:
                    creds_dict = json.loads(creds_input)
                except json.JSONDecodeError:
                    try:
                        # ast.literal_eval is safer for Python dict-like strings
                        creds_dict = ast.literal_eval(creds_input)
                    except (ValueError, SyntaxError):
                        st.error("Firebase credentials in secrets are a malformed string.")
                        return None
            else:
                # If it's not a string, assume it's a dict-like object (the ideal case)
                creds_dict = dict(creds_input)

            # The private_key needs to have newlines correctly formatted.
            # The .toml format might escape them, so we replace '\\n' with '\n'.
            if creds_dict and 'private_key' in creds_dict:
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')

            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.warning("Firebase connection is required for 1v1 mode. Please ensure your `secrets.toml` is configured correctly.")
        return None

# Initialize Firebase and get the database client
db = init_firebase()


# --- CSS and Styling ---
def apply_styles():
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                right: 10px;
                bottom: 10px;
                color: #B0B0B0;
                font-size: 12px;
            }
            /* Style for the Admin Options button */
            .admin-button {
                position: fixed;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }
            .admin-button button {
                font-size: 12px !important;
                padding: 0.25rem 0.5rem !important;
                height: auto !important;
                background-color: #f0f2f6 !important;
                color: #31333F !important;
                border: 1px solid #B0B0B0 !important;
            }
             .admin-button button:hover {
                border: 1px solid #31333F !important;
             }
            /* Styling for the How to Play button */
            .how-to-play-button button {
                background-color: #6df2b0 !important;
                border: 1px solid #6df2b0 !important;
                color: black !important;
            }
            /* Styling for the Back button */
            .back-button button {
                background-color: #4A90E2 !important;
                border: 1px solid #4A90E2 !important;
                color: white !important;
            }
            .stButton > button {
                color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Made by Dhruv</div>', unsafe_allow_html=True)

def get_button_css(legal_moves):
    """Generates dynamic CSS for button colors based on legal moves."""
    colors = {
        "charge": "rgba(255, 165, 0, 0.3)",
        "shield": "rgba(108, 122, 137, 0.4)",
        "fireball": "rgba(255, 0, 0, 0.3)",
        "iceball": "rgba(77, 208, 225, 0.4)",
        "megaball": "rgba(128, 0, 128, 0.3)"
    }
    css = "<style>"
    for i, move in enumerate(legal_moves):
        color = colors.get(move, "#FFFFFF")
        css += f"""
        div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button {{
            background-color: {color};
            border: 1px solid {color.replace('0.3', '1').replace('0.4', '1')};
            color: black !important;
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
    def flush(self): pass

# --- Core Game and AI Logic ---
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
            with open(filename, 'rb') as f: self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
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

# --- Training and Evaluation Functions ---
# (These functions remain largely the same, but could be enhanced with the new 1v1 data)
def train_ai(episodes=100000, self_play_ratio=0.4, st_log_container=None):
    # This function can now be enhanced to also train from data collected in Firestore
    ai = FireballQLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.5, lambda_val=0.9)
    # ... (rest of the training logic is unchanged for now)
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
            else:
                 p_moves = FireballQLearning.get_legal_moves(p_chg)
                 p_move = random.choice(p_moves) if p_moves else "charge"
            result = game.execute_turn(p_move, action)
            ai.update_histories(action, p_move)
            if use_self_play: frozen_opponent_ai.update_histories(p_move, action)
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


# --- Streamlit Page Rendering ---
@st.cache_resource
def load_ai_model():
    ai = FireballQLearning(epsilon=0)
    if ai.load_model("fireball_ai_model.pkl"): return ai
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
        st.error("Could not load `fireball_ai_model.pkl`. Please train a model from the admin panel.")
        st.stop()
    col1, col2 = st.columns(2)
    col1.metric("Your Charges", game.player_charges)
    col2.metric("AI Charges", game.comp_charges)
    st.markdown("---")
    if game.game_over:
        if game.winner == "player1": st.success("You won!")
        else: st.error("The AI won. Better luck next time!")
        if st.button("Play Again?"):
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

def page_guide():
    st.title("How to Play Fireball")
    st.markdown('<div class="back-button">', unsafe_allow_html=True)
    if st.button("← Back", use_container_width=False):
        st.session_state.page = "Play vs AI"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    **Charge**: Gets "energy" to use attack moves.
    **Shield**: Shields you from any attack except Megaball.
    **Fireball**: Requires 1 charge to use. You beat the opponent if they use Charge.
    **Iceball**: Requires 2 charges to use. It can beat Fireball and Charge.
    **Megaball**: Requires 5 charges. When you use it you instantly win, unless your opponent uses Megaball too.
    """)
    st.markdown("---")
    st.info("The computer is a ML Model that has trained from playing against itself and other bots. The goal is to eventually train it on real player data from 1v1 matches!")

# Replace the page_1v1_fireball function with this fixed version

def page_1v1_fireball():
    st.title("1v1 Fireball")
    if not db:
        st.error("Cannot connect to the game server. 1v1 mode is disabled.")
        return

    # Initialize player ID and state for 1v1 mode
    if 'player_id' not in st.session_state:
        st.session_state.player_id = str(uuid.uuid4())
    if 'match_id' not in st.session_state:
        st.session_state.match_id = None
    
    player_id = st.session_state.player_id
    
    # If not in a match, show the matchmaking UI
    if not st.session_state.match_id:
        st.subheader("Find a real-time match against another player.")
        if st.button("Find Match", type="primary", use_container_width=True):
            st.session_state.match_status = "searching"
            st.rerun()

        if st.session_state.get("match_status") == "searching":
            with st.spinner("Finding players..."):
                # Matchmaking logic
                pending_matches = db.collection("matches").where("status", "==", "pending").limit(1).get()
                
                if pending_matches: # Join an existing match
                    match = pending_matches[0]
                    st.session_state.match_id = match.id
                    db.collection("matches").document(match.id).update({
                        "players": firestore.ArrayUnion([player_id]),
                        "status": "active"
                    })
                else: # Create a new match
                    new_match_ref = db.collection("matches").document()
                    st.session_state.match_id = new_match_ref.id
                    new_match_ref.set({
                        "players": [player_id],
                        "status": "pending",
                        "game_state": {"p1_charges": 0, "p2_charges": 0, "turn": 1, "moves": {}},
                        "game_over": False,
                        "winner": None
                    })
                st.session_state.match_status = "found"
                st.rerun()
    
    # If in a match, show the game UI
    else:
        match_ref = db.collection("matches").document(st.session_state.match_id)
        
        # Add error handling for match fetch
        try:
            match_doc = match_ref.get()
            if not match_doc.exists:
                st.error("Match not found or has been disconnected.")
                st.session_state.match_id = None
                if st.button("Return to menu"): 
                    st.rerun()
                return
            match_data = match_doc.to_dict()
        except Exception as e:
            st.error(f"Error fetching match data: {e}")
            return

        if match_data.get("status") == "pending":
            st.info("Waiting for another player to join...")
            time.sleep(2)
            st.rerun()
            return
        
        # --- Active Game Logic ---
        game_state = match_data.get("game_state", {})
        players = match_data.get("players", [])
        
        try:
            player_index = players.index(player_id)
        except ValueError:
            st.error("Error: You are not part of this match.")
            st.session_state.match_id = None
            st.rerun()
            return

        is_p1 = (player_index == 0)
        my_charges = game_state.get("p1_charges", 0) if is_p1 else game_state.get("p2_charges", 0)
        opp_charges = game_state.get("p2_charges", 0) if is_p1 else game_state.get("p1_charges", 0)
        
        st.subheader(f"Match found! You are Player {player_index + 1}")
        
        col1, col2 = st.columns(2)
        col1.metric("Your Charges", my_charges)
        col2.metric("Opponent's Charges", opp_charges)
        st.markdown("---")
        
        turn_number = game_state.get("turn", 1)
        moves_this_turn = game_state.get("moves", {}).get(str(turn_number), {})
        my_move = moves_this_turn.get(player_id)

        if match_data.get("game_over"):
            winner_id = match_data.get("winner")
            if winner_id == player_id: 
                st.success("You won!")
            elif winner_id: 
                st.error("You lost!")
            else: 
                st.info("It's a draw!")
            if st.button("Find New Match"):
                st.session_state.match_id = None
                st.rerun()
        
        # Check if both players have made moves and process the turn
        elif len(moves_this_turn) == 2:
            # Both players have moved, process the turn
            p1_id, p2_id = players[0], players[1]
            p1_move = moves_this_turn[p1_id]
            p2_move = moves_this_turn[p2_id]
            
            # Check if this turn has already been processed
            turn_key = f'processed_turn_{turn_number}_{st.session_state.match_id}'
            if not st.session_state.get(turn_key, False):
                # Use a local game instance to calculate results
                temp_game = FireballGame()
                temp_game.player_charges = game_state.get("p1_charges", 0)
                temp_game.comp_charges = game_state.get("p2_charges", 0)
                result = temp_game.execute_turn(p1_move, p2_move)

                # --- ML DATA COLLECTION ---
                try:
                    db.collection("training_data").add({
                        "match_id": st.session_state.match_id,
                        "turn": turn_number,
                        "p1_id": p1_id,
                        "p2_id": p2_id,
                        "p1_charges_before": game_state.get("p1_charges", 0),
                        "p2_charges_before": game_state.get("p2_charges", 0),
                        "p1_move": p1_move,
                        "p2_move": p2_move,
                        "result": result,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                except Exception as e:
                    st.warning(f"Failed to save training data: {e}")

                # Prepare the update payload
                new_game_state = {
                    "p1_charges": temp_game.player_charges,
                    "p2_charges": temp_game.comp_charges,
                    "turn": turn_number + 1,
                    "moves": {}  # Clear moves for next turn
                }
                
                update_payload = {"game_state": new_game_state}
                
                if temp_game.game_over:
                    update_payload["game_over"] = True
                    if temp_game.winner == "player1": 
                        update_payload["winner"] = p1_id
                    elif temp_game.winner == "player2": 
                        update_payload["winner"] = p2_id
                    else:
                        update_payload["winner"] = None  # Draw

                # Update the match in Firestore
                try:
                    match_ref.update(update_payload)
                    st.session_state[turn_key] = True
                except Exception as e:
                    st.error(f"Failed to update match: {e}")
                    return
                
                # Show what happened this turn
                opp_move = p2_move if is_p1 else p1_move
                my_actual_move = p1_move if is_p1 else p2_move
                st.success(f"Turn {turn_number}: You chose **{my_actual_move}**, opponent chose **{opp_move}**")
                
                # Brief pause to show the result
                time.sleep(4)
                st.rerun()
            else:
                # Turn already processed, just show the result
                opp_move = p2_move if is_p1 else p1_move
                my_actual_move = p1_move if is_p1 else p2_move
                st.info(f"Turn {turn_number}: You chose **{my_actual_move}**, opponent chose **{opp_move}**")
                st.info("Processing turn...")
                time.sleep(1)
                st.rerun()
        
        elif my_move:
            st.info("You chose your move. Waiting for opponent...")
            # Auto-refresh to check for opponent's move
            time.sleep(2)
            st.rerun()
        else:
            # Player hasn't made a move yet
            st.subheader("Choose your move:")
            legal_moves = FireballQLearning.get_legal_moves(my_charges)
            st.markdown(get_button_css(legal_moves), unsafe_allow_html=True)
            cols = st.columns(len(legal_moves))
            for i, move in enumerate(legal_moves):
                if cols[i].button(move.capitalize(), use_container_width=True):
                    # Record the move in Firestore
                    try:
                        # Use dot notation for nested field update
                        move_path = f"game_state.moves.{turn_number}.{player_id}"
                        match_ref.update({move_path: move})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record move: {e}")
# --- Admin Panel Pages ---
def page_admin_panel():
    st.title("Admin Panel")
    st.warning("You are in Admin Mode. These tools can affect the live AI model.")
    admin_pages = {
        "Train AI": page_train_ai,
        "Collect Training Data": page_collect_training_data,
        # Add other admin pages here
    }
    choice = st.radio("Select Tool", list(admin_pages.keys()))
    admin_pages[choice]()

def page_train_ai():
    st.header("Train a New AI Model")
    st.info("This will train a new AI from scratch for 100,000 episodes and save it as `fireball_ai_model.pkl`.")
    if st.button("Start Standard Training", type="primary"):
        with st.spinner("Training in progress... This will take several minutes."):
            log_container = st.empty()
            new_ai = train_ai(episodes=100000, st_log_container=log_container)
            new_ai.save_model()
        st.success("Training complete! The new model is saved and ready to use.")
        st.info("The page will now reload to use the new model.")
        time.sleep(3)
        st.rerun()

def page_collect_training_data():
    st.header("ML Data from 1v1 Matches")
    st.info("This section shows the raw data collected from human-vs-human games. In the future, this data can be used to train a more advanced model.")
    if not db:
        st.error("Firestore not connected.")
        return
    
    data_limit = st.number_input("Number of recent turns to display", 10, 500, 50)
    
    with st.spinner("Fetching training data..."):
        docs = db.collection('training_data').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(data_limit).stream()
        data = [doc.to_dict() for doc in docs]

    if not data:
        st.write("No training data found yet. Play some 1v1 games to collect data!")
    else:
        st.dataframe(data)

# --- Main App ---
def main():
    st.set_page_config(page_title="Fireball AI", layout="centered")
    apply_styles()

    # Initialize session state variables
    if 'admin_mode' not in st.session_state:
        st.session_state.admin_mode = False
    if 'page' not in st.session_state:
        st.session_state.page = "Play vs AI"

    # Admin Mode Button - Updated Logic
    st.markdown('<div class="admin-button">', unsafe_allow_html=True)
    button_label = "Exit Admin Mode" if st.session_state.admin_mode else "Admin Options"
    if st.button(button_label, key="admin_toggle"):
        if st.session_state.admin_mode:
            st.session_state.admin_mode = False
            st.toast("Exited Admin Mode.")
            st.rerun()
        else:
            # Show password prompt only if not in admin mode
            st.session_state.show_password_prompt = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle password prompt
    if st.session_state.get('show_password_prompt'):
        with st.form("password_form"):
            st.subheader("Enter Admin Password")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter")
            if submitted:
                # IMPORTANT: Use a more secure password management system in a real application
                if hashlib.sha256(password.encode()).hexdigest() == "ea4de091b760a4e538140c342585130649e646c54d4939ae7f142bb81d5506fa":
                    st.session_state.admin_mode = True
                    st.session_state.show_password_prompt = False
                    st.success("Admin Mode Activated")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Incorrect password.")

    st.sidebar.title("Game Menu")
    
    st.sidebar.markdown('<div class="how-to-play-button">', unsafe_allow_html=True)
    if st.sidebar.button("How to Play", use_container_width=True):
        st.session_state.page = "Guide"
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Sidebar navigation
    page_options = ["Play vs AI", "1v1 Fireball"]
    if st.session_state.admin_mode:
        page_options.append("Admin Panel")

    # If current page is invalid (e.g., admin turned off), default to home
    if st.session_state.page not in page_options and st.session_state.page != "Guide":
        st.session_state.page = "Play vs AI"

    # Display radio buttons for navigation
    if st.session_state.page != "Guide":
        current_index = page_options.index(st.session_state.page)
        selected_page = st.sidebar.radio("Choose an option", page_options, index=current_index)
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()

    # Page routing
    page_map = {
        "Play vs AI": page_play_vs_ai,
        "1v1 Fireball": page_1v1_fireball,
        "Guide": page_guide,
        "Admin Panel": page_admin_panel,
    }
    page_map[st.session_state.page]()

if __name__ == "__main__":
    main()
