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
        st.warning("Firebase connection is required for 1v1 mode and statistics. Please ensure your `secrets.toml` is configured correctly.")
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
class AdaptiveFireballAI(FireballQLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.human_patterns = defaultdict(int)
        self.adaptation_weight = 0.3
        
    def analyze_human_pattern(self, opp_history):
        """Analyze opponent's move patterns"""
        if len(opp_history) >= 3:
            pattern = "_".join(opp_history[-3:])
            self.human_patterns[pattern] += 1
    
    def predict_human_move(self, opp_history, legal_moves):
        """Predict human's next move based on patterns"""
        if len(opp_history) >= 2:
            recent_pattern = "_".join(opp_history[-2:])
            # Look for patterns that start with this sequence
            matching_patterns = {p: count for p, count in self.human_patterns.items() 
                               if p.startswith(recent_pattern) and count > 2}
            
            if matching_patterns:
                # Get the most common next move
                next_moves = [p.split('_')[-1] for p in matching_patterns.keys()]
                predicted = max(set(next_moves), key=next_moves.count)
                if predicted in legal_moves:
                    return predicted
        return None
    
    def choose_action(self, state, legal_moves, training=True, opp_history=None):
        """Enhanced action selection with human pattern consideration"""
        base_action = super().choose_action(state, legal_moves, training)
        
        if opp_history and len(opp_history) >= 2:
            predicted_human_move = self.predict_human_move(opp_history, 
                                   FireballQLearning.get_legal_moves(10))  # Assume max charges for prediction
            
            if predicted_human_move:
                # Choose counter-move
                counter_moves = {
                    "charge": ["fireball", "iceball"],
                    "fireball": ["iceball", "shield"],
                    "iceball": ["shield", "megaball"],
                    "shield": ["charge", "fireball"],
                    "megaball": ["megaball"]
                }
                
                counters = [m for m in counter_moves.get(predicted_human_move, []) if m in legal_moves]
                if counters and random.random() < self.adaptation_weight:
                    return random.choice(counters)
        
        return base_action
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
        
        # Load adaptive AI
        base_ai = load_ai_model()
        if base_ai:
            adaptive_ai = AdaptiveFireballAI(epsilon=0)
            adaptive_ai.q_table = base_ai.q_table
            # Auto-learn from recent human data
            auto_learn_from_human_data(adaptive_ai)
            st.session_state.ai_player = adaptive_ai
        else:
            st.session_state.ai_player = None

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
        winner = ""
        if game.winner == "player1":
            st.success("You won!")
            winner = "human"
        else:
            st.error("The AI won. Better luck next time!")
            winner = "ai"

        # Log game result to Firestore for statistics
        if db and 'logged_game' not in st.session_state:
            try:
                db.collection("ai_vs_human_matches").add({
                    "winner": winner,
                    "turns": len(st.session_state.turn_history),
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
                st.session_state.logged_game = True # Ensure we only log once
            except Exception as e:
                st.warning(f"Could not log game statistics: {e}")

        if st.button("Play Again?"):
            st.session_state.game = FireballGame()
            st.session_state.turn_history = []
            st.session_state.pop('logged_game', None) # Reset log flag
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
def load_turn_history_from_firestore(match_id):
    """Load existing turn history from Firestore training data"""
    if not db:
        return []
    
    try:
        docs = db.collection('training_data').where('match_id', '==', match_id).order_by('turn').stream()
        history = []
        for doc in docs:
            data = doc.to_dict()
            turn_text = f"Turn {data['turn']}: Player 1 chose **{data['p1_move']}**, Player 2 chose **{data['p2_move']}**"
            history.append(turn_text)
        return history
    except Exception as e:
        print(f"Error loading turn history: {e}")
        return []
def page_guide():
    st.title("How to Play Fireball")
    st.markdown('<div class="back-button">', unsafe_allow_html=True)
    if st.button("← Back", use_container_width=False):
        st.session_state.page = "Play vs AI"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Game Objective
    Use strategy to beat your opponent using different moves.
    
    ## Available Moves
    
    **Charge** Gains 1 "charge". This is the way to build up power for attacks. Always available and costs nothing.
    
    **Shield** Protects you from Fireball and Iceball attacks. Cannot block Megaball. Always available and costs nothing.
    
    **Fireball** A basic attack that defeats opponents that use Charge. Blocked by Shield. You need at least 1 charge to use this.
    
    **Iceball** A stronger attack that defeats both Charge and Fireball. Blocked by Shield. You need at least 2 charges to use this.
    
    **Megaball** The best attack. Instantly wins the game unless your opponent also uses Megaball (which cancel each other out). Cannot be blocked by Shield. You need at least 5 charges to use this.
    
    ## Basic Strategy 
    
    - **Build up charges** early to unlock powerful attacks
    - **Use Shield wisely** to predict when your opponent plays an attack
    - **Watch your opponent's charges** - if they have 5+, they basically win.
    - **Mix up your patterns** - don't be predictable!
    
    ## AI
    The computer opponent is a machine learning model that has been trained through:
    - **Training**: Playing thousands of games against itself and random bots
    - **Pattern recognition**: Learning to predict and counter human strategies  
    - **Adaptive learning**: Continuously improving from real player data
    """)

# Replace the page_1v1_fireball function with this fixed version
def page_1v1_fireball():
    st.title("1v1 Fireball")
    if not db:
        st.error("Cannot connect to the game server. 1v1 mode is disabled.")
        return

    # Initialize player ID and username
    if 'player_id' not in st.session_state:
        st.session_state.player_id = str(uuid.uuid4())
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'match_id' not in st.session_state:
        st.session_state.match_id = None
    
    # Username selection
    if not st.session_state.username:
        st.subheader("Choose Your Username")
        with st.form("username_form"):
            username = st.text_input(" ", max_chars=15)
            if st.form_submit_button("Continue"):
                if 1 <= len(username) <= 15:
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Username must be 1-15 characters long.")
        return
    
    player_id = st.session_state.player_id
    
    # Get current player count
    try:
        pending_matches = db.collection("matches").where("status", "==", "pending").get()
        searching_players = len(pending_matches)
        if st.session_state.get("match_status") == "searching":
            searching_players += 1
    except:
        searching_players = 0
    
    # If not in a match, show the matchmaking UI
    if not st.session_state.match_id:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Welcome, {st.session_state.username}!")
            st.write("Play a live 1v1 game against another player.")
            if st.button("Find Match", type="primary", use_container_width=True):
                st.session_state.match_status = "searching"
                st.session_state.turn_history_1v1 = []
                st.rerun()
        
        with col2:
            st.info(f"🔍 {searching_players} players searching for match currently...")

        if st.session_state.get("match_status") == "searching":
            with st.spinner("Finding players..."):
                # Matchmaking logic
                pending_matches = db.collection("matches").where("status", "==", "pending").limit(1).get()
                
                if pending_matches: # Join an existing match
                    match = pending_matches[0]
                    st.session_state.match_id = match.id
                    db.collection("matches").document(match.id).update({
                        "players": firestore.ArrayUnion([player_id]),
                        f"player_usernames.{player_id}": st.session_state.username,
                        "status": "active"
                    })
                else: # Create a new match
                    new_match_ref = db.collection("matches").document()
                    st.session_state.match_id = new_match_ref.id
                    new_match_ref.set({
                        "players": [player_id],
                        "player_usernames": {player_id: st.session_state.username},  
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
        
        player_usernames = match_data.get("player_usernames", {})
        my_username = player_usernames.get(player_id, f"Player {player_index + 1}")
        opp_id = players[1] if is_p1 else players[0]
        opp_username = player_usernames.get(opp_id, f"Player {2 if is_p1 else 1}")

        st.subheader(f"Match found! You are {my_username}")
        
        col1, col2 = st.columns(2)
        col1.metric(f"{my_username}'s Charges", my_charges)
        col2.metric(f"{opp_username}'s Charges", opp_charges)
        st.markdown("---")
        
        turn_number = game_state.get("turn", 1)
        moves_this_turn = game_state.get("moves", {}).get(str(turn_number), {})
        my_move = moves_this_turn.get(player_id)

        if 'turn_history_1v1' not in st.session_state or st.session_state.match_id not in st.session_state.get('current_match_id_for_history', ''):
            st.session_state.turn_history_1v1 = load_turn_history_from_firestore(st.session_state.match_id)
            st.session_state.current_match_id_for_history = st.session_state.match_id

        # Game over check
        if match_data.get("game_over"):
            winner_id = match_data.get("winner")
            if winner_id == player_id: 
                st.success(f"{my_username} won!")
            elif winner_id: 
                winner_username = player_usernames.get(winner_id, "Unknown Player")
                st.error(f"{winner_username} won!")
            else:
                st.info("It's a draw!")
            
            if st.button("Find New Match"):
                st.session_state.match_id = None
                st.session_state.turn_history_1v1 = []
                st.rerun()
        
        # Check if both players have made moves and process the turn
        elif len(moves_this_turn) == 2:
            p1_id, p2_id = players[0], players[1]
            p1_move = moves_this_turn[p1_id]
            p2_move = moves_this_turn[p2_id]
            
            if not game_state.get('turn_processed_for_turn', {}).get(str(turn_number)):
                try:
                    @firestore.transactional
                    def process_turn_transaction(transaction, match_ref):
                        match_snapshot = match_ref.get(transaction=transaction)
                        current_data = match_snapshot.to_dict()
                        current_game_state = current_data.get("game_state", {})
                        
                        if current_game_state.get('turn_processed_for_turn', {}).get(str(turn_number)):
                            return None
                        
                        temp_game = FireballGame()
                        temp_game.player_charges = current_game_state.get("p1_charges", 0)
                        temp_game.comp_charges = current_game_state.get("p2_charges", 0)
                        result = temp_game.execute_turn(p1_move, p2_move)
                        
                        new_game_state = {
                            "p1_charges": temp_game.player_charges,
                            "p2_charges": temp_game.comp_charges,
                            "turn": turn_number + 1,
                            "moves": {},
                            "turn_processed_for_turn": {str(turn_number): True}
                        }
                        
                        update_payload = {"game_state": new_game_state}
                        
                        if temp_game.game_over:
                            update_payload["game_over"] = True
                            update_payload["end_timestamp"] = firestore.SERVER_TIMESTAMP
                            if temp_game.winner == "player1": 
                                update_payload["winner"] = p1_id
                            elif temp_game.winner == "player2": 
                                update_payload["winner"] = p2_id
                            else:
                                update_payload["winner"] = None
                        
                        transaction.update(match_ref, update_payload)
                        return result
                    
                    transaction = db.transaction()
                    result = process_turn_transaction(transaction, match_ref)
                    
                    if result is not None:
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
                        
                        p1_username = player_usernames.get(p1_id, "Player 1")
                        p2_username = player_usernames.get(p2_id, "Player 2")
                        turn_result_text = f"Turn {turn_number}: {p1_username} chose **{p1_move}**, {p2_username} chose **{p2_move}**"
                        st.session_state.turn_history_1v1.append(turn_result_text)
                        st.success(turn_result_text)
                    
                except Exception as e:
                    st.error(f"Failed to process turn: {e}")
                    return
                
                time.sleep(1.5)
                st.rerun()
            else:
                st.info("Processing turn...")
                time.sleep(1)
                st.rerun()

        elif my_move:
            st.info(f"You chose **{my_move}**. Waiting for {opp_username}...")
            time.sleep(2)
            st.rerun()
        
        else:
            st.subheader("Choose your move:")
            legal_moves = FireballQLearning.get_legal_moves(my_charges)
            st.markdown(get_button_css(legal_moves), unsafe_allow_html=True)
            cols = st.columns(len(legal_moves))
            for i, move in enumerate(legal_moves):
                if cols[i].button(move.capitalize(), use_container_width=True):
                    try:
                        move_path = f"game_state.moves.{turn_number}.{player_id}"
                        match_ref.update({move_path: move})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record move: {e}")

        if st.session_state.get('turn_history_1v1'):
            st.markdown("---")
            st.subheader("Turn History")
            for turn in reversed(st.session_state.turn_history_1v1):
                st.info(turn)
# --- Admin Panel Pages ---
def page_admin_panel():
    st.title("Admin Panel")
    st.warning("You are in Admin Mode. These tools can affect the live AI model.")
    admin_pages = {
        "Train AI": page_train_ai,
        "Collect Training Data": page_collect_training_data,
        "Statistics": page_statistics,
    }
    choice = st.radio("Select Tool", list(admin_pages.keys()))
    admin_pages[choice]()
def auto_learn_from_human_data(ai_instance, limit=1000):
    """Automatically learn from recent human vs human games"""
    if not db:
        return
    
    try:
        docs = db.collection('training_data').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
        
        for doc in docs:
            data = doc.to_dict()
            p1_state = ai_instance.get_state(data['p1_charges_before'], data['p2_charges_before'])
            p2_state = ai_instance.get_state(data['p2_charges_before'], data['p1_charges_before'])
            
            if data['result'] == 'player1':
                reward_p1, reward_p2 = 10, -10
            elif data['result'] == 'player2':
                reward_p1, reward_p2 = -10, 10
            else:
                reward_p1, reward_p2 = 0, 0
            
            ai_instance.q_table[p1_state][data['p1_move']] += 0.1 * reward_p1
            ai_instance.q_table[p2_state][data['p2_move']] += 0.1 * reward_p2
            
    except Exception as e:
        print(f"Auto-learning error: {e}")
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

def page_statistics():
    st.title("Live Game Statistics")
    if not db:
        st.error("Firestore not connected.")
        return
    # ADD THIS CODE BLOCK inside the page_statistics() function

    st.markdown("---")
    st.header("Website Traffic")
    with st.spinner("Counting unique visitors..."):
        try:
            # Get all documents in the collection. The number of documents is the count.
            visitors_ref = db.collection("unique_visitors").stream()
            unique_visitor_count = len(list(visitors_ref))
            st.metric("Unique Visitors (Sessions)", unique_visitor_count)
        except Exception as e:
            st.error(f"Could not load visitor statistics: {e}")

    # --- AI vs. Human Statistics ---
    st.header("AI vs. Human Performance")
    with st.spinner("Calculating AI performance..."):
        try:
            matches_ref = db.collection("ai_vs_human_matches").stream()
            matches = [doc.to_dict() for doc in matches_ref]
            
            if not matches:
                st.info("No AI vs. Human games have been played yet.")
            else:
                total_games = len(matches)
                ai_wins = sum(1 for m in matches if m['winner'] == 'ai')
                total_turns = sum(m.get('turns', 0) for m in matches)
                
                win_rate = (ai_wins / total_games) * 100 if total_games > 0 else 0
                avg_length = total_turns / total_games if total_games > 0 else 0
                
                col1, col2 = st.columns(2)
                col1.metric("AI Win Rate vs. Humans", f"{win_rate:.2f}%", f"{ai_wins} wins in {total_games} games")
                col2.metric("Average AI Match Length", f"{avg_length:.2f} rounds")

        except Exception as e:
            st.error(f"Could not load AI statistics: {e}")

    st.markdown("---")

    # --- 1v1 Match History ---
    st.header("Recent 1v1 Match Summary")
    with st.spinner("Fetching recent 1v1 matches..."):
        try:
            # Query the last 45 completed matches
            matches_ref = db.collection("matches") \
                            .where("status", "==", "active") \
                            .where("game_over", "==", True) \
                            .order_by("end_timestamp", direction=firestore.Query.DESCENDING) \
                            .limit(45) \
                            .stream()
            
            recent_matches = list(matches_ref)

            if not recent_matches:
                st.info("No completed 1v1 matches found.")
            else:
                st.write(f"Displaying the last {len(recent_matches)} completed matches.")
                
                for match_doc in recent_matches:
                    match = match_doc.to_dict()
                    game_state = match.get('game_state', {})
                    usernames = match.get('player_usernames', {})
                    winner_id = match.get('winner')
                    
                    winner_username = "Draw"
                    if winner_id and winner_id in usernames:
                        winner_username = usernames[winner_id]
                    elif winner_id:
                        winner_username = "Unknown Player"

                    p1_id, p2_id = match['players'][0], match['players'][1]
                    p1_username = usernames.get(p1_id, "Player 1")
                    p2_username = usernames.get(p2_id, "Player 2")

                    p1_charges = game_state.get('p1_charges', 'N/A')
                    p2_charges = game_state.get('p2_charges', 'N/A')
                    # The turn number is incremented before the game ends, so subtract 1
                    rounds = game_state.get('turn', 1) - 1

                    summary = f"**Winner:** {winner_username} ({rounds} rounds) - **{p1_username}** ({p1_charges} charges) vs **{p2_username}** ({p2_charges} charges)"
                    st.markdown(f"- {summary}")

        except Exception as e:
            st.error(f"Could not load 1v1 match history: {e}")
            st.warning("This may be due to a missing Firestore composite index. Please create one for the 'matches' collection with fields: status (ASC), game_over (ASC), and end_timestamp (DESC).")


# --- Main App ---
def main():
    st.set_page_config(page_title="Fireball AI", layout="centered", initial_sidebar_state="expanded")
    apply_styles()

    # Initialize session state variables
    if 'admin_mode' not in st.session_state:
        st.session_state.admin_mode = False
    if 'page' not in st.session_state:
        st.session_state.page = "Play vs AI"
    if db and 'session_logged' not in st.session_state:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            session_id = ctx.session_id

            # Use the session_id as the document ID.
            # .set() with merge=True will create the doc if it doesn't exist
            # and do nothing if it does. This is an efficient way to track uniques.
            db.collection("unique_visitors").document(session_id).set(
                {"first_visit": firestore.SERVER_TIMESTAMP}, merge=True
            )
            st.session_state.session_logged = True
        except Exception as e:
            # Silently fail on the backend if something goes wrong
            print(f"Could not log unique visitor: {e}")
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

    # Remove the condition that hides the sidebar
    st.sidebar.title("Game Menu")

    st.sidebar.markdown('<div class="how-to-play-button">', unsafe_allow_html=True)
    if st.sidebar.button("How to Play", use_container_width=True):
        st.session_state.page = "Guide"
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Always show navigation options
    page_options = ["Play vs AI", "1v1 Fireball"]
    if st.session_state.admin_mode:
        page_options.append("Admin Panel")

    # Always display radio buttons
    if st.session_state.page != "Guide":
        current_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
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
