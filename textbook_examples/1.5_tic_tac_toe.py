import random
import numpy as np

class TicTacToe:
    """
    Tic-tac-toe environment.
    The board is represented as a list of length 9 with:
      0 = empty
      1 = 'X'
      2 = 'O'
    Player 1 is 'X', Player 2 is 'O'.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to start a new game."""
        self.board = [0]*9  # 9 empty cells
        self.current_player = 1  # X always starts
        self.done = False
        self.winner = None
        return self.get_state(), self.current_player

    def get_state(self):
        """
        Return an immutable representation (tuple) of the board,
        plus whose turn it is, so each turn is a distinct state.
        """
        return tuple(self.board), self.current_player

    def available_actions(self):
        """Return the list of valid actions (indices of empty cells)."""
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        """
        Place the current player's mark at the given action (0-8).
        Returns:
          next_state, reward, done, info
        """
        if self.done:
            raise ValueError("Game has ended. Reset to play again.")

        # Execute the move
        self.board[action] = self.current_player

        # Check for terminal conditions
        self._check_game_status()

        # If game is done, compute reward from perspective of the current player
        if self.done:
            if self.winner is None:
                # Draw
                reward = 0.5
            else:
                # Current player just caused a win
                reward = 1.0
        else:
            # Non-terminal, no immediate reward
            reward = 0.0

        # Switch player if not done
        if not self.done:
            self.current_player = 1 if self.current_player == 2 else 2

        next_state = self.get_state()
        return next_state, reward, self.done, {}

    def _check_game_status(self):
        """Check if there's a winner or if the board is full."""
        # Possible winning combinations:
        wins = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # columns
            (0,4,8), (2,4,6)            # diagonals
        ]
        for (a,b,c) in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                self.done = True
                self.winner = self.board[a]
                return

        if all(self.board[i] != 0 for i in range(9)):
            # Board is full, it's a draw
            self.done = True
            self.winner = None


class RLAgent:
    """
    Simple Reinforcement Learning agent for tic-tac-toe.
    Maintains a value function: state -> estimated probability of winning.
    """
    def __init__(self, alpha=0.1, epsilon=0.1):
        self.alpha = alpha          # learning rate
        self.epsilon = epsilon      # exploration rate
        self.values = {}            # state -> value
        self.default_value = 0.5    # default estimate for unseen states

    def get_value(self, state):
        """
        Return the value of a given state from this agent's perspective.
        'state' includes the board layout and which player's turn it is.
        """
        return self.values.get(state, self.default_value)

    def update_value(self, state, new_value):
        """Set/update the value for a given state."""
        self.values[state] = new_value

    def choose_action(self, env):
        """
        Choose an action (0-8) given the environment.
        With probability epsilon, choose a random valid action.
        Otherwise choose the action that leads to the highest-value next state.
        """
        valid_actions = env.available_actions()
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy w.r.t. value function
        best_value = -float('inf')
        best_action = None
        for action in valid_actions:
            # Temporarily make the move
            board_copy = list(env.board)
            board_copy[action] = env.current_player
            next_state = (tuple(board_copy), 1 if env.current_player == 2 else 2)
            # Evaluate
            value = self.get_value(next_state)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def update_episode_history(self, episode_history, final_reward):
        """
        After a game ends, update the value function for each visited state.
        We use an incremental update approach:
            V(s) <- V(s) + alpha * [V(s') - V(s)]
        Then for the last state, we update toward the final reward.
        """
        # episode_history is a list of (state, next_state) for each step
        # final_reward is the eventual outcome from this agent's perspective
        # We'll walk backward, updating each state's value.
        for i in range(len(episode_history) - 1):
            s = episode_history[i]
            s_next = episode_history[i+1]
            v_s = self.get_value(s)
            v_s_next = self.get_value(s_next)
            updated_value = v_s + self.alpha * (v_s_next - v_s)
            self.update_value(s, updated_value)

        # For the last state in the history, update toward final_reward
        if len(episode_history) > 0:
            last_state = episode_history[-1]
            v_last = self.get_value(last_state)
            updated_value = v_last + self.alpha * (final_reward - v_last)
            self.update_value(last_state, updated_value)


def play_game(env, agent_X, agent_O):
    """
    Play one episode (one game) using the environment and two agents.
    Returns a log of visited states for X and O separately, plus final rewards.
    """
    # Each agent keeps track of the states it visited (for updating its value function).
    episode_history_X = []
    episode_history_O = []

    state, player = env.reset()

    done = False
    reward_X = 0.0
    reward_O = 0.0

    while not done:
        if player == 1:
            # X's turn
            action = agent_X.choose_action(env)
        else:
            # O's turn
            action = agent_O.choose_action(env)

        next_state, reward, done, info = env.step(action)

        # Record state transitions for each agent
        if player == 1:
            episode_history_X.append(state)
        else:
            episode_history_O.append(state)

        # If the game ended, assign final rewards
        if done:
            # reward is from perspective of the player who moved
            if reward == 1.0:
                if player == 1:
                    reward_X = 1.0
                    reward_O = 0.0
                else:
                    reward_X = 0.0
                    reward_O = 1.0
            elif reward == 0.5:
                # Draw
                reward_X = 0.5
                reward_O = 0.5
        else:
            # Switch current state/player
            state = next_state
            player = env.current_player

    return episode_history_X, episode_history_O, reward_X, reward_O


def evaluate_agents(agent_X, agent_O, num_episodes=1000):
    """
    Evaluate the performance of two agents by letting them play
    num_episodes games with epsilon=0 (purely greedy).
    Returns the counts of X_wins, O_wins, draws.
    """
    # Temporarily store original epsilons
    old_epsilon_X = agent_X.epsilon
    old_epsilon_O = agent_O.epsilon
    # Set to 0 for pure greedy evaluation
    agent_X.epsilon = 0.0
    agent_O.epsilon = 0.0

    env = TicTacToe()
    X_wins = 0
    O_wins = 0
    draws = 0

    for _ in range(num_episodes):
        episode_history_X, episode_history_O, reward_X, reward_O = play_game(env, agent_X, agent_O)
        # No learning updates here—just evaluation
        if reward_X == 1.0:
            X_wins += 1
        elif reward_O == 1.0:
            O_wins += 1
        else:
            draws += 1

    # Restore original epsilons
    agent_X.epsilon = old_epsilon_X
    agent_O.epsilon = old_epsilon_O

    return X_wins, O_wins, draws


def train_and_evaluate(num_episodes=10000, alpha=0.1, epsilon=0.1, eval_interval=2000):
    """
    Train two RL agents (X and O) in self-play and print training statistics.
    Periodically evaluate the agents with epsilon=0 to gauge progress.
    """
    env = TicTacToe()
    agent_X = RLAgent(alpha=alpha, epsilon=epsilon)
    agent_O = RLAgent(alpha=alpha, epsilon=epsilon)

    # For tracking training outcomes (win/draw/loss for X and O)
    X_wins = 0
    O_wins = 0
    draws = 0

    for episode in range(1, num_episodes+1):
        episode_history_X, episode_history_O, reward_X, reward_O = play_game(env, agent_X, agent_O)

        # Update X's value function
        agent_X.update_episode_history(episode_history_X, reward_X)
        # Update O's value function
        agent_O.update_episode_history(episode_history_O, reward_O)

        # Tally results for this episode
        if reward_X == 1.0:
            X_wins += 1
        elif reward_O == 1.0:
            O_wins += 1
        else:
            draws += 1

        # Print training metrics at regular intervals
        if episode % eval_interval == 0:
            total = X_wins + O_wins + draws
            x_win_rate = X_wins / total
            o_win_rate = O_wins / total
            draw_rate  = draws  / total

            print(f"--- Training stats at episode {episode} ---")
            print(f"X win rate  : {x_win_rate:.2f}")
            print(f"O win rate  : {o_win_rate:.2f}")
            print(f"Draw rate   : {draw_rate:.2f}")
            print("----------------------------------------")

            # Evaluate with epsilon=0 to see how well the agents perform greedily
            eval_episodes = 1000
            eval_X_wins, eval_O_wins, eval_draws = evaluate_agents(agent_X, agent_O, num_episodes=eval_episodes)
            print(f"Evaluation over {eval_episodes} episodes (greedy play):")
            print(f"  X wins : {eval_X_wins}")
            print(f"  O wins : {eval_O_wins}")
            print(f"  Draws  : {eval_draws}")
            print()

            # Reset counters so we can see stats for the *next* interval
            X_wins = 0
            O_wins = 0
            draws = 0

    return agent_X, agent_O


if __name__ == "__main__":
    # Train and evaluate the agents
    agent_X, agent_O = train_and_evaluate(num_episodes=10000, alpha=0.1, epsilon=0.1, eval_interval=2000)

    # Final evaluation with epsilon=0
    print("Final evaluation with epsilon=0 after full training:")
    X_wins, O_wins, draws = evaluate_agents(agent_X, agent_O, num_episodes=2000)
    print(f"Over 2000 games:")
    print(f"  X wins : {X_wins}")
    print(f"  O wins : {O_wins}")
    print(f"  Draws  : {draws}")
    print()

    # Quick check: play a single game with the learned agents (epsilon=0)
    env = TicTacToe()
    agent_X.epsilon = 0.0
    agent_O.epsilon = 0.0

    state, player = env.reset()
    done = False

    print("Single game demonstration with greedy (epsilon=0) agents:")
    while not done:
        if player == 1:
            action = agent_X.choose_action(env)
        else:
            action = agent_O.choose_action(env)

        state, reward, done, info = env.step(action)
        board, _ = state

        # Print the board in a readable 3x3 format
        symbols = {0: ".", 1: "X", 2: "O"}
        print("\nBoard:")
        for i in range(0, 9, 3):
            print(" ".join(symbols[board[j]] for j in range(i, i+3)))
        print()

        if done:
            if reward == 1.0:
                if player == 1:
                    print("X wins!")
                else:
                    print("O wins!")
            elif reward == 0.5:
                print("It's a draw!")
        else:
            player = env.current_player
