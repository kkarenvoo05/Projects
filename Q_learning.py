import numpy as np
import pandas as pd

# Given a Markov decision process, this algorithm finds the best policy that maximizes the total expected reward
def Q_learning(data_path, n_states, n_actions, gamma=0.95, alpha=0.1, iterations=100):
    data = pd.read_csv(data_path)

    # Initialize Q-matrix as all ones (actions are 1-indexed)
    Q = np.ones((n_states, n_actions))
    transitions = data.to_numpy() # faster processing

    for _ in range(iterations):
        np.random.shuffle(transitions)
        # Update Q-values for each transition
        for s, a, r, sp in transitions:
            s = int(s) - 1
            a = int(a) - 1
            sp = int(sp) - 1

            # Q-learning update
            bestNextAction = np.argmax(Q[sp])
            td_target = r + gamma * Q[sp][bestNextAction]
            td_error = td_target - Q[s][a]
            Q[s][a] += alpha * td_error

    # Extract action with highest Q-value for each state -> optimal policy
    policy = np.argmax(Q, axis=1) + 1 
    return Q, policy

def main():
    n_states = 100  # 10x10 grid
    n_actions = 4
    
    Q, policy = Q_learning(
        data_path='/Users/karenvo/Downloads/small.csv',
        n_states=n_states,
        n_actions=n_actions,
        gamma=0.95,
        alpha=0.1,
        iterations=100
    )
    
    # Save policy
    np.savetxt('small.policy', policy, fmt='%d')
    print("Successful.")
    
    return Q, policy

if __name__ == "__main__":
    Q, policy = main()
