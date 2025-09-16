import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from model import Net

class Controller(nn.Module):
    """
    Controller for Neural Architecture Search (NAS).
    This is the "RL agent" (a GRU-based RNN) that samples actions,
    where each action corresponds to part of a neural network architecture.
    
    Workflow:
    1. Controller generates a sequence of actions (layer sizes, activations, EOS).
    2. A child network (Net) is instantiated using those actions.
    3. The child network is trained and evaluated, producing a reward (accuracy).
    4. REINFORCE updates the controller so it is more likely to propose good architectures.
    """
    def __init__(self, num_actions=10, hidden_size=64):
        super(Controller, self).__init__()

        # GRU cell = recurrent unit that keeps track of controller state
        self.cell = nn.GRUCell(
            input_size=num_actions,
            hidden_size=hidden_size
        )

        # Fully connected layer to map hidden state -> logits over possible actions
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_actions
        )

        # Hyperparameters
        self.num_actions = num_actions   # number of discrete choices (layer sizes + activations + EOS)
        self.hidden_size = hidden_size   # GRU hidden dimension
        self.epsilon = 0.8               # (not directly used here, possibly for exploration if extended)
        self.gamma = 1.0                 # reward discount factor
        self.beta = 0.01                 # entropy regularization weight (encourages exploration)
        self.max_depth = 6               # max number of steps in a sampled architecture
        self.clip_norm = 0               # gradient clipping threshold (0 = disabled)

        # Buffers for RL
        self.log_probs = []              # log-probabilities of sampled actions
        self.actions = []                # sequence of sampled actions
        self.entropies = []              # entropy of action distributions
        self.reward = None               # final reward (child network accuracy + penalties)

        # Mapping from indices -> actual architectural choices
        self.index_to_action = {
            0: 1,          # layer size 1
            1: 2,          # layer size 2
            2: 4,          # layer size 4
            3: 8,          # layer size 8
            4: 16,         # layer size 16
            5: 'Sigmoid',  # activation function
            6: 'Tanh',
            7: 'ReLU',
            8: 'LeakyReLU',
            9: 'EOS'       # end of sequence
        }

        # Optimizer for updating controller parameters (policy gradient)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)


    def forward(self, x, h):
        """
        Forward step through the GRU controller.
        Given input token x and hidden state h:
          - Update hidden state using GRU
          - Produce logits for next action
        """
        x = x.unsqueeze(dim=0)  # add batch dimension
        h = h.unsqueeze(dim=0)

        # Update hidden state
        h = self.cell(x, h)

        # Map hidden state to logits over actions
        x = self.fc(h)

        # Remove batch dimension
        x = x.squeeze(dim=0)
        h = h.squeeze(dim=0)

        return x, h

    
    def step(self, state):
        """
        Perform one action-sampling step:
        - Get logits from controller
        - Sample an action (stochastic policy)
        - Apply constraints/penalties
        - Check termination condition (EOS or max depth)
        """
        # Pass dummy input (zeros) since controller only relies on hidden state
        logits, new_state = self(torch.zeros(self.num_actions), state)

        # Sample from categorical distribution defined by logits
        idx = torch.distributions.Categorical(logits=logits).sample()
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)

        # Convert index to actual action (layer size, activation, or EOS)
        action = self.index_to_action[int(idx)]
        self.actions.append(action)

        # Compute entropy for exploration bonus
        entropy = -(log_probs * probs).sum(dim=-1)
        self.entropies.append(entropy)

        # Apply constraints to avoid nonsensical architectures
        if action == 'EOS' and len(self.actions) <= 2:
            # Ending too early is penalized
            self.reward -= 1
        elif len(self.actions) >= 2 and isinstance(self.actions[-1], int) and isinstance(self.actions[-2], int):
            # Two consecutive layer sizes (no activation in between) is penalized
            self.reward -= 0.1
        elif len(self.actions) >= 2 and isinstance(self.actions[-1], str) and isinstance(self.actions[-2], str) and action != 'EOS':
            # Two consecutive activations (without a size in between) is penalized
            self.reward -= 0.1

        # Termination condition: EOS or reaching maximum depth
        terminate = action == 'EOS' or len(self.actions) == self.max_depth

        return log_probs[idx], new_state, terminate


    def generate_rollout(self, iter_train, iter_dev, verbose=True):
        """
        Generate one full rollout (one proposed child network):
        - Sample a sequence of actions until EOS or max depth
        - Instantiate child network Net(actions)
        - Train/evaluate child network
        - Use validation accuracy as reward
        """
        # Reset buffers
        self.log_probs = []
        self.actions = []
        self.entropies = []
        self.reward = 0

        # Initial hidden state = zero
        state = torch.zeros(self.hidden_size)
        terminated = False

        # Keep sampling actions until termination
        while not terminated:
            log_prob, state, terminated = self.step(state)
            self.log_probs.append(log_prob)

        if verbose:
            print('\nGenerated network:')
            print(self.actions)

        # Build and train the child network defined by sampled actions
        net = Net(self.actions)
        accuracy = net.fit(iter_train, iter_dev)

        # Final reward = accuracy (+ any penalties from step)
        self.reward += accuracy

        return self.reward

    
    def optimize(self):
        """
        Update controller parameters using REINFORCE:
        - Compute discounted returns
        - Compute policy gradient loss
        - Add entropy bonus for exploration
        - Backprop and update parameters
        """
        G = torch.ones(1) * self.reward  # initialize return with reward
        loss = 0

        # Iterate over sampled log_probs (in reverse)
        for i in reversed(range(len(self.log_probs))):
            G = self.gamma * G  # discounting (gamma=1 here, so no decay)
            # REINFORCE objective: maximize log_prob * reward
            loss = loss - (self.log_probs[i] * Variable(G)) - self.beta * self.entropies[i]

        # Normalize loss
        loss /= len(self.log_probs)
        
        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        self.optimizer.step()

        return float(loss.data.numpy())
