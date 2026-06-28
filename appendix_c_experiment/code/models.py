
# models.py

# Neural network definitions for the ICU-Sepsis offline learning project.

# Purpose of this file:
# 1. Define the small neural-network building blocks used by all agents.
# 2. Keep the model code readable and closely aligned with the experiment logic.
# 3. Support four agent families used in the coursework:
#    - Behavior Cloning (BC)
#    - Conservative Q-Learning (CQL)
#    - Vanilla Offline Actor-Critic (VOAC)
#    - Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic (LAADAN-AC)

# Libraries used in this file

# Importing the main PyTorch package.
import torch

# Importing PyTorch neural-network layers and module base class.
import torch.nn as nn


class MLPEncoder(nn.Module):
    
    """
    Small feed-forward encoder for the released ICU-Sepsis state features.

    The ICU-Sepsis benchmark provides one feature vector per discrete state.
    This encoder transforms the input vector into a learned latent
    representation, which is then consumed by policy or value heads.

   This class exists because:
    - It avoids repeating the same encoder definition in several models.
    - It keeps the code modular and easier to maintain.
    - It makes the architectures more consistent across baselines and the
      proposed model.
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=128, dropout=0.10):
        
        """
        Building the MLP encoder.

        Parameters
        ----------
        input_dim : int
            Number of input features per state.
        hidden_dim : int, default=128
            Width of the hidden layer.
        latent_dim : int, default=128
            Size of the output latent representation.
        dropout : float, default=0.10
            Dropout probability used as regularisation during training.
        """

        # Initialising the parent nn.Module class.
        
        super().__init__()

        # Building the encoder as a simple sequential stack:
        # Linear -> ReLU -> Dropout -> Linear -> ReLU
        #
        # This is intentionally small because the benchmark state space is not an
        # image or sequence task requiring deep or convolutional architectures.
        self.net = nn.Sequential(
            # First linear layer maps raw state features to hidden features.
            nn.Linear(input_dim, hidden_dim),

            # ReLU adds non-linearity so the encoder can learn non-linear mappings.
            nn.ReLU(),

            # Dropout acts only during training and helps reduce overfitting.
            nn.Dropout(dropout),

            # Second linear layer maps hidden features to the latent space.
            nn.Linear(hidden_dim, latent_dim),

            # Final ReLU keeps the latent representation non-linear.
            nn.ReLU(),
        )

    def forward(self, x):
        
        """
        Running a forward pass through the encoder.

        Parameters

        x : torch.Tensor
            Input state features of shape [batch_size, input_dim] or [num_states, input_dim].

        Returns

        torch.Tensor
            Encoded latent representation.
        """

        # Passing the input through the sequential MLP defined above.
        return self.net(x)


class PolicyHead(nn.Module):
    
    """
    Policy output head.

    This head maps a latent representation to one logit per discrete action.
    The head itself does not apply softmax. That is done later in training or
    evaluation depending on whether the code wants:
    - logits
    - a soft policy
    - a masked soft policy
    - a greedy one-hot policy
    """

    def __init__(self, latent_dim, num_actions):
        
        """
        Building the policy head.

        Parameters

        latent_dim : int
            Size of the incoming latent representation.
        num_actions : int
            Number of discrete actions in the benchmark.
        """

        # Initialising parent class.
        super().__init__()

        # Single linear layer projecting latent features to action logits.
        self.linear = nn.Linear(latent_dim, num_actions)

    def forward(self, z):
        
        """
        Map latent vector(s) to action logits.

        Parameters
    
        z : torch.Tensor
            Latent representation.

        Returns
      
        torch.Tensor
            One raw score (logit) per action.
        """

        # Applying the linear projection.
        return self.linear(z)


class QHead(nn.Module):
    
    """
    Q-value / score output head.

    This head maps a latent representation to one scalar score per discrete
    action. In CQL and actor-critic methods, these scores represent learned
    action values or value-like estimates.
    """

    def __init__(self, latent_dim, num_actions):
        
        """
        Building the Q head.

        Parameters
        ----------
        latent_dim : int
            Size of the incoming latent representation.
        num_actions : int
            Number of discrete actions in the benchmark.
        """

        # Initialising parent class.
        super().__init__()

        # Single linear layer projecting latent features to one value per action.
        self.linear = nn.Linear(latent_dim, num_actions)

    def forward(self, z):
        
        """
        Mapping latent vector(s) to action-value scores.

        Parameters
  
        z : torch.Tensor
            Latent representation.

        Returns
       
        torch.Tensor
            One score per action.
        """

        # Applying the linear projection.
        return self.linear(z)


class BehaviorCloningNet(nn.Module):
    
    """
    Behaviour Cloning baseline.

    This model is the simplest policy-learning baseline in the project.
    It only tries to imitate the released expert policy.

    Architectural structure:
    state features -> encoder -> policy head -> logits

    It does not learn:
    - Q-values
    - critics
    - cost estimates
    """

    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):
        
        """
        Building the Behaviour Cloning network.

        Parameters

        input_dim : int
            Number of input state features.
        num_actions : int
            Number of discrete actions.
        hidden_dim : int, default=128
            Encoder hidden size.
        latent_dim : int, default=128
            Encoder output size.
        dropout : float, default=0.10
            Dropout probability in the encoder.
        """

        # Initialising parent class.
        super().__init__()

        # Building the feature encoder for the state representation.
        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Building the policy head that converts latent features to action logits.
        self.policy_head = PolicyHead(latent_dim, num_actions)

    def forward(self, x):
        
        """
        Forward pass for Behaviour Cloning.

        Parameters
    
        x : torch.Tensor
            Input state features.

        Returns
 
        torch.Tensor
            Policy logits over actions.
        """

        # Encoding the raw state features into a latent representation.
        z = self.encoder(x)

        # Mapping latent representation to action logits.
        return self.policy_head(z)


class ConservativeQNet(nn.Module):
    
    """
    Conservative fitted-Q baseline.

    This model is used by the CQL baseline.
    It learns one Q-value for each discrete action and later training code
    applies a conservative offline-RL regulariser to these Q-values.

    Architectural structure:
    state features -> encoder -> Q head -> Q-values
    """

    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):
        
        """
        Building the Conservative Q network.

        Parameters
    
        input_dim : int
            Number of input state features.
        num_actions : int
            Number of discrete actions.
        hidden_dim : int, default=128
            Encoder hidden size.
        latent_dim : int, default=128
            Encoder output size.
        dropout : float, default=0.10
            Dropout probability in the encoder.
        """

        # Initialising parent class.
        super().__init__()

        # Building the state encoder.
        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Building the Q-value head that outputs one score per action.
        self.q_head = QHead(latent_dim, num_actions)

    def forward(self, x):
        
        """
        Forward pass for the Conservative Q network.

        Parameters
   
        x : torch.Tensor
            Input state features.

        Returns
     
        torch.Tensor
            Q-values for all actions.
        """

        # Encoding state features into latent space.
        z = self.encoder(x)

        # Mapping latent features to Q-values.
        return self.q_head(z)


class OfflineActorCriticNet(nn.Module):
    
    """
    Generic offline actor-critic backbone.

    Shared by:
    - VOAC
    - LAADAN-AC
    - LAADAN-AC-PD / LAADAN-AC-RC

    Outputs:
    - actor logits
    - two reward critics q1 and q2
    - optional cost critic ensemble

    If num_cost_heads = 1, the cost output has shape [num_states, num_actions].
    If num_cost_heads > 1, the cost output has shape [num_cost_heads, num_states, num_actions].
    """

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_dim=128,
        latent_dim=128,
        dropout=0.10,
        use_cost_head=False,
        num_cost_heads=1,
    ):
        super().__init__()

        self.actor_encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.critic_encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.actor_head = PolicyHead(latent_dim, num_actions)

        self.q1_head = QHead(latent_dim, num_actions)
        self.q2_head = QHead(latent_dim, num_actions)

        self.use_cost_head = bool(use_cost_head)
        self.num_cost_heads = int(max(1, num_cost_heads))

        if self.use_cost_head:
            self.cost_heads = nn.ModuleList(
                [QHead(latent_dim, num_actions) for _ in range(self.num_cost_heads)]
            )

            # Backward-compatible alias used by older LAADAN-AC code.
            self.cost_head = self.cost_heads[0]
        else:
            self.cost_heads = None
            self.cost_head = None

    def actor_logits(self, x):
        actor_latent = self.actor_encoder(x)
        return self.actor_head(actor_latent)

    def critic_values(self, x):
        critic_latent = self.critic_encoder(x)

        q1 = self.q1_head(critic_latent)
        q2 = self.q2_head(critic_latent)

        if self.cost_heads is None:
            cost = None
        else:
            cost_values = [head(critic_latent) for head in self.cost_heads]

            if len(cost_values) == 1:
                cost = cost_values[0]
            else:
                cost = torch.stack(cost_values, dim=0)

        return q1, q2, cost

    def forward(self, x):
        logits = self.actor_logits(x)
        q1, q2, cost = self.critic_values(x)

        return {
            "logits": logits,
            "q1": q1,
            "q2": q2,
            "cost": cost,
        }

@torch.no_grad()
def soft_update(target_module, source_module, tau):
    
    """
    Standard target-network soft update.

    This function updates the parameters of a target network so that they move
    gradually toward the parameters of a source network.

    Update rule:
        target = tau * source + (1 - tau) * target

    Interpretation of tau:
    - tau close to 0  -> slow tracking, smoother but less responsive
    - tau close to 1  -> fast tracking, more responsive but less stable

    The @torch.no_grad() decorator is used because this is a parameter-copying
    operation, not something that should create gradients for backpropagation.
    """

    # Looping through target and source parameters in matching order.
    for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
        # Replacing target parameter data with the soft-updated combination.
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
