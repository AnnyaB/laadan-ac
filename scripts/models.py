
# models.py

# Neural network definitions for the ICU-Sepsis, eICU, and other experiments.


# Libraries used

# Importing the main PyTorch package.
import torch

# Importing PyTorch neural-network layers and module base class.
import torch.nn as nn


class MLPEncoder(nn.Module):
    
    """
    Small feed-forward encoder for tabular MDP state features.

    The ICU-Sepsis benchmark provides one feature vector per discrete state.
    This encoder transforms that input vector into a learned latent
    representation, which is then consumed by policy or value heads.

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
            nn.Dropout(dropout),   # During evaluation, dropout is switched off by: model.eval()

            # Second linear layer maps hidden features to the latent space.
            nn.Linear(hidden_dim, latent_dim),    # Take the 128 hidden features and learn a second transformation 
                                                  # that refines them into a final 128-dimensional latent representation.

            # Final ReLU keeps the latent representation non-linear.
            nn.ReLU(),
        )

    def forward(self, x):      # x shape = [716, 47]
        
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
        return self.net(x)  # passes x through the full sequence: Linear -> ReLU -> Dropout -> Linear -> ReLU
    
        # output 128-dimensional latent representation 
        # For all 716 states: [716, 47] -> [716, 128]


class PolicyHead(nn.Module):  # defines the layer that produces action scores.
    
    # It is used by:
    # BC
    # VOAC actor
    # LAADAN-AC actor

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

    def __init__(self, latent_dim, num_actions):   # latent_dim = 128, num_actions = 25

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
        

    def forward(self, z):  # z is the 128-dimensional latent representation from the encoder.
        
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
        return self.linear(z) # 25 logits
    
    # also the policy head does not choose the action by itself. 
    # It only gives raw scores. The training/evaluation code later applies softmax, greedy selection, or masking.


class QHead(nn.Module): # defines a layer that outputs Q-values.
    
    # It is used by:

    # CQL
    # VOAC critic Q1/Q2
    # LAADAN-AC critic Q1/Q2
    # LAADAN-AC cost head

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
        self.linear = nn.Linear(latent_dim, num_actions)   # nn.Linear(128, 25), this time the 25 outputs are interpreted as: 
                                                           # Q-values or cost values
                                                           # For instance, for Q values - How good is each action in this state for long-term reward/survival?
                                                           # For cost values: How costly/unsafe is each action in this state?

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


class BehaviorCloningNet(nn.Module):  # defines the BC baseline, Learn to imitate the expert policy.
    
    """
    Behaviour Cloning baseline.

    This model is the simplest policy-learning baseline in the project.
    It only tries to imitate the released expert policy.

    Architectural structure:
    state features -> encoder -> policy head -> 25 logits
    
    For one state: 47 -> 128 -> 128 -> 25

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
        )  # converts the 47 state features into a 128-dimensional latent representation.

        # Building the policy head that converts latent features to action logits.
        self.policy_head = PolicyHead(latent_dim, num_actions) # 128 latent features -> 25 action logits
        

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
    
    # input state -> encode it into z -> convert z into action logits
    
    # BC does not have: 
    # Q1
    # Q2
    # critic
    # cost head
    # target network
    # Bellman update


class ConservativeQNet(nn.Module):  # defines the CQL baseline.
    
    """
    Conservative fitted-Q baseline.

    This model is used by the CQL baseline.
    It learns one Q-value for each discrete action and later training code
    applies a conservative offline-RL regulariser to these Q-values.

    Architectural structure:
    state features -> encoder -> Q head -> 25 Q-values

    For one state: 47 -> 128 -> 128 -> 25
    
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
        )  # converts the 47 state features into a 128-dimensional latent representation.

        # Building the Q-value head that outputs one score per action.
        self.q_head = QHead(latent_dim, num_actions)  # This converts the 128 latent features into 25 Q-values.
        

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
    
    """ state x -> latent representation z -> Q-values for all 25 actions
    CQL later chooses the action with the highest Q-value. conservative penalty is not in models.py. It is in trainers.py. """
    
class OfflineActorCriticNet(nn.Module):
    
    """
    Generic offline actor-critic backbone.

    This class is shared by:
    - VOAC: the plain offline actor-critic ablation
    - LAADAN-AC: the proposed admissibility-aware actor-critic

    The actor and critic use separate encoders rather than a single shared one.

    Output components:
    - actor logits
    - first Q head
    - second Q head
    - optional cost head
    """

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_dim=128,
        latent_dim=128,
        dropout=0.10,
        use_cost_head=False,
    ):
        """
        Building the actor-critic backbone.

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
            Dropout probability in the encoders.
        use_cost_head : bool, default=False
            If True, include an extra cost head.
            This is used by LAADAN-AC but not by VOAC.
        """

        # Initialising parent class.
        super().__init__()


        # Actor branch

        # Separating encoder for policy learning.
        self.actor_encoder = MLPEncoder(        
            input_dim,  
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        """ For LAADAN-AC:
        
            47 features
            -> Linear(47,128)
            -> ReLU
            -> Dropout(0.10)
            -> Linear(128,128)
            -> ReLU
            -> 128-dimensional latent representation

            So, the actor's job is to understand the current ICU state before choosing an action."""

        # Critic branch

        # Separating encoder for value learning.
        self.critic_encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Policy head produces one logit per action from actor latent features.
        # Then the training code converts them into a policy using softmax/masking.
        self.actor_head = PolicyHead(latent_dim, num_actions)

        
        # First reward/value critic head.
        self.q1_head = QHead(latent_dim, num_actions)

        # Second reward/value critic head.
        # Using two critics is common in actor-critic methods because it can
        # reduce over-optimistic value estimates.
        self.q2_head = QHead(latent_dim, num_actions)

        # Recording whether this network should include a cost head.
        self.use_cost_head = bool(use_cost_head)

        # If requested, create the cost head; otherwise store None.
        #
        # LAADAN-AC uses this cost head to model inadmissibility-related cost.
        # VOAC does not use it.
        
        self.cost_head = QHead(latent_dim, num_actions) if self.use_cost_head else None
        
        # The cost head outputs: 25 cost estimates, for how unsafe/costly might each action be in this state

    def actor_logits(self, x):
        
        """
        Computing policy logits from the actor branch only.

        Parameters
       
        x : torch.Tensor
            Input state features.

        Returns
   
        torch.Tensor
            Action logits from the actor.
        """

        # Encoding state features using the actor-specific encoder.
        actor_latent = self.actor_encoder(x)

        # Converting actor latent features into action logits.
        return self.actor_head(actor_latent)

    def critic_values(self, x):
        
        """
        Computing critic outputs from the critic branch only.

        Parameters
        
        x : torch.Tensor
            Input state features.

        Returns
        
        tuple
            (q1, q2, cost)
            where cost is None if no cost head exists.
        """

        # Encoding state features using the critic-specific encoder.
        critic_latent = self.critic_encoder(x)

        # First critic estimates one score per action.
        q1 = self.q1_head(critic_latent)

        # Second critic estimates one score per action.
        q2 = self.q2_head(critic_latent)
        
        # These produce two Q-value tables.

        # If cost head exists, compute cost scores too; otherwise return None.
        cost = self.cost_head(critic_latent) if self.cost_head is not None else None

        return q1, q2, cost
    

    def forward(self, x):
        
        """
        Full forward pass for the actor-critic model.

        Parameters
      
        x : torch.Tensor
            Input state features.

        Returns
  
        dict
            Dictionary containing:
            - "logits": actor policy logits
            - "q1": first critic output
            - "q2": second critic output
            - "cost": optional cost output or None
        """

        # Getting actor logits from actor branch.
        logits = self.actor_logits(x)

        # Getting critic outputs from critic branch.
        q1, q2, cost = self.critic_values(x)

        # Returning all outputs together so training code can access whichever
        # parts it needs.
        return {
            "logits": logits, # Actor action scores
            "q1": q1,         # First Q-value estimate
            "q2": q2,         # Second Q-value estimate
            "cost": cost,     # Cost estimate, only for LAADAN-AC
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
        
        # new target = tau x main model + (1 - tau) x old target
        
        """ 
        If:

            tau = 0.01

        then:

            target = 1% new model + 99% old target

        So the target network moves slowly. It makes Bellman target learning more stable, 
        because the target does not change too violently every epoch."""
        
