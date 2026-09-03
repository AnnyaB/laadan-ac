
import torch

import torch.nn as nn


class MLPEncoder(nn.Module):


    def __init__(self, input_dim, hidden_dim=128, latent_dim=128, dropout=0.10):


        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, latent_dim),

            nn.ReLU(),
        )

    def forward(self, x):      # x shape = [716, 47]

        return self.net(x)

class PolicyHead(nn.Module):


    def __init__(self, latent_dim, num_actions):

        super().__init__()

        # Single linear layer projecting latent features to action logits.
        self.linear = nn.Linear(latent_dim, num_actions)


    def forward(self, z):

        return self.linear(z) # 25 logits


class QHead(nn.Module):


    def __init__(self, latent_dim, num_actions):

        super().__init__()

        self.linear = nn.Linear(latent_dim, num_actions)

    def forward(self, z):

        return self.linear(z)


class BehaviorCloningNet(nn.Module):


    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):

        super().__init__()

        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.policy_head = PolicyHead(latent_dim, num_actions) # 128 latent features -> 25 action logits


    def forward(self, x):

        z = self.encoder(x)

        return self.policy_head(z)

class ConservativeQNet(nn.Module):


    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):

        super().__init__()

        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )  # converts the 47 state features into a 128-dimensional latent representation.

        self.q_head = QHead(latent_dim, num_actions)  # This converts the 128 latent features into 25 Q-values.


    def forward(self, x):

        z = self.encoder(x)

        return self.q_head(z)

    """ state x -> latent representation z -> Q-values for all 25 actions
    CQL later chooses the action with the highest Q-value. conservative penalty is not in models.py. It is in trainers.py. """

class OfflineActorCriticNet(nn.Module):

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_dim=128,
        latent_dim=128,
        dropout=0.10,
        use_cost_head=False,
    ):

        super().__init__()


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
        self.critic_encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.actor_head = PolicyHead(latent_dim, num_actions)


        # First reward/value critic head.
        self.q1_head = QHead(latent_dim, num_actions)

        # Second reward/value critic head.
        # Using two critics is common in actor-critic methods because it can
        # reduce over-optimistic value estimates.
        self.q2_head = QHead(latent_dim, num_actions)


        self.use_cost_head = bool(use_cost_head)

        self.cost_head = QHead(latent_dim, num_actions) if self.use_cost_head else None

        # The cost head outputs: 25 cost estimates, for how unsafe/costly might each action be in this state

    def actor_logits(self, x):

        actor_latent = self.actor_encoder(x)

        return self.actor_head(actor_latent)

    def critic_values(self, x):

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

    for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):

        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
        """
        If:

            tau = 0.01

        then:

            target = 1% new model + 99% old target

        So the target network moves slowly. It makes Bellman target learning more stable,
        because the target does not change too violently every epoch."""
