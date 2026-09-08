
import torch

import torch.nn as nn


# models
# state encoder
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

    def forward(self, x):

        return self.net(x)


# policy head
class PolicyHead(nn.Module):


    def __init__(self, latent_dim, num_actions):

        super().__init__()


        self.linear = nn.Linear(latent_dim, num_actions)

    def forward(self, z):

        return self.linear(z)


# q head
class QHead(nn.Module):

    def __init__(self, latent_dim, num_actions):


        super().__init__()


        self.linear = nn.Linear(latent_dim, num_actions)

    def forward(self, z):


        return self.linear(z)


# bc model
class BehaviorCloningNet(nn.Module):



    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):



        super().__init__()


        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )


        self.policy_head = PolicyHead(latent_dim, num_actions)

    def forward(self, x):


        z = self.encoder(x)


        return self.policy_head(z)


# cql model
class ConservativeQNet(nn.Module):

    def __init__(self, input_dim, num_actions, hidden_dim=128, latent_dim=128, dropout=0.10):



        super().__init__()


        self.encoder = MLPEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )


        self.q_head = QHead(latent_dim, num_actions)

    def forward(self, x):




        z = self.encoder(x)


        return self.q_head(z)


# actor and twin critics
class OfflineActorCriticNet(nn.Module):


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
# target update
def soft_update(target_module, source_module, tau):



    for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):

        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
