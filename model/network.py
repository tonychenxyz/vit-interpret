import torch
import torch.nn as nn
import torch.nn.functional as F

# class LinearReadOut(nn.Module):
#     def __init__(self, out_dim=2, hidden_dim=512):
#         super(LinearReadOut, self).__init__()
#         self.fc_c1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_c2 = nn.Linear(hidden_dim, out_dim)

#     def forward(self, z):
#         z = self.fc_c1(z)
#         z = self.fc_c2(F.relu(z))
#         return z


class LinearReadOut(nn.Module):
    def __init__(self, out_dim=2, hidden_dim=512):
        super(LinearReadOut, self).__init__()
        # self.fc_c1 = nn.Linear(hidden_dim, hidden_dim)
        # self.BN = nn.BatchNorm1d(hidden_dim)
        self.fc_c2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        # z = self.fc_c1(z)
        z = self.fc_c2(z)
        return z
