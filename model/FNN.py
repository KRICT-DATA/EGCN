import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, dim_out)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out
