import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GATLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=False)
        self.attn_fc = nn.Linear(2 * dim_out, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(dim_in, dim_out))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super(Net, self).__init__()
        self.gc1 = MultiHeadGATLayer(dim_in, 100, num_heads)
        self.gc2 = MultiHeadGATLayer(100 * num_heads, 20, 1)
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
