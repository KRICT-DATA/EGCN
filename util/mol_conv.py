import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
from mendeleev import get_table


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sel_prop_names = ['atomic_weight',
                'atomic_radius',
                'atomic_volume',
                'dipole_polarizability',
                'fusion_heat',
                'thermal_conductivity',
                'vdw_radius',
                'en_pauling']
dim_atomic_feat = len(sel_prop_names)
dim_self_feat = 3


class molDGLGraph(dgl.DGLGraph):
    def __init__(self, smiles, adj_mat, feat_mat, mol):
        super(molDGLGraph, self).__init__()
        self.smiles = smiles
        self.adj_mat = adj_mat
        self.feat_mat = feat_mat
        self.atomic_nodes = []
        self.neighbors = {}

        node_id = 0
        for atom in mol.GetAtoms():
            self.atomic_nodes.append(atom.GetSymbol())
            self.neighbors[node_id] = atoms_to_symbols(atom.GetNeighbors())
            node_id += 1


def read_atom_prop():
    tb_atomic_props = get_table('elements')
    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=np.int)
    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=np.float32))
    arr_atomic_props = util.zscore(arr_atomic_props)
    atomic_props_mat = {arr_atomic_nums[i]: arr_atomic_props[i, :] for i in range(0, arr_atomic_nums.shape[0])}

    return atomic_props_mat


def construct_mol_graph(smiles, mol, adj_mat, feat_mat):
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol)
    edges = util.adj_mat_to_edges(adj_mat)
    src, dst = tuple(zip(*edges))

    molGraph.add_nodes(adj_mat.shape[0])
    molGraph.add_edges(src, dst)
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32).to(device)

    return molGraph


def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        node_feat_mat = np.empty([mol.GetNumAtoms(), atomic_props.get(1).shape[0]])

        ind = 0
        for atom in mol.GetAtoms():
            node_feat_mat[ind, :] = atomic_props.get(atom.GetAtomicNum())
            ind = ind + 1

        return mol, construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
    except:
        print(smiles + ' could not be converted to molecular graph due to the internal errors of RDKit')
        return None, None


def atoms_to_symbols(atoms):
    symbols = []

    for atom in atoms:
        symbols.append(atom.GetSymbol())

    return symbols


def normalize_self_feat(mol_graphs, self_feat_name):
    self_feats = []

    for mol_graph in mol_graphs:
        self_feats.append(getattr(mol_graph, self_feat_name))

    mean_self_feat = np.mean(self_feats)
    std_self_feat = np.std(self_feats)

    for mol_graph in mol_graphs:
        if std_self_feat == 0:
            setattr(mol_graph, self_feat_name, 0)
        else:
            setattr(mol_graph, self_feat_name, (getattr(mol_graph, self_feat_name) - mean_self_feat) / std_self_feat)


def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pandas.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = np.array(data_mat[:, 1:3], dtype=np.float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:
            mol_graph.num_atoms = mol.GetNumAtoms()
            mol_graph.weight = dsc.ExactMolWt(mol)
            mol_graph.num_rings = mol.GetRingInfo().NumRings()

            mol_graph.max_abs_charge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.min_abs_charge = dsc.MinAbsPartialCharge(mol)
            mol_graph.num_rad_elc = dsc.NumValenceElectrons(mol)
            mol_graph.num_val_elc = dsc.NumValenceElectrons(mol)

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    normalize_self_feat(mol_graphs, 'num_atoms')
    normalize_self_feat(mol_graphs, 'weight')
    normalize_self_feat(mol_graphs, 'num_rings')
    normalize_self_feat(mol_graphs, 'max_abs_charge')
    normalize_self_feat(mol_graphs, 'min_abs_charge')
    normalize_self_feat(mol_graphs, 'num_rad_elc')
    normalize_self_feat(mol_graphs, 'num_val_elc')

    return samples


atomic_props = read_atom_prop()
