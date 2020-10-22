import torch
from torch.utils.data import Dataset, DataLoader
from mol_featurizer import mol_features
from word2vec import seq_to_kmers, get_protein_embedding
import pandas as pd
from gensim.models import Word2Vec


class DTADataset(Dataset):
    """ transformerCPI dataset. """

    def __init__(self, csv_file, smiles_map_file, protein_map_file, model):
        self.model = model  # Word2Vex model

        self.datalist = pd.read_csv(csv_file, sep=" ", header=None, index_col=None,
                                    names=['smiles', 'protein_seq', 'pIC50'])

        self.smiles_map = pd.read_pickle(smiles_map_file).set_index('smiles').to_dict('index')
        self.protein_map = pd.read_pickle(protein_map_file).set_index('protein_seq').to_dict('index')

    def __len__(self):
        return self.datalist.index.size

    def __getitem__(self, idx):
        data_ref = self.datalist.loc[idx]

        atom = self.smiles_map[data_ref['smiles']]['atom_feature']
        adj = self.smiles_map[data_ref['smiles']]['adj']

        protein = self.protein_map[data_ref['protein_seq']]['protein_embedding']

        label = data_ref['pIC50']

        return torch.from_numpy(atom), torch.from_numpy(adj), torch.from_numpy(protein), label


def pack(atoms, adjs, proteins, labels):
    atoms_len = 0
    proteins_len = 0

    N = len(atoms)

    atom_num = torch.zeros((N, 1))
    i = 0
    for atom in atoms:
        atom_num[i] = atom.shape[0]
        i += 1
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    protein_num = torch.zeros((N, 1))
    i = 0
    for protein in proteins:
        protein_num[i] = protein.shape[0]
        i += 1
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 34))
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len, 100))
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1

    labels_new = torch.zeros(N)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num


def collate_fn(batch):
    """
    Args batch: list of data, each atom, adj, protein, label = data
    """
    atoms, adjs, proteins, labels = zip(*batch)
    return pack(atoms, adjs, proteins, labels)


if __name__ == "__main__":
    model = Word2Vec.load("word2vec_30.model")
    dta_ds = DTADataset("data/input/sample_input.csv", "data/output/smiles_map.pkl",
                        "data/output/protein_map.pkl", model)
    dataloader = DataLoader(dta_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
