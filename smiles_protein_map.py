from mol_featurizer import mol_features
from word2vec import seq_to_kmers, get_protein_embedding
import pandas as pd
from gensim.models import Word2Vec

if __name__ == "__main__":
    csv_file = "data/input/converted_all_data_drop_RCX_PDB.csv"

    model = Word2Vec.load("word2vec_30.model")
    print("loading word2vec.model finished!", flush=True)

    datalist = pd.read_csv(csv_file, sep=" ", header=None, index_col=None, names=['smiles', 'protein_seq', 'pIC50'])
    print("loading {} finished!".format(csv_file), flush=True)
    
    protein_map = datalist['protein_seq'].drop_duplicates().to_frame().reset_index(drop=True)
    print("create protein_map finished!", flush=True)
    protein_map['protein_embedding'] = protein_map.apply(lambda x: get_protein_embedding(model, seq_to_kmers(x['protein_seq'])),  axis=1)
    print("add protein_map embedding finished!", flush=True)
    protein_map.to_pickle("data/output/protein_map.pkl")
    print("save protein_map finished!", flush=True)
    
    smiles_map = datalist['smiles'].drop_duplicates().to_frame().reset_index(drop=True)
    print("create smiles_map finished!", flush=True)
    smiles_map[['atom_feature', 'adj']] = smiles_map.apply(lambda x: mol_features(x['smiles']),  axis=1, result_type='expand')
    print("add smiles_map embedding finished!", flush=True)
    smiles_map.to_pickle("data/output/smiles_map.pkl")
    print("save smiles_map finished!", flush=True)
