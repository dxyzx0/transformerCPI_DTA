# transformerCPI_DTA
Refine https://github.com/lifanchen-simm/transformerCPI/ for large dataset and multiple GPU training

### Install Requirements
`conda env create -f py36_tCPI.yml`

### Training and testing
First run 
`sh script/generate_map.sh` to generate `protein_map.pkl` and `smiles_map.pkl`, which is the mapping from `smiles` to `smiles_feature`, and `protein_seq` to `protein_seq_feature`
Then run `sh script/main.sh` to start training.
If you want to stop the training process, run `sh script/stop.sh`

### Comparison to the orginal repo https://github.com/lifanchen-simm/transformerCPI
Advantages:
1. You can use `torch.nn.DataParallel` to accelerate your training process.
2. For large scale dataset, Using `DTADataset` in `DataUtil.py` along with `torch.nn.DataLoader` can accelerate your data loading process and tremendously reduce the memory usage.
3. I change the code into solving **regression** problem instead of **classification** problem in the original paper.

Thanks for the brilliant work of authors in this paper https://doi.org/10.1093/bioinformatics/btaa524
