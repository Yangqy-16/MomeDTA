# Instructions on Data

This is the data processing pipeline of our method. Please switch to `preprocess/` and follow the instructions. Note that all the output should be in a separate `data/` folder.

## Data Preprocessing

This part imitates but is different from [LLMDTA](https://github.com/Chris-Tang6/LLMDTA).

Given any DTA dataset, the raw data in the beginning at least contains:

- `dataset.csv`\
  [`drug SMILES`, `protein sequence`, `affinity`]

  where `drug SMILES` are isomorphic form by default.

Then, you can refer to `step1.py` to process the data. After that, we expect all the data should have the following format:

- `drugs.csv`\
  [`drug_id`, `iso_smiles`]

- `prots.csv`\
  [`prot_id`, `prot_seq`, `pdb`]

  where `pdb` is the path to the PDB file (predicted by [AlphaFold2](https://github.com/google-deepmind/alphafold));

- `pairs.csv`\
  [`index`, `drug_id`, `prot_id`, `affinity`]

where `id`'s are all in the form of md5.

Note that some datasets may contain indices of drugs or proteins in the beginning. You can directly use these indices in this case.

## Multi-modal Input Generation

Run `step2.py`. After that, we expect all the data should have the following format:

- `drugs.csv`\
  [`drug_id`, `iso_smiles`, `selfies`] 

- `prots.csv`\
  [`prot_id`, `prot_seq`, `pdb`, `sa_seq`]

- `pairs.csv`\
  [`index`, `drug_id`, `prot_id`, `affinity`] 
  
  where `index` is just the index of the original dataframe.

## Running Pretrained Models

Next, please run:

1. `selformer_rep.py` to generate [SELFormer](https://github.com/hubiodatalab/selformer) embeddings;
2. `unimol_conf.py`and `unimol_infer.py` sequentially to generate [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) embeddings;
3. `esm2_rep.py` to generate [ESM-2](https://github.com/facebookresearch/esm) embeddings;
4. `saprot_rep.py` to generate [SaProt](https://github.com/westlake-repl/SaProt) embeddings.

After all of the above, you will get:

- `embed/`
  - `selformer/`
    - <drug_id>.pt
    - ...
  - `unimol/`
    - <drug_id>.pt
    - ...
  - `esm2/`
    - <prot_id>.pt
    - ...
  - `saprot/`
    - <prot_id>.pt
    - ...
- `token/` (won't be used in our method)
  - `unimol/`
    - <drug_id>.pt
    - ...

## Train-Val-Test Split

To split the dataset into different sets, please first run `cluster.py` to cluster the drugs and proteins.

Then, run `split.py` to split the data into different sets, in the form of:

- `splits/`
  - `warm/`
    - `fold_0_train.csv`\
      [`index`]
    - `fold_0_val.csv`\
      [`index`]
    - `fold_0_test.csv`\
      [`index`]
    - `fold_1_train.csv`\
      ...
    - ...
    - `fold_4_test.csv`\
      ...
  - `novel_drug/`
    - ...
  - `novel_prot/`
    - ...
  - `novel_pair/`
    - ...

## Generate Drug and Protein Graph

Run `drug_graph.py` and `prot_graph.py`.

## Final Directory Tree

In the end, we expect the directory tree be like:

- `data/`
  - `<dataset1>/`
    - `drugs.csv`
    - `prots.csv`
    - `pairs.csv`
    - `embed/`
      - ... (see above)
    - `splits/`
      - ... (see above)
    - `drug_graphs.pkl`
    - `prot_graphs.pkl`
  - `<dataset2>/`
    - ...
  - ...

You can also modify these and change the corresponding paths in our code.
