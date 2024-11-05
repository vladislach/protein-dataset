# Protein Dataset in PyTorch Geometric

This repository contains a collection of code snippets and methods to create a custom A PyTorch Geometric dataset for protein sequences. It also includes methods for generating sequnce embeddings using the ESM3 model. For each sequence, the dataset contains the following:

* `x`: The ESM3 embeddings for each residue concatenated with one-hot encodings of the amino acid.
* `name`: The PDB ID of the protein.
* `pos`: Coordinates of the alpha carbon atom in each residue.
* `sidechain_feats`: Concatenation of the following features:
  * `chi_angles`: The chi angles for each residue normalized to the range [0, 1] by dividing by 360.
  * `n_rel_pos`: Position of the backbone nitrogen atom relative to the alpha carbon in each residue.
  * `c_rel_pos`: Position of the backbone carbon atom relative to the alpha carbon in each residue.

All fucntions and classes can be found in the `dataset.py` file. The `example.ipynb` notebook shows how to use them on a toy dataset.
