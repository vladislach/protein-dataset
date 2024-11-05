from constants import amino_acids, atom_order, chi_atoms, chi_idxs

from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

from esm.sdk.api import ESMProtein, SamplingConfig

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import one_hot

import numpy as np
import torch
import os


def get_esm_embs(data_dir: str) -> dict:
    """Get ESM3 embeddings for proteins in the given directory."""

    names = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.pdb')]
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdb')]

    parser = PDBParser()
    esm_embs = {}

    for name, path in zip(names, paths):
        structure = parser.get_structure(name, path)
        chain_embs_list = []

        for chain in structure.get_chains():
            seq = ''
            for residue in chain:
                # if not is_aa(residue): continue
                seq += seq1(residue.resname)

            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            output = client.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            chain_embs_list.append(output.per_residue_embedding[1:-1, :].cpu())

        emb = torch.cat(chain_embs_list, dim=0)
        esm_embs[name] = emb

    return esm_embs

def get_dihedral(A, B, C, D) -> float:
    """Calculate the dihedral angle between points A, B, C, and D in the range [0, 360] using arctan2."""
    AB = B - A
    BC = C - B
    CD = D - C

    N1 = np.cross(AB, BC)
    N2 = np.cross(BC, CD)

    N1 /= np.linalg.norm(N1)
    N2 /= np.linalg.norm(N2)

    x = np.dot(N1, N2)
    y = np.dot(np.cross(N1, N2), BC / np.linalg.norm(BC))
    angle = np.degrees(np.arctan2(y, x))

    if angle < 0:
        angle += 360

    return angle

def get_res_coords(res):
    """Get the one-letter code and coordinates of atoms in a residue."""
    res_code = seq1(res.get_resname())
    coords = [None] * 14    # 14 atoms in the longest amino acid
    for atom in res:
        if atom.get_name() in atom_order[res_code]:
            coords[atom_order[res_code].index(atom.get_name())] = atom.get_coord()
    return res_code, coords

def get_chi_angles(res_code: str, coords: list) -> torch.Tensor:
    """Calculate the chi angles for a residue given its one-letter code and coordinates."""
    chi_angles = [0.0] * 4   # 4 chi angles for the longest amino acid
    if res_code in chi_atoms:
        for i, idxs in chi_idxs[res_code].items():
            chi_coords = [coords[idx] for idx in idxs]
            chi_angles[i-1] = get_dihedral(*chi_coords)
    return torch.tensor(chi_angles)

def get_c_alpha_pos(coords: list) -> torch.Tensor:
    """Get the position of the alpha carbon of a residue given its coordinates."""
    return torch.tensor(coords[1])

def get_n_rel_pos(coords: list) -> torch.Tensor:
    """Get the relative position of the backbone nitrogen of a residue given its coordinates."""
    return torch.tensor(coords[0] - coords[1])

def get_c_rel_pos(coords: list) -> torch.Tensor:
    """Get the relative position of the backbone carbon of a residue given its coordinates."""
    return torch.tensor(coords[2] - coords[1])


class ProteinDataset(InMemoryDataset):
    '''Custom protein dataset with ESM3 embeddings, one-hot encoded amino acids, chi angles, and relative positions.'''

    url = 'https://github.com/vladislach/protein-dataset/raw/main/proteins.zip'

    def __init__(self, root, esm_embs_path, transform=None):
        self.esm_embs = torch.load(esm_embs_path)
        self.pdb_ids = list(self.esm_embs.keys())
        super().__init__(root, transform)
        self.load(self.processed_paths[0])
    
    def raw_file_names(self):
        return [f"{pdb_id}.pdb" for pdb_id in self.pdb_ids]
    
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
    
    def process(self):
        data_list = []
        parser = PDBParser()

        for pdb_id, path in zip(self.pdb_ids, self.raw_paths):
            structure = parser.get_structure(pdb_id, path)
            seq = ''
            chi_angles, c_alpha_pos, n_rel_pos, c_rel_pos = [], [], [], []

            for chain in structure.get_chains():
                for res in chain:
                    if not is_aa(res): continue
                    res_code, coords = self.get_res_coords(res)
                    chi_angles.append(self.get_chi_angles(res_code, coords) / 360)
                    c_alpha_pos.append(self.get_c_alpha_pos(coords))
                    n_rel_pos.append(self.get_n_rel_pos(coords))
                    c_rel_pos.append(self.get_c_rel_pos(coords))
                    seq += res_code
            
            esm_emb = self.esm_embs[pdb_id]
            res_one_hot = one_hot(torch.tensor([amino_acids.index(aa) for aa in seq]), len(amino_acids), dtype=torch.float)
            chi_angles = torch.stack(chi_angles)
            c_alpha_pos = torch.stack(c_alpha_pos)
            n_rel_pos = torch.stack(n_rel_pos)
            c_rel_pos = torch.stack(c_rel_pos)
            data = Data(
                name=pdb_id,
                x=torch.cat([res_one_hot, esm_emb], dim=-1),
                pos=c_alpha_pos,
                sidechain_feats=torch.cat([chi_angles, n_rel_pos, c_rel_pos], dim=-1)
            )
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
    
    def get_res_coords(self, res):
        """Get the coordinates of the atoms of a given residue."""
        res_code = seq1(res.get_resname())
        coords = [None] * 14
        for atom in res:
            if atom.get_name() in atom_order[res_code]:
                coords[atom_order[res_code].index(atom.get_name())] = atom.get_coord()
        return res_code, coords
    
    def get_dihedral(self, A, B, C, D) -> float:
        """Calculate the dihedral angle between points A, B, C, and D in the range [0, 360] using arctan2."""
        AB = B - A
        BC = C - B
        CD = D - C

        N1 = np.cross(AB, BC)
        N2 = np.cross(BC, CD)

        N1 /= np.linalg.norm(N1)
        N2 /= np.linalg.norm(N2)

        x = np.dot(N1, N2)
        y = np.dot(np.cross(N1, N2), BC / np.linalg.norm(BC))
        angle = np.degrees(np.arctan2(y, x))

        if angle < 0:
            angle += 360
        return angle
    
    def get_chi_angles(self, res_code: str, coords: list) -> torch.Tensor:
        """Calculate the chi angles for a residue given its one-letter code and coordinates."""
        chi_angles = [0.0] * 4
        if res_code in chi_atoms:
            for i, idxs in chi_idxs[res_code].items():
                chi_coords = [coords[idx] for idx in idxs]
                chi_angles[i-1] = get_dihedral(*chi_coords)
        return torch.tensor(chi_angles)
    
    def get_c_alpha_pos(self, coords: list) -> torch.Tensor:
        """Get the position of the alpha carbon of a residue given its coordinates."""
        return torch.tensor(coords[1])
    
    def get_n_rel_pos(self, coords: list) -> torch.Tensor:
        """Get the relative position of the backbone nitrogen of a residue given its coordinates."""
        return torch.tensor(coords[0] - coords[1])
    
    def get_c_rel_pos(self, coords: list) -> torch.Tensor:
        """Get the relative position of the backbone carbon of a residue given its coordinates."""
        return torch.tensor(coords[2] - coords[1])
