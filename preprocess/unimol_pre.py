""" Fetched and modified from https://github.com/deepmodeling/Uni-Mol """

# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="[CLS]",
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]
    
    def special_index(self):
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.index(self.unk_word)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word, self.indices[word])
                    )
                else:
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

dict_name = "mol.dict.txt"
dictionary = Dictionary.load(os.path.join('/data/qingyuyang/dta_ours/weights/unimol', dict_name))
dictionary.add_symbol("[MASK]", is_special=True)

def inner_coords(atoms, coordinates, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates. 

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.
    
    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.
    
    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }

def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True, return_mol=False):
    '''
    This function is responsible for converting a SMILES (Simplified Molecular Input Line Entry System) string into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

    :param smi: (str) The SMILES representation of the molecule.
    :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
    :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
    :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

    :return: A tuple containing the list of atom symbols and their corresponding 3D coordinates.
    :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.
    '''
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))

    if return_mol:
        return mol # for unimolv2
    
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

"""
Example:

atoms
['N', 'N', 'N', 'N', 'C', 'N', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'N']

coordinates - np.array((n, 3))
[[ 3.1647997   1.2588629   0.93152195]
 [ 4.500329    1.2227619   1.0367055 ]
 [ 4.853871    0.05704636  0.51067823]
 [ 3.7527063  -0.6472704   0.07804436]
 [ 2.7099469   0.11008963  0.34572718]
 [ 1.3867953  -0.14462778  0.12441161]
 [ 0.34861386  0.6994654   0.48074982]
 [ 0.4142686   1.7908412   1.0241714 ]
 [-0.90605515 -0.01360526  0.10465826]
 [-2.2524626   0.5354054   0.40564668]
 [-2.6689835   1.5969392  -0.601047  ]
 [-0.4962347  -1.1878055  -0.41411576]
 [-1.3402914  -2.3269262  -0.8769708 ]
...
 [ 4.261556   -1.1653419   0.7010728 ]
 [ 5.3304396  -0.35270053  0.67986095]
 [ 5.0912013   0.92672396  0.36243713]
 [ 3.7812142   0.98226905  0.1628057 ]]
"""

def smi2coords(content):
    id, smile = content
    try:
        return id, inner_smi2coords(smile)
    except:
        print("Failed smiles: {}".format(id))
        return None

def write_results(smiles_list, job_name, outpath, nthreads=8):
    with Pool(nthreads) as pool:
        i = 0
        for id, inner_output in tqdm(pool.imap(smi2coords, smiles_list), dynamic_ncols=True):
            if inner_output is not None:
                atoms, coordinates = inner_output
                final_output = coords2unimol(atoms, coordinates, dictionary)
                torch.save(final_output, f"{outpath}/{id}.pt")
                i += 1
        print(f'{job_name} process {i} lines')