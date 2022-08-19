#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


def morgan_binary_features_generator(mol: Union[str, Chem.Mol],
                                     radius: int,
                                     num_bits: int) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


def morgan_counts_features_generator(mol: Union[str, Chem.Mol],
                                     radius: int,
                                     num_bits: int) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


def rdkit_2d_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
    """
    Generates RDKit 2D features_mol for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features_mol.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdDescriptors.RDKit2D()
    features = generator.process(smiles)[1:]

    return features


def rdkit_2d_normalized_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
    """
    Generates RDKit 2D normalized features_mol for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D normalized features_mol.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]

    return features
