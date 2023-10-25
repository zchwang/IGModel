import torch
from rdkit import Chem
import os
import numpy as np
import shutil
from utils import obabel

class Ligand():
    def __init__(self, mol):
        self.mol = mol
        self.lig_defined_eles = ["C", "N", "O", "P", "S", "Hal", "DU"]
        self.ref_lig_center = torch.tensor([])  # center of the reference ligand
        self.pose_ha_xyz = torch.tensor([])

    def load_mol(self, lig_content):
        self.mol = Chem.MolFromMol2Block(lig_content)
        return self

    def get_ele(self, atomic_num):

        idx_ele_dict = {1: "H", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 12: "Mg", 14: "Si", 15: "P",
                        16: "S", 17: "Cl", 23: "V", 26: "Fe", 27: "Co", 29: "Cu", 30: "Zn", 33: "As", 34: "Se",
                        35: "Br", 44: "Ru", 45: "Rh", 51: "Sb", 53: "I", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt"}
        try:
            return idx_ele_dict[atomic_num]
        except:
            return "DU"

    def get_atom_types(self, ele):

        Hal = ["F", "Cl", "Br", "I"]
        try:
            if ele in Hal:
                _type = "Hal"
            elif ele in self.lig_defined_eles:
                _type = ele
            else:
                _type = "DU"
        except:
            _type = "DU"

        return _type

    def get_eletype(self, ele):

        hal = ["F", "Cl", "Br", "I"]
        if ele in self.lig_defined_eles:
            return ele
        elif ele in hal:
            return "Hal"
        else:
            return "DU"