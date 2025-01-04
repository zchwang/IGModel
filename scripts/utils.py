import os
import json
import numpy as np
from openbabel import openbabel as ob
from spyrmsd import io, rmsd
import torch

_current_dpath = os.path.dirname(os.path.abspath(__file__))
with open(_current_dpath + "/rec_types_mapping.json") as f:
    rec_types_mapping = json.load(f)
all_defined_rec_ha_types = sorted(set(rec_types_mapping.values())) + ["OTH-DU"]
all_rec_ha_keys = sorted(set(rec_types_mapping.keys()))

def get_defined_type(res_name, pdb_type):
    if pdb_type == "OXT":
        _type = res_name + "-MO"
    else:
        res_atom_type = res_name + "-" + pdb_type
        if res_atom_type in all_rec_ha_keys:
            _type = rec_types_mapping[res_atom_type]
        else:
            print("DU:", res_name, pdb_type)
            _type = "OTH-DU"

    return _type

rec_defined_elements = ["C", "N", "O", "S", "DU"]
rec_defined_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                        'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', "OTH"]
HETATM_list = ["WA", "HEM", "NAD", "NAP", "UMP", "MG", "SAM", "ADP", "FAD", "CA", "ZN", "FMN", "CA", "NDP", "TPO", "LLP"]

ad4_to_ele_dict = {
    "C": "C",
    "A": "C",
    "N": "N",
    "NA": "N",
    "OA": "O",
    "S": "S",
    "SA": "S",
    "Se": "S",
    "P": "P",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
}

correct_residue_dict = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HIZ": "HIS",
    "HIY": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "CYT": "CYS",
    "MEU": "MET",
    "LEV": "LEU",
    "ASQ": "ASP",
    "ASH": "ASP",
    "DID": "ASP",
    "DIC": "ASP",
    "GLZ": "GLY",
    "GLV": "GLU",
    "GLH": "GLU",
    "GLM": "GLU",
    "ASZ": "ASN",
    "ASM": "ASN",
    "GLO": "GLN",
    "SEM": "SER",
    "TYM": "TYR",
    "ALB": "ALA"
}

aromatic_ring_types = [
    "PHE-CG",
    "PHE-CD1",
    "PHE-CD2",
    "PHE-CE1",
    "PHE-CE2",
    "PHE-CZ",
    "TRP-CG",
    "TRP-CD1",
    "TRP-CD2",
    "TRP-NE1",
    "TRP-CE2",
    "TRP-CE3",
    "TRP-CZ2",
    "TRP-CZ3",
    "TRP-CH2",
    "TYR-CG",
    "TYR-CD1",
    "TYR-CD2",
    "TYR-CE1",
    "TYR-CE2",
    "TYR-CZ"
]

positive_types = [
    "LYS-NZ",
    "ARG-NH1",
    "ARG-NH2",
    "HIS-ND1",
]

negative_types = [
    "ASP-OD1",
    "ASP-OD2",
    "GLU-OE1",
    "GLU-OE2"
]

# sidechain chi-(1, 2, 3, 4, 5)
sidechain_dihedral_idx_dict = {
    "VAL": {"chi-1": ["N", "CA", "CB", "CG1"]},
     "LEU": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD1"]},
     "ILE": {"chi-1": ["N", "CA", "CB", "CG1"], "chi-2": ["CA", "CB", "CG1", "CD1"]},
     "PHE": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD1"]},
     "PRO": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD"]},
     "SER": {"chi-1": ["N", "CA", "CB", "OG"]},
     "THR": {"chi-1": ["N", "CA", "CB", "OG1"]},
     "HIS": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "ND1"]},
     "TRP": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD1"]},
     "CYS": {"chi-1": ["N", "CA", "CB", "SG"]},
     "ASP": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "OD1"]},
     "GLU": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD"], "chi-3": ["CB", "CG", "CD", "OE1"]},
     "LYS": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD"], "chi-3": ["CB", "CG", "CD", "CE"],
            "chi-4": ["CG", "CD", "CE", "NZ"]},
     "TYR": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD1"]},
     "MET": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "SD"], "chi-3": ["CB", "CG", "SD", "CE"]},
     "ASN": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "OD1"]},
     "GLN": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD"], "chi-3": ["CB", "CG", "CD", "OE1"]},
     "ARG": {"chi-1": ["N", "CA", "CB", "CG"], "chi-2": ["CA", "CB", "CG", "CD"], "chi-3": ["CB", "CG", "CD", "NE"],
            "chi-4": ["CG", "CD", "NE", "CZ"], "chi-5": ["CD", "NE", "CZ", "NH1"]}
    }

def sdf_split(infile):
    contents = open(infile, 'r').read()
    mols = [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
    names = [x.split("\n")[0] for x in mols]
    return names, mols

def mol2_split(infile):
    contents = open(infile, 'r').read()
    mols = ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
    names = [x.split("\n")[1] for x in mols]
    return names, mols

def generate_1d_dist(coord1, coord2):
    return torch.sqrt(torch.sum(torch.square(coord1.reshape(-1, 3) - coord2), axis=1))

def generate_2d_dist(mtx_1, mtx_2):

    N, C = mtx_1.size()
    M, _ = mtx_2.size()
    dist = -2 * torch.matmul(mtx_1, mtx_2.permute(1, 0))
    dist += torch.sum(mtx_1 ** 2, -1).view(N, 1)
    dist += torch.sum(mtx_2 ** 2, -1).view(1, M)

    dist = (dist >= 0) * dist
    dist = torch.sqrt(dist)

    return dist

def generate_3d_dist(mtx_1, mtx_2):

    """
    Args:
        mtx_1, mtx_2: torch.tensor, shape [n, m, 3], where n is the number of mols, m is the number of atoms in the ligand.
    Returns:
        dist: torch.tensor, shape [n, m1, m2]
    """

    n, N, C = mtx_1.size()
    n, M, _ = mtx_2.size()
    dist = -2 * torch.matmul(mtx_1, mtx_2.permute(0, 2, 1))
    dist += torch.sum(mtx_1 ** 2, -1).view(-1, N, 1)
    dist += torch.sum(mtx_2 ** 2, -1).view(-1, 1, M)

    dist = (dist >= 0) * dist
    dist = torch.sqrt(dist)

    return dist

def generate_2d_pairwise_dist(mtx_1, mtx_2):

    d_square_sum = torch.sum(torch.square(mtx_1 - mtx_2), axis=1)
    d_square_sum = (d_square_sum >= 0) * d_square_sum + (d_square_sum < 0) * 0.
    dist = torch.sqrt(d_square_sum)

    return dist

def obabel(infile, outfile):
    basename = os.path.basename(infile).split(".")[0]
    _format = outfile.split(".")[-1]

    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(basename, _format)
    ligand = ob.OBMol()
    obConversion.ReadFile(ligand, infile)
    obConversion.WriteFile(ligand, outfile)

def cal_rmsd(ref_mol, test_mol):
    ref = io.loadmol(ref_mol)
    ref.strip()
    coords_ref = ref.coordinates
    anum_ref = ref.atomicnums
    adj_ref = ref.adjacency_matrix

    mol = io.loadmol(test_mol)
    mol.strip()

    coords_mol = mol.coordinates
    anum_mol = mol.atomicnums
    adj_mol = mol.adjacency_matrix

    sym_RMSD = rmsd.symmrmsd(coords_ref, coords_mol, anum_ref, anum_mol, adj_ref, adj_mol)
    hRMSD = rmsd.hrmsd(coords_ref, coords_mol, anum_ref, anum_mol, center=False)

    return sym_RMSD, hRMSD

def merge_graph(keys, graphs):
    """

    :param keys: the pdb code
    :param graphs: the graph of the protein or the ligand
    :return:
        the merged graph
    """
    final_graph = {}
    for k, v in zip(keys, graphs):
        final_graph[k] = v

    return final_graph


def cal_angle_between_vectors(u, v):
    """
    Args:
        u, v: input matrix [N, 3]

    Returns:
        radian: the [N, ]
    """

    u_norm = torch.linalg.norm(u, dim=1)
    v_norm = torch.linalg.norm(v, dim=1)

    res = torch.sum(u * v, axis=1) / (u_norm * v_norm)
    angle = torch.acos(res)

    angle = torch.where(torch.isnan(angle), torch.tensor(0.0), angle)

    return angle


def cal_dihedral_anlge(vec_1, vec_2, vec_3):

    """

    Args:
        vec_1, vec_2, vec_3: shape: [N, 3]

    Returns:
        dihedral anglel: shape [N, ]
    """

    n1, n2 = torch.cross(vec_1, vec_2, dim=1), torch.cross(vec_2, vec_3, dim=1)

    norm_1 = torch.linalg.norm(n1, dim=1)
    norm_2 = torch.linalg.norm(n2, dim=1)

    angle = torch.acos(torch.sum(n1 * n2, axis=1) / (norm_1 * norm_2))
    angle = torch.where(torch.isnan(angle), torch.tensor(0.0), angle)

    sign = torch.sign(torch.sum(n1 * vec_3, axis=1))
    angle *= sign

    return angle

def onek_encoding_unk(value, choices):
    """
    A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
         If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """

    #encoding = [0] * (len(choices) + 1)
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return torch.tensor(encoding)

def run_an_eval_epoch(model, data_loader, device="cuda"):
    model.eval()

    with torch.no_grad():
        pred_rmsd_list = []
        pred_pkd_list = []
        for step, data in enumerate(data_loader):
            idx, rec_g, cplx_g = data
            if device == "cuda":
                rec_g = rec_g.to("cuda:0")
                cplx_g = cplx_g.to("cuda:0")

            pred_hrmsd, pred_pkd, lig_embed, cplx_embed, W, _, _, _, _, _ = model(rec_g, cplx_g)
            pred_rmsd_list.append(pred_hrmsd.cpu().detach().numpy().ravel())
            pred_pkd_list.append(pred_pkd.cpu().detach().numpy().ravel())

        pred_rmsd = np.concatenate(pred_rmsd_list, axis=0)
        pred_pkd = np.concatenate(pred_pkd_list, axis=0)

    return pred_rmsd, pred_pkd
