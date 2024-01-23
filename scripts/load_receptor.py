import os
import shutil
import torch
import numpy as np
from random import randint
from utils import generate_3d_dist, correct_residue_dict, ad4_to_ele_dict, \
    rec_defined_residues, ad4_to_ele_dict, obabel, HETATM_list


class ReceptorFile():
    def __init__(self,
                 rec_fpath: str = None,
                 ref_lig_fpath: str=None,
                 temp_dpath: str="temp_files",
                 clip_cutoff: float=8.0,
                 pocket_cutoff: float=8.0):

        self.rec_fpath = rec_fpath
        self.ref_lig_fpath = ref_lig_fpath
        self.temp_dpath = temp_dpath
        os.makedirs(self.temp_dpath, exist_ok=True)

        self.clip_cutoff = clip_cutoff
        self.pocket_cutoff = pocket_cutoff

        self.all_residues = []  #
        self.all_atom_residues = []
        self.all_pdb_types = []
        self.all_elements = []
        self.all_res_ha_xyz = []
        self.all_atom_xyz_tensor = torch.tensor([])
        self.all_resid_atom_indices = []
        self.all_res_pdb_xyz_dict = {}
        self.all_resid_xyz_tensor = torch.tensor([])  # [N, MAX_RES_HA, 3]

        self.clip_ha_indices = []
        self.pock_ha_indices = []
        self.pock_ha_lines = []
        self.pock_center = torch.tensor([])
        self.ref_lig_xyz = torch.tensor([])

    def parse_ref_lig(self):
        basename, format_ = os.path.basename(self.ref_lig_fpath).split(".")
        rand_str = str(randint(1, 10000000))
        pdb_file = f"{self.temp_dpath}/{basename}-{rand_str}.pdb"
        if format_ != "pdb":
            obabel(self.ref_lig_fpath, pdb_file)
        else:
            shutil.copyfile(self.ref_lig_fpath, pdb_file)

        with open(pdb_file) as f:
            lines = [x.strip() for x in f.readlines() if x.startswith("ATOM") or x.startswith("HETATM")]

        ha_xyz = []
        for line in lines:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            xyz = torch.from_numpy(np.array([[x, y, z]]))
            ele = line.split()[-1]
            if ele != "H":
                ha_xyz.append(xyz)
        self.ref_lig_xyz = torch.cat(ha_xyz, axis=0).to(torch.float32)
        self.pock_center = torch.mean(self.ref_lig_xyz, axis=0)

        return self
    
    def parse_ref_coords(self, ref_lig_xyz):
        self.ref_lig_xyz = ref_lig_xyz
        self.pock_center = torch.mean(self.ref_lig_xyz, axis=0)
        return self

    def load_rec(self):
        with open(self.rec_fpath) as f:
            self.lines = [line.strip() for line in f.readlines() if line[:4] == "ATOM" or line[:6] == "HETATM"]

        all_atom_xyz_list = []
        resid_symbol_pool = []
        temp_res_xyz = []
        temp_indices_list = []
        temp_pdb_types_list = []
        MAX_HA_NUM = 0
        res_idx = -1
        atom_idx = -1
        for num, line in enumerate(self.lines):
            resid_symbol = line[17:27].strip()
            res = line[17:20].strip()
            if res in HETATM_list or res[:2] in HETATM_list:
                continue
            if not res in rec_defined_residues:
                try:
                    res = correct_residue_dict[res]
                except KeyError:
                    res = "OTH"
            ele = line.split()[-1]
            if not ele in ["H", "C", "N", "O", "S"]:
                ele = "DU"
            
            atom_idx += 1
            pdb_type = line[11:17].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_xyz = np.c_[x, y, z]

            self.all_elements.append(ele)
            self.all_pdb_types.append(pdb_type)
            self.all_atom_residues.append(res)
            all_atom_xyz_list.append(atom_xyz)

            if num == len(self.lines) - 1:
                if ele != "H":
                    temp_res_xyz.append(atom_xyz)
                    temp_pdb_types_list.append(pdb_type)
                    temp_indices_list.append(atom_idx)

                res_idx += 1
                temp_xyz = torch.from_numpy(np.concatenate(temp_res_xyz, axis=0)).to(torch.float32)
                self.all_res_ha_xyz.append(temp_xyz)
                self.all_resid_atom_indices.append(temp_indices_list)
                if len(temp_xyz) > MAX_HA_NUM:
                    MAX_HA_NUM = len(temp_xyz)

                pdb_to_xyz_dict = dict(zip(temp_pdb_types_list, temp_xyz))
                self.all_res_pdb_xyz_dict[res_idx] = pdb_to_xyz_dict
                continue

            if ele == "H":
                continue

            if num == 0:
                resid_symbol_pool.append(resid_symbol)
                self.all_residues.append(res)
                temp_res_xyz.append(atom_xyz)
                temp_pdb_types_list.append(pdb_type)
                temp_indices_list.append(atom_idx)

            else:
                if resid_symbol != resid_symbol_pool[-1]:
                    res_idx += 1
                    temp_xyz = torch.from_numpy(np.concatenate(temp_res_xyz, axis=0)).to(torch.float32)
                    self.all_res_ha_xyz.append(temp_xyz)
                    self.all_resid_atom_indices.append(temp_indices_list)
                    pdb_to_xyz_dict = dict(zip(temp_pdb_types_list, temp_xyz))
                    self.all_res_pdb_xyz_dict[res_idx] = pdb_to_xyz_dict

                    if len(temp_xyz) > MAX_HA_NUM:
                        MAX_HA_NUM = len(temp_xyz)

                    resid_symbol_pool.append(resid_symbol)
                    self.all_residues.append(res)
                    temp_res_xyz = [atom_xyz]
                    temp_pdb_types_list = [pdb_type]
                    temp_indices_list = [atom_idx]

                else:
                    temp_res_xyz.append(atom_xyz)
                    temp_pdb_types_list.append(pdb_type)
                    temp_indices_list.append(atom_idx)

        self.all_atom_xyz_tensor = torch.from_numpy(np.concatenate(all_atom_xyz_list, axis=0)).to(torch.float32)

        final_all_resid_xyz_list = []
        for xyz in self.all_res_ha_xyz:
            if xyz.shape[0] < MAX_HA_NUM:
                temp_xyz = np.concatenate([xyz, np.ones((MAX_HA_NUM - xyz.shape[0], 3)) * 9999.], axis=0)
            else:
                temp_xyz = xyz
            final_all_resid_xyz_list.append(temp_xyz.reshape(1, -1, 3))

        self.all_resid_xyz_tensor = torch.from_numpy(np.concatenate(final_all_resid_xyz_list, axis=0)).to(torch.float32)

        return self

    def clip_rec(self):
        self.load_rec()
        if self.ref_lig_fpath != None:
            self.parse_ref_lig()

        N_res = len(self.all_resid_xyz_tensor)
        pose_xyz = self.ref_lig_xyz.unsqueeze(0).repeat(N_res, 1, 1)

        self.dist_mtx = generate_3d_dist(self.all_resid_xyz_tensor, pose_xyz)

        clip_res_indices = sorted(set(torch.where(self.dist_mtx <= self.clip_cutoff)[0].tolist()))
        clip_ha_indices = []
        clip_ha_ca_xyz = []
        for i in clip_res_indices:
            clip_ha_indices += self.all_resid_atom_indices[i]
            N = len(self.all_resid_atom_indices[i])
            try:
                ca_xyz = self.all_res_pdb_xyz_dict[i]["CA"].reshape(1, 3)
            except KeyError:
                ca_xyz = list(self.all_res_pdb_xyz_dict[i].values())[0].reshape(1, 3) 
            clip_ha_ca_xyz += [ca_xyz] * N

        self.clip_rec_ca_xyz_tensor = torch.cat(clip_ha_ca_xyz, axis=0)

        self.clip_ha_indices = sorted(clip_ha_indices)
        self.clip_rec_ele = []
        self.clip_rec_ha_residues = []
        self.clip_rec_ha_pdb_types = []
        clip_rec_ha_xyz_list = []
        for i in self.clip_ha_indices:
            self.clip_rec_ele.append(self.all_elements[i])
            self.clip_rec_ha_residues.append(self.all_atom_residues[i])
            self.clip_rec_ha_pdb_types.append(self.all_pdb_types[i])
            clip_rec_ha_xyz_list.append(self.all_atom_xyz_tensor[i].reshape(1, 3))

        self.clip_rec_ha_xyz_tensor = torch.cat(clip_rec_ha_xyz_list, axis=0)

        return self

    def define_pocket(self):
        self.pock_res_indices = sorted(set(torch.where(self.dist_mtx <= self.pocket_cutoff)[0].tolist()))
        self.pock_res_dict = {}

        for i in self.pock_res_indices:
            res = self.all_residues[i]
            ha_indices = self.all_resid_atom_indices[i]
            self.pock_ha_indices += ha_indices
            pdb_types = [self.all_pdb_types[x] for x in ha_indices]
            ha_xyz = self.all_res_ha_xyz[i]
            pdb_to_xyz_dict = dict(zip(pdb_types, ha_xyz))
            try:
                ca_xyz = pdb_to_xyz_dict["CA"]
            except KeyError:
                ca_xyz = ha_xyz[0]

            try:
                last_C_xyz = self.all_res_pdb_xyz_dict[i - 1]["C"]
                last_C_N_dist = torch.sqrt(torch.sum(torch.square(last_C_xyz - pdb_to_xyz_dict["N"])))
                if last_C_N_dist > 1.5:
                    last_C_xyz = None
            except KeyError:
                last_C_xyz = None

            try:
                next_N_xyz = self.all_res_pdb_xyz_dict[i + 1]["N"]
                C_next_N_dist = torch.sqrt(torch.sum(torch.square(pdb_to_xyz_dict["C"] - next_N_xyz)))
                if C_next_N_dist > 1.5:
                    next_N_xyz = None
            except KeyError:
                next_N_xyz = None

            self.pock_res_dict[i] = {"res": res, "ha_indices": ha_indices, "pdb_to_xyz": pdb_to_xyz_dict,
                                     "last_C_xyz": last_C_xyz, "next_N_xyz": next_N_xyz, "CA_xyz": ca_xyz}
        return self