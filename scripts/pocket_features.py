import torch
import numpy as np
import dgl
import os
import shutil
from itertools import permutations
from utils import obabel, rec_defined_residues, generate_1d_dist, generate_2d_dist, \
    sidechain_dihedral_idx_dict, onek_encoding_unk

class PocketFeatures():
    def __init__(self,
                 rec: str = None,
                 pock_center: torch.tensor = None,
                 cutoff: float = 10.):

        self.rec = rec
        self.pock_center = pock_center
        self.cutoff = cutoff

        self.pock_res_indices = rec.pock_res_indices
        self.pock_res_dict = rec.pock_res_dict


    def cal_dihedral_angle(self, p0, p1, p2, p3):
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        # Calculate the normal vectors of the two planes defined by the points
        n1 = torch.cross(b0, b1)
        n2 = torch.cross(b1, b2)

        # Calculate the angle between the two normal vectors
        cos_angle = torch.dot(n1, n2) / (torch.norm(n1) * torch.norm(n2))
        dihedral_angle = torch.acos(cos_angle)
        dihedral_angle = torch.where(torch.isnan(dihedral_angle), torch.tensor(0.0), dihedral_angle)      

        # Determine the sign of the dihedral angle
        sign = torch.sign(torch.dot(n1, b2))

        # Apply the sign to the dihedral angle
        dihedral_angle *= sign

        dihedral_feat = torch.sin(dihedral_angle / 2)

        return dihedral_feat

    def generate_dihedral_feats(self, res, pdb_to_xyz_dict, last_c_xyz, next_n_xyz):
        ca_xyz = pdb_to_xyz_dict["CA"]
        n_xyz = pdb_to_xyz_dict["N"]
        c_xyz = pdb_to_xyz_dict["C"]
        o_xyz = pdb_to_xyz_dict["O"]

        init_sidechain_dihedral_feats = -2 * torch.ones(7)
        # if not res in ["ALA", "GLY"]:
        try:
            phi = self.cal_dihedral_angle(last_c_xyz, n_xyz, ca_xyz, c_xyz)
            init_sidechain_dihedral_feats[0] = phi
        except:
            print("Warnning: phi feats calculate failed ...")

        try:
            psi = self.cal_dihedral_angle(n_xyz, ca_xyz, c_xyz, next_n_xyz)
            init_sidechain_dihedral_feats[1] = psi
        except:
            print("Warnning: psi feats calculate failed ...")

        if not res in ["ALA", "GLY"]:
            sidechain_dihedral_dict = sidechain_dihedral_idx_dict[res]
            chi_keys = list(sidechain_dihedral_dict.keys())

            # chi-1
            if "chi-1" in chi_keys:
                chi1_index = sidechain_dihedral_dict["chi-1"]
                try:
                    chi1_xyz = [pdb_to_xyz_dict[i] for i in chi1_index]
                    chi_feat = self.cal_dihedral_angle(chi1_xyz[0], chi1_xyz[1], chi1_xyz[2], chi1_xyz[3])
                    init_sidechain_dihedral_feats[2] = chi_feat
                except:
                    print("Warnning: Chi-1 feats calculate failed ...")

            # chi-2
            if "chi-2" in chi_keys:
                chi2_index = sidechain_dihedral_dict["chi-2"]
                try:
                    chi2_xyz = [pdb_to_xyz_dict[i] for i in chi2_index]
                    chi_feat = self.cal_dihedral_angle(chi2_xyz[0], chi2_xyz[1], chi2_xyz[2], chi2_xyz[3])
                    init_sidechain_dihedral_feats[3] = chi_feat
                except:
                    print("Warnning: Chi-2 feats calculate failed ...")

            # chi-3
            if "chi-3" in chi_keys:
                chi3_index = sidechain_dihedral_dict["chi-3"]
                try:
                    chi3_xyz = [pdb_to_xyz_dict[i] for i in chi3_index]
                    chi_feat = self.cal_dihedral_angle(chi3_xyz[0], chi3_xyz[1], chi3_xyz[2], chi3_xyz[3])
                    init_sidechain_dihedral_feats[4] = chi_feat
                except:
                    print("Warnning: Chi-3 feats calculate failed ...")

            # chi-4
            if "chi-4" in chi_keys:
                chi4_index = sidechain_dihedral_dict["chi-4"]
                try:
                    chi4_xyz = [pdb_to_xyz_dict[i] for i in chi4_index]
                    chi_feat = self.cal_dihedral_angle(chi4_xyz[0], chi4_xyz[1], chi4_xyz[2], chi4_xyz[3])
                    init_sidechain_dihedral_feats[5] = chi_feat
                except:
                    print("Warnning: Chi-4 feats calculate failed ...")

            # chi-5
            if "chi-5" in chi_keys:
                chi5_index = sidechain_dihedral_dict["chi-5"]
                try:
                    chi5_xyz = [pdb_to_xyz_dict[i] for i in chi5_index]
                    chi_feat = self.cal_dihedral_angle(chi5_xyz[0], chi5_xyz[1], chi5_xyz[2], chi5_xyz[3])
                    init_sidechain_dihedral_feats[6] = chi_feat
                except:
                    print("Warnning: Chi-5 feats calculate failed ...")

        return init_sidechain_dihedral_feats

    def cal_node_feats(self):

        node_feats = []
        for i in self.pock_res_indices:
            pock_res_top_dict = self.pock_res_dict[i]
            res_name = pock_res_top_dict["res"]
            pdb_to_xyz_dict = pock_res_top_dict["pdb_to_xyz"]
            res_xyz_tensor = torch.cat(list(pdb_to_xyz_dict.values()), axis=0).reshape(-1, 3)

            res_idx = rec_defined_residues.index(res_name)

            res_feat = onek_encoding_unk(res_name, rec_defined_residues)  # 21

            ca_xyz = pdb_to_xyz_dict["CA"]
            n_xyz = pdb_to_xyz_dict["N"]
            c_xyz = pdb_to_xyz_dict["C"]
            o_xyz = pdb_to_xyz_dict["O"]

            last_c_xyz = pock_res_top_dict["last_C_xyz"]
            next_n_xyz = pock_res_top_dict["next_N_xyz"]

            # the max distance between any atoms in this residues
            _max_dist_feat = torch.tensor([generate_2d_dist(res_xyz_tensor, res_xyz_tensor).max()])  # 1

            # the max distance between ca and other atoms
            _max_ca_dist_feat = torch.tensor([generate_1d_dist(res_xyz_tensor, ca_xyz).max()])  # 1

            # the distance between N and O
            dist_n_o_feat = generate_1d_dist(n_xyz, o_xyz)  # 1
            # print(i, n_xyz, o_xyz)

            # this distance between CA and ligand center
            ca_center_dist_feat = generate_1d_dist(ca_xyz, self.pock_center)  # 1

            # the max/min distance between ligand center and any atoms in this residues
            res_center_dist = generate_1d_dist(res_xyz_tensor, self.pock_center)
            max_res_center_feat = torch.tensor([res_center_dist.max()])  # 1
            min_res_center_feat = torch.tensor([res_center_dist.min()])  # 1

            # dihedral 7
            dihedral_feats = self.generate_dihedral_feats(res_name, pdb_to_xyz_dict, last_c_xyz, next_n_xyz)  # 7

            node_feats.append(torch.cat(
                [res_feat, _max_dist_feat, _max_ca_dist_feat, dist_n_o_feat, ca_center_dist_feat, max_res_center_feat,
                 min_res_center_feat, dihedral_feats]).reshape(1, -1))

        node_feats = torch.cat(node_feats, axis=0)

        return node_feats

    def cal_edge_feats(self):

        u_edge_list = []
        v_edge_list = []
        edge_feats_list = []

        pock_res_idx_dict = dict(zip(self.pock_res_indices, [x for x in range(len(self.pock_res_indices))]))
        for res1, res2 in permutations(self.pock_res_indices, 2):

            idx_1 = pock_res_idx_dict[res1]
            idx_2 = pock_res_idx_dict[res2]

            res1_top_dict = self.pock_res_dict[res1]
            res2_top_dict = self.pock_res_dict[res2]

            res1_pdb_to_xyz = res1_top_dict["pdb_to_xyz"]
            res2_pdb_to_xyz = res2_top_dict["pdb_to_xyz"]

            res1_xyz_tensor = torch.cat(list(res1_pdb_to_xyz.values()), axis=0).reshape(-1, 3)
            res2_xyz_tensor = torch.cat(list(res2_pdb_to_xyz.values()), axis=0).reshape(-1, 3)

            res1_ca = res1_top_dict["CA_xyz"]
            res2_ca = res2_top_dict["CA_xyz"]
            ca_dist = generate_1d_dist(res1_ca, res2_ca)

            dist = generate_2d_dist(res1_xyz_tensor, res2_xyz_tensor)
            if dist.min() <= self.cutoff:
                res1_center = torch.mean(res1_xyz_tensor, axis=0)
                res2_center = torch.mean(res2_xyz_tensor, axis=0)

                res1_o = res1_pdb_to_xyz["O"]
                res1_n = res1_pdb_to_xyz["N"]
                res1_c = res1_pdb_to_xyz["C"]

                res2_o = res2_pdb_to_xyz["O"]
                res2_n = res2_pdb_to_xyz["N"]
                res2_c = res2_pdb_to_xyz["C"]

                o_dist = generate_1d_dist(res1_o, res2_o)
                n_dist = generate_1d_dist(res1_n, res2_n)
                c_dist = generate_1d_dist(res1_c, res2_c)
                center_dist = generate_1d_dist(res1_center, res2_center)
                min_dist = dist.min().reshape(1)
                max_dist = dist.max().reshape(1)

                edge_feat = torch.cat([ca_dist, o_dist, n_dist, c_dist, center_dist, min_dist, max_dist],
                                      axis=0).reshape(1, -1)
                u_edge_list += [idx_1]
                v_edge_list += [idx_2]
                edge_feats_list += [edge_feat] 

        edge_feats = torch.cat(edge_feats_list, axis=0)

        return u_edge_list, v_edge_list, edge_feats

    def pock_to_graph(self):
        node_feats = self.cal_node_feats()
        u_edge_list, v_edge_list, edge_feats = self.cal_edge_feats()

        g = dgl.DGLGraph()
        g.add_nodes(node_feats.size()[0])
        g.ndata["feats"] = node_feats
        g.add_edges(u_edge_list, v_edge_list)
        g.edata["feats"] = edge_feats

        return g
