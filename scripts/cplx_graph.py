import torch
import dgl
from load_receptor import ReceptorFile
from ligand_features import LigandFeature
from utils import rec_defined_elements, rec_defined_residues, positive_types, negative_types, \
    aromatic_ring_types, generate_2d_dist, generate_3d_dist, generate_2d_pairwise_dist, \
    onek_encoding_unk, cal_angle_between_vectors, cal_dihedral_anlge, get_defined_type, \
    all_defined_rec_ha_types

class ComplexGraph():
    def __init__(self,
                 rec: str = ReceptorFile,
                 lig: str = LigandFeature,
                 rec_cutoff: float = 5.0,
                 inter_cutoff: float = 8.0):
        self.rec = rec
        self.lig = lig
        self.rec_cutoff = rec_cutoff
        self.inter_cutoff = inter_cutoff

        # rec after clipping
        self.clip_rec_ha_indices = rec.clip_ha_indices
        self.clip_rec_ele = rec.clip_rec_ele
        self.clip_rec_ha_residues = rec.clip_rec_ha_residues
        self.clip_rec_ha_pdb_types = rec.clip_rec_ha_pdb_types
        self.clip_rec_ha_xyz_tensor = rec.clip_rec_ha_xyz_tensor
        self.clip_rec_ca_xyz_tensor = rec.clip_rec_ca_xyz_tensor

        # lig
        self.lig_ha_ele = lig.pose_ha_eles
        self.pose_ha_xyz = lig.pose_ha_xyz
        self.pose_center = lig.pose_center

        # lig graph
        self.lig_node_feats = lig.lig_node_feats
        self.lig_u_edge_list = lig.u_lig_edge_list
        self.lig_v_edge_list = lig.v_lig_edge_list
        self.lig_edge_feats = lig.lig_edge_feats

        #
        self.poses_rec_nodes_dist = torch.tensor([])

    def get_rec_lig_dist(self):

        self.pose_rec_nodes_dist = generate_2d_dist(self.pose_ha_xyz, self.clip_rec_ha_xyz_tensor)

        return self

    def get_polar_aromatic_feat(self, res, pdb_type):
        res_pdb_type = res + "-" + pdb_type
        if res_pdb_type in positive_types:
            polar_feat = torch.tensor([[1, 0, 0]])
        elif res_pdb_type in negative_types:
            polar_feat = torch.tensor([[0, 0, 1]])
        else:
            polar_feat = torch.tensor([[0, 1, 0]])

        if res_pdb_type in aromatic_ring_types:
            aromtic_feat = torch.tensor([[1]])
        else:
            aromtic_feat = torch.tensor([[0]])

        return polar_feat, aromtic_feat

    def get_rec_graph(self):

        self.selec_pose_ha_indices, self.selec_rec_ha_indices = torch.where(self.pose_rec_nodes_dist <= self.inter_cutoff)
        selec_rec_ha_list = list(set(self.selec_rec_ha_indices.numpy()))

        # node feats
        types_feats = []
        polar_feats = []
        aromatic_feats = []
        selec_ha_xyz = []
        selec_ca_xyz = []
        self.selec_rec_ha_map = {}
        for num, i in enumerate(selec_rec_ha_list):
            self.selec_rec_ha_map[i] = num
            res = self.clip_rec_ha_residues[i]
            pdb_type = self.clip_rec_ha_pdb_types[i]
            defined_type = get_defined_type(res, pdb_type)
            types_feats.append(onek_encoding_unk(defined_type,
                                                 all_defined_rec_ha_types).reshape(1, -1))
            _polar_feat, _aromatic_feat = self.get_polar_aromatic_feat(res, pdb_type)   # [1, -1]
            polar_feats.append(_polar_feat)
            aromatic_feats.append(_aromatic_feat)

            selec_ha_xyz.append(self.clip_rec_ha_xyz_tensor[i].reshape(1, 3))
            selec_ca_xyz.append(self.clip_rec_ca_xyz_tensor[i].reshape(1, 3))

        self.selec_ha_xyz = torch.cat(selec_ha_xyz, axis=0)
        self.selec_ca_xyz = torch.cat(selec_ca_xyz, axis=0)

        # nodes feats
        types_feats = torch.cat(types_feats, axis=0)  # 95
        polar_feats = torch.cat(polar_feats, axis=0)  # 3
        aromatic_feats = torch.cat(aromatic_feats, axis=0)  # 1
        ca_dist_feats = generate_2d_pairwise_dist(self.selec_ha_xyz, self.selec_ca_xyz).reshape(-1, 1)  # 1

        self.selec_rec_node_feats = torch.cat([types_feats, polar_feats, aromatic_feats, ca_dist_feats], axis=1).to(
            torch.float32)

        # edge_feats
        rec_ha_dist = generate_2d_dist(self.selec_ha_xyz, self.selec_ha_xyz)
        condition = (rec_ha_dist <= self.rec_cutoff) * (rec_ha_dist > 0.1)
        u_edge_tensor, v_edge_tensor = torch.where(condition)
        self.rec_u_edge_list = u_edge_tensor.numpy().tolist()
        self.rec_v_edge_list = v_edge_tensor.numpy().tolist()

        rec_intra_dist = rec_ha_dist * condition
        rec_intra_dist = rec_intra_dist[rec_intra_dist != 0]
        self.rec_edge_length = rec_intra_dist.ravel().to(torch.float32).reshape(-1, 1)

        return self

    def get_cplx_edge_feats(self):
        self.cplx_pose_indices = self.selec_pose_ha_indices.numpy().tolist()
        self.cplx_rec_node_indices = [self.selec_rec_ha_map[int(i)] for i in self.selec_rec_ha_indices]
        rec_xyz = []
        lig_xyz = []
        rec_ca_xyz = []
        for l, r in zip(self.cplx_pose_indices, self.cplx_rec_node_indices):
            rec_xyz.append(self.selec_ha_xyz[r])
            lig_xyz.append(self.pose_ha_xyz[l])
            rec_ca_xyz.append(self.selec_ca_xyz[r])

        rec_xyz_tensor = torch.cat(rec_xyz, axis=0).reshape(-1, 3)
        lig_xyz_tensor = torch.cat(lig_xyz, axis=0).reshape(-1, 3)
        rec_ca_xyz_tensor = torch.cat(rec_ca_xyz, axis=0).reshape(-1, 3)

        edge_length = generate_2d_pairwise_dist(rec_xyz_tensor, lig_xyz_tensor).reshape(-1, 1)

        rec_2_ca_vec = rec_xyz_tensor - rec_ca_xyz_tensor
        rec_2_lig_vec = rec_xyz_tensor - lig_xyz_tensor
        lig_2_rec_vec = lig_xyz_tensor - rec_xyz_tensor
        lig_2_cent_vec = lig_xyz_tensor - self.pose_center
        cent_2_lig_vec = self.pose_center - lig_xyz_tensor

        angle_1 = torch.cos(cal_angle_between_vectors(rec_2_lig_vec, rec_2_ca_vec).reshape(-1, 1))
        angle_2 = torch.cos(cal_angle_between_vectors(lig_2_cent_vec, lig_2_rec_vec).reshape(-1, 1))
        dihedral_angle = torch.sin(0.5 * cal_dihedral_anlge(cent_2_lig_vec, lig_2_rec_vec, rec_2_ca_vec)).reshape(-1, 1)

        self.cplx_edge_feats = torch.cat([edge_length, dihedral_angle, angle_1, angle_2, torch.ones_like(edge_length)],
                                         axis=1).to(torch.float32)

        no_connect_pose_ha = list(set(self.cplx_pose_indices) ^ set([x for x in range(self.pose_ha_xyz.size()[0])]))
        if len(no_connect_pose_ha) != 0:
            self.u_rv_edge_indices = self.cplx_rec_node_indices.copy()
            self.v_rv_edge_indices = self.cplx_pose_indices.copy()

            virtual_cplx_rv_edge_feats = []
            for p in no_connect_pose_ha:
                self.u_rv_edge_indices.append(0)
                self.v_rv_edge_indices.append(p)
                _e_feats = torch.zeros(1, self.cplx_edge_feats.size()[-1])
                virtual_cplx_rv_edge_feats.append(_e_feats)
            virtual_cplx_rv_edge_feats = torch.cat(virtual_cplx_rv_edge_feats, axis=0)
            self.cplx_rv_edge_feats = torch.cat([self.cplx_edge_feats, virtual_cplx_rv_edge_feats], axis=0)
        else:
            self.u_rv_edge_indices = self.cplx_rec_node_indices.copy()
            self.v_rv_edge_indices = self.cplx_pose_indices.copy()
            self.cplx_rv_edge_feats = self.cplx_edge_feats.clone()

        return self

    def get_cplx_graph(self):
        self.get_rec_lig_dist()

        gs_list = []

        # rec graph
        self.get_rec_graph()

        # cplx graph
        self.get_cplx_edge_feats()

        # create hetero-graph
        graph_data = {
            ("rec", "rec_intra", "rec"): (self.rec_u_edge_list, self.rec_v_edge_list),
            ("lig", "lig_intra", "lig"): (self.lig_u_edge_list, self.lig_v_edge_list),
            ("lig", "lig-rec", "rec"): (self.cplx_pose_indices, self.cplx_rec_node_indices),
            ("rec", 'rec-lig', "lig"): (self.u_rv_edge_indices, self.v_rv_edge_indices)
        }

        g = dgl.heterograph(graph_data)
        g.nodes["rec"].data["feats"] = self.selec_rec_node_feats
        g.nodes["lig"].data["feats"] = self.lig_node_feats
        g.nodes["rec"].data["coord"] = self.selec_ha_xyz
        g.nodes["lig"].data["coord"] = self.pose_ha_xyz
        g.edges["rec_intra"].data["feats"] = self.rec_edge_length
        g.edges["lig_intra"].data["feats"] = self.lig_edge_feats
        g.edges["lig-rec"].data["feats"] = self.cplx_edge_feats
        g.edges["rec-lig"].data["feats"] = self.cplx_rv_edge_feats
        g = dgl.add_self_loop(g, etype=("rec_intra"))
        g = dgl.add_self_loop(g, etype=("lig_intra"))

        return g
