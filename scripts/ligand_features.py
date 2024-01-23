import torch
from rdkit import Chem
from load_ligand import Ligand
import numpy as np
from utils import onek_encoding_unk, generate_1d_dist
import time

class LigandFeature(Ligand):
    def __init__(self, mol):
        super(LigandFeature, self).__init__(mol)

        self.atom_indices_dict = {}
        self.pose_ha_xyz = torch.tensor([])
        self.pose_ha_eles = []
        self.all_lig_eles = []
        self.pose_center = torch.tensor([])

    def _get_hybrid_types_feat(self, _type):

        all_types = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2,
                     "OTH"]

        if _type in all_types:
            feat = onek_encoding_unk(_type, all_types)
        else:
            feat = onek_encoding_unk("OTH", all_types)

        return feat

    def get_atom_chem_feats(self):

        pose_all_xyz = self.mol.GetConformer().GetPositions().astype(np.float32)

        node_chem_feats = []
        pose_ha_xyz = []
        ha_num = -1
        element_index = []
        for num, atom in enumerate(self.mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            ele = self.get_atom_types(self.get_ele(atomic_num))
            self.all_lig_eles.append(ele)

            if ele == "H":
                continue

            ha_num += 1
            self.atom_indices_dict[num] = ha_num
            pose_ha_xyz.append(pose_all_xyz[num].reshape(1, 3))
            self.pose_ha_eles.append(ele)
            element_index.append(self.lig_defined_eles.index(ele))
            atom_type_one_hot = onek_encoding_unk(ele, self.lig_defined_eles).to(torch.float32) # 7
            node_chem_feats.append(atom_type_one_hot.reshape(1, -1))

        lig_nodes_feats = torch.cat(node_chem_feats, axis=0)

        self.pose_ha_xyz = torch.from_numpy(np.concatenate(pose_ha_xyz, axis=0))
        self.pose_center = torch.mean(self.pose_ha_xyz, axis=0)

        return element_index, lig_nodes_feats, self.pose_ha_xyz

    def get_edge_feats(self):

        u_edge_list = []
        v_edge_list = []
        edge_feats = []

        for i in range(self.mol.GetNumBonds()):
            bond = self.mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()

            if self.all_lig_eles[u] == "H" or self.all_lig_eles[v] == "H":
                continue

            bt = bond.GetBondType()
            bt_feat = onek_encoding_unk(bt, [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
            conj = torch.tensor([bond.GetIsConjugated() * 1])
            is_ring = torch.tensor([bond.IsInRing() * 1])
            stereo = onek_encoding_unk(bond.GetStereo(), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
            bond_feat = torch.cat([bt_feat, conj, is_ring, stereo])

            u = self.atom_indices_dict[u]
            v = self.atom_indices_dict[v]
            u_edge_list += [u, v]
            v_edge_list += [v, u]
            edge_feats.append(bond_feat.reshape(1, -1))
            edge_feats.append(bond_feat.reshape(1, -1))

        t = time.time()
        edge_chem_feats = torch.cat(edge_feats, axis=0)
        edge_geom_feats = torch.cat([generate_1d_dist(self.pose_ha_xyz[u], self.pose_ha_xyz[v]).reshape(1, -1)
                           for u, v in zip(u_edge_list, v_edge_list)], axis=0)

        edge_feats = torch.cat([edge_chem_feats, edge_geom_feats], axis=1).to(torch.float32)
        return u_edge_list, v_edge_list, edge_feats

    def lig_to_graph(self):

        element_index, lig_nodes_feats, pose_ha_xyz = self.get_atom_chem_feats()
        self.lig_node_feats = lig_nodes_feats
        self.u_lig_edge_list, self.v_lig_edge_list, self.lig_edge_feats = self.get_edge_feats()

        return self