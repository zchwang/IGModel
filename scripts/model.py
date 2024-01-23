import torch
import torch.nn as nn
import dgl
from dgl.nn import EGATConv
import copy
import numpy as np
import torch.nn.functional as F

def get_batch_feats(batch_num_nodes, feats, device):
    num_nodes = sum(batch_num_nodes)
    batch_size = len(batch_num_nodes)

    max_num_nodes = int(batch_num_nodes.max())
    batch = torch.cat([torch.full((1, x.type(torch.int)), y)
                       for x, y in zip(batch_num_nodes, range(batch_size))],
                      dim=1).reshape(-1).type(torch.long).to(device)

    cum_nodes = torch.cat([batch.new_zeros(1), batch_num_nodes.cumsum(dim=0)])
    idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, 0)
    out[idx] = feats
    out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask

class SpatialGatingUnit(nn.Module):
    def __init__(self, hid_feats):
        super().__init__()
        self.norm_1 = nn.BatchNorm1d(hid_feats)
        self.norm_2 = nn.BatchNorm1d(hid_feats)
        self.spatial_proj = nn.Linear(hid_feats, hid_feats)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        a = self.norm_1(v)
        a = self.spatial_proj(a)
        out_1 = u * a

        b = self.norm_2(u)
        b = self.spatial_proj(b)
        out_2 = v * b

        out = out_1 + out_2

        return out

class gMLPBlock(nn.Module):
    def __init__(self, in_feats, hid_feats, rate):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_feats)
        self.channel_proj1 = nn.Sequential(nn.Linear(in_feats, hid_feats * 2), nn.Dropout(p=rate))
        self.channel_proj2 = nn.Sequential(nn.Linear(hid_feats, in_feats), nn.Dropout(p=rate))
        self.sgu = SpatialGatingUnit(hid_feats)

    def forward(self, x):
        residual = x  # in_feats
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x)) # hid_feats * 2

        x = self.sgu(x)  # hid_feats * 2 -> hid_feats
        x = self.channel_proj2(x)  # hid_feats -> in_feats
        out = x + residual

        return out

class gMLP(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 rate=0.0,
                 num_layers=1):
        super(gMLP, self).__init__()

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.rate = rate
        self.num_layers = num_layers

        self.model = nn.Sequential(
            *[gMLPBlock(self.in_feats, self.hid_feats, self.rate) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class EGATBlock(nn.Module):
    def __init__(self,
                 in_node_feats,
                 # hid_node_feats,
                 in_edge_feats,
                 # hid_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads):
        super(EGATBlock, self).__init__()

        self.in_node_feats = in_node_feats
        # self.hid_node_feats = hid_node_feats
        self.in_edge_feats = in_edge_feats
        # self.hid_edge_feats = hid_edge_feats
        self.out_node_feats = out_node_feats
        self.out_edge_feats = out_edge_feats
        self.num_heads = num_heads

        self.egat_1 = EGATConv(in_node_feats=self.in_node_feats,
                               in_edge_feats=self.in_edge_feats,
                               out_node_feats=self.out_node_feats,
                               out_edge_feats=self.out_edge_feats,
                               bias=True,
                               num_heads=self.num_heads,
                               )

    def forward(self, g, n_feats, e_feats, get_attention=False):
        if get_attention == True:
            n_feats_1, e_feats_1, attention = self.egat_1(g, n_feats, e_feats, get_attention=get_attention)
            return n_feats_1, e_feats_1, attention
        else:
            n_feats_1, e_feats_1 = self.egat_1(g, n_feats, e_feats, get_attention=get_attention)
            return n_feats_1, e_feats_1


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class HeteroGNN(nn.Module):
    def __init__(self, num_heads=2):
        super().__init__()

        self.num_heads = num_heads
        self.rec_egat_1 = EGATBlock(in_node_feats=100,
                                    in_edge_feats=1,
                                    out_node_feats=128,
                                    out_edge_feats=32,
                                    num_heads=self.num_heads)

        self.lig_egat_1 = EGATBlock(in_node_feats=7,
                                    in_edge_feats=11,
                                    out_node_feats=128,
                                    out_edge_feats=32,
                                    num_heads=self.num_heads)

        self.cplx_fw_egat_1 = EGATBlock(in_node_feats=(7, 100),
                                        in_edge_feats=5,
                                        out_node_feats=128,
                                        out_edge_feats=32,
                                        num_heads=self.num_heads)
        self.cplx_rv_egat_1 = EGATBlock(in_node_feats=(100, 7),
                                        in_edge_feats=5,
                                        out_node_feats=128,
                                        out_edge_feats=32,
                                        num_heads=self.num_heads)

        self.rec_egat_2 = EGATBlock(in_node_feats=256,
                                    in_edge_feats=64,
                                    out_node_feats=512,
                                    out_edge_feats=128,
                                    num_heads=self.num_heads)
        self.lig_egat_2 = EGATBlock(in_node_feats=256,
                                    in_edge_feats=64,
                                    out_node_feats=512,
                                    out_edge_feats=128,
                                    num_heads=self.num_heads)
        self.cplx_fw_egat_2 = EGATBlock(in_node_feats=(256, 256),
                                        in_edge_feats=64,
                                        out_node_feats=512,
                                        out_edge_feats=128,
                                        num_heads=self.num_heads)
        self.cplx_rv_egat_2 = EGATBlock(in_node_feats=(256, 256),
                                        in_edge_feats=64,
                                        out_node_feats=512,
                                        out_edge_feats=128,
                                        num_heads=self.num_heads)

        self.nodes_mlp_1 = clones(nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256)), 2)
        self.nodes_mlp_2 = clones(nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512)), 2)

        self.norm_n_1 = clones(nn.BatchNorm1d(256), 4)
        self.norm_e_1 = clones(nn.BatchNorm1d(64), 4)

        self.norm_n_2 = clones(nn.BatchNorm1d(512), 4)

        self.mlp_layer = clones(nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512)), 2)
        #self.norm = nn.BatchNorm1d(512)

        self.initial_params()

    def parse_hetero_graph(self, hetero_g):

        self.rec_sub_g = dgl.node_type_subgraph(hetero_g, ["rec"])
        self.lig_sub_g = dgl.node_type_subgraph(hetero_g, ["lig"])
        self.cplx_fw_sub_g = dgl.edge_type_subgraph(hetero_g, ["lig-rec"])
        self.cplx_rv_sub_g = dgl.edge_type_subgraph(hetero_g, ["rec-lig"])

        self.init_rec_n_feats = self.rec_sub_g.ndata["feats"]
        self.init_rec_e_feats = self.rec_sub_g.edata["feats"]

        self.init_lig_n_feats = self.lig_sub_g.ndata["feats"]
        self.init_lig_e_feats = self.lig_sub_g.edata["feats"]

        self.init_cplx_fw_n_feats = self.cplx_fw_sub_g.ndata["feats"]
        self.init_cplx_fw_e_feats = self.cplx_fw_sub_g.edata["feats"]

        self.init_cplx_rv_n_feats = self.cplx_rv_sub_g.ndata["feats"]
        self.init_cplx_rv_e_feats = self.cplx_rv_sub_g.edata["feats"]

        # cplx rv mask
        cplx_rv_e_feats = torch.sum(self.init_cplx_rv_e_feats, axis=1)
        self.rv_edge_nan_idx = torch.where(torch.isnan(cplx_rv_e_feats))[0]

        u_rv_list, v_rv_list = self.cplx_rv_sub_g.edges()  # u is rec, v is lig
        self.v_rv_nan_idx = [v_rv_list[i] for i in self.rv_edge_nan_idx]

        return self

    def update_cplx_rv_feats(self, n_feats, e_feats):
        N, n_dim = n_feats.size()
        M, e_dim = e_feats.size()

        for i in self.rv_edge_nan_idx:
            e_feats[i] = torch.zeros(e_dim)

        for i in self.v_rv_nan_idx:
            n_feats[i] = torch.zeros(n_dim)

        return n_feats, e_feats

    def initial_params(self):

        for l in self.nodes_mlp_1:
            torch.nn.init.normal_(l[0].weight.data)
            l[0].bias.data.fill_(0)

        for l in self.nodes_mlp_2:
            torch.nn.init.normal_(l[0].weight.data)
            l[0].bias.data.fill_(0)

        for l in self.mlp_layer:
            torch.nn.init.normal_(l[0].weight.data)
            l[0].bias.data.fill_(0)

    def parse_hetero_information(self, hetero_g):

        rec_nodes_number = []
        lig_nodes_number = []
        unbatch_graphs = dgl.unbatch(hetero_g)
        for g in unbatch_graphs:
            rec_nodes_number.append(g.num_nodes("rec"))
            lig_nodes_number.append(g.num_nodes("lig"))

        self.rec_nodes_number = torch.from_numpy(np.array(rec_nodes_number)).to(hetero_g.device)
        self.lig_nodes_number = torch.from_numpy(np.array(lig_nodes_number)).to(hetero_g.device)

        return self

    def forward(self, hetero_g):

        self.parse_hetero_graph(hetero_g)
        self.parse_hetero_information(hetero_g)

        # step 1 & egat 1
        ## rec
        rec_n_feats_1, rec_e_feats_1 = self.rec_egat_1(self.rec_sub_g, self.init_rec_n_feats, self.init_rec_e_feats)
        rec_n_feats_1 = rec_n_feats_1.view(-1, 128 * self.num_heads)
        rec_n_feats_1 = self.norm_n_1[0](rec_n_feats_1)
        rec_e_feats_1 = rec_e_feats_1.view(-1, 32 * self.num_heads)
        rec_e_feats_1 = self.norm_e_1[0](rec_e_feats_1)

        ## lig
        lig_n_feats_1, lig_e_feats_1 = self.lig_egat_1(self.lig_sub_g, self.init_lig_n_feats, self.init_lig_e_feats)
        lig_n_feats_1 = lig_n_feats_1.view(-1, 128 * self.num_heads)
        lig_n_feats_1 = self.norm_n_1[1](lig_n_feats_1)
        lig_e_feats_1 = lig_e_feats_1.view(-1, 32 * self.num_heads)
        lig_e_feats_1 = self.norm_e_1[1](lig_e_feats_1)

        ## cplx forward: hetero graph: lig -> rec
        cplx_fw_n_feats_1, cplx_fw_e_feats_1 = self.cplx_fw_egat_1(self.cplx_fw_sub_g,
                                                                   (self.init_lig_n_feats, self.init_rec_n_feats),
                                                                   self.init_cplx_fw_e_feats)
        cplx_fw_n_feats_1 = cplx_fw_n_feats_1.view(-1, 128 * self.num_heads)
        cplx_fw_n_feats_1 = self.norm_n_1[2](cplx_fw_n_feats_1)
        cplx_fw_e_feats_1 = cplx_fw_e_feats_1.view(-1, 32 * self.num_heads)
        cplx_fw_e_feats_1 = self.norm_e_1[2](cplx_fw_e_feats_1)

        ## cplx reverse: hetero graph: rec -> lig
        cplx_rv_n_feats_1, cplx_rv_e_feats_1 = self.cplx_rv_egat_1(self.cplx_rv_sub_g,
                                                                   (self.init_rec_n_feats, self.init_lig_n_feats),
                                                                   self.init_cplx_rv_e_feats)
        cplx_rv_n_feats_1 = cplx_rv_n_feats_1.view(-1, 128 * self.num_heads)
        cplx_rv_n_feats_1 = self.norm_n_1[3](cplx_rv_n_feats_1)
        cplx_rv_e_feats_1 = cplx_rv_e_feats_1.view(-1, 32 * self.num_heads)
        cplx_rv_e_feats_1 = self.norm_e_1[3](cplx_rv_e_feats_1)

        ## update nodes & edage
        rec_n_feats_2 = rec_n_feats_1 + self.nodes_mlp_1[0](cplx_fw_n_feats_1)  # 256
        lig_n_feats_2 = lig_n_feats_1 + self.nodes_mlp_1[1](cplx_rv_n_feats_1)  # 256

        # step 2 & egat 2
        ## rec
        rec_n_feats_3, rec_e_feats_3, cplx_rec_attention = \
            self.rec_egat_2(self.rec_sub_g, rec_n_feats_2, rec_e_feats_1, get_attention=True)
        rec_n_feats_3 = torch.mean(rec_n_feats_3, axis=1)
        rec_n_feats_3 = self.norm_n_2[0](rec_n_feats_3)

        ## lig
        lig_n_feats_3, lig_e_feats_3, cplx_lig_attention = \
            self.lig_egat_2(self.lig_sub_g, lig_n_feats_2, lig_e_feats_1, get_attention=True)
        lig_n_feats_3 = torch.mean(lig_n_feats_3, axis=1)
        lig_n_feats_3 = self.norm_n_2[1](lig_n_feats_3)

        ## cplx forward: hetero graph: lig -> rec
        cplx_fw_n_feats_3, cplx_fw_e_feats_3, cplx_fw_attention = self.cplx_fw_egat_2(self.cplx_fw_sub_g,
                                    (lig_n_feats_2, rec_n_feats_2), cplx_fw_e_feats_1, get_attention=True)
        cplx_fw_n_feats_3 = torch.mean(cplx_fw_n_feats_3, axis=1)
        cplx_fw_n_feats_3 = self.norm_n_2[2](cplx_fw_n_feats_3)

        ## cplx reverse: hetero graph: rec -> lig
        cplx_rv_n_feats_3, cplx_rv_e_feats_3, cplx_rv_attention = self.cplx_rv_egat_2(self.cplx_rv_sub_g,
                                    (rec_n_feats_2, lig_n_feats_2), cplx_rv_e_feats_1, get_attention=True)
        cplx_rv_n_feats_3 = torch.mean(cplx_rv_n_feats_3, axis=1)
        cplx_rv_n_feats_3 = self.norm_n_2[3](cplx_rv_n_feats_3)

        ## update nodes
        rec_n_feats = rec_n_feats_3 + self.nodes_mlp_2[0](cplx_fw_n_feats_3)
        lig_n_feats = lig_n_feats_3 + self.nodes_mlp_2[1](cplx_rv_n_feats_3)

        rec_n_feats, rec_n_mask = get_batch_feats(self.rec_nodes_number, rec_n_feats, hetero_g.device)
        lig_n_feats, lig_n_mask = get_batch_feats(self.lig_nodes_number, lig_n_feats, hetero_g.device)

        return rec_n_feats, lig_n_feats, rec_n_mask, lig_n_mask, cplx_rec_attention, cplx_lig_attention, \
               cplx_fw_attention, cplx_rv_attention

class RecGNN(nn.Module):
    def __init__(self, num_heads=2):
        super(RecGNN, self).__init__()

        self.num_heads = num_heads
        self.egat_1 = EGATBlock(in_node_feats=34,
                                in_edge_feats=7,
                                out_node_feats=128,
                                out_edge_feats=32,
                                num_heads=self.num_heads)

        self.egat_2 = EGATBlock(in_node_feats=256,
                                in_edge_feats=64,
                                out_node_feats=512,
                                out_edge_feats=128,
                                num_heads=self.num_heads)

        self.norm = nn.BatchNorm1d(512)

    def parse_graph_information(self, gs):

        nodes_number = []
        unbatch_graphs = dgl.unbatch(gs)
        for g in unbatch_graphs:
            nodes_number.append(g.num_nodes())
        self.nodes_number = torch.from_numpy(np.array(nodes_number)).to(gs.device)

        return self

    def forward(self, rec_g):

        self.parse_graph_information(rec_g)

        n_feats = rec_g.ndata["feats"]
        e_feats = rec_g.edata["feats"]

        n_feats_1, e_feats_1 = self.egat_1(rec_g, n_feats, e_feats)
        n_feats_1 = n_feats_1.view(-1, 128 * self.num_heads)
        e_feats_1 = e_feats_1.view(-1, 32 * self.num_heads)

        n_feats_2, e_feats_2, pock_attention = self.egat_2(rec_g, n_feats_1, e_feats_1, get_attention=True)
        n_feats_2 = torch.mean(n_feats_2, axis=1)

        n_feats, _ = get_batch_feats(self.nodes_number, n_feats_2, rec_g.device)  # 512
        n_feats = self.norm(torch.sum(n_feats, axis=1))

        return n_feats, pock_attention

class FinalModel(nn.Module):
    def __init__(self,
                 rec_gnn=None,
                 cplx_gnn=None,
                 rate=0.0
                 ):
        super(FinalModel, self).__init__()

        self.rec_gnn = rec_gnn
        self.cplx_gnn = cplx_gnn
        self.drop_rate = rate

        self.cplx_rec_encoder = nn.Sequential(nn.Linear(512, 1024, bias=False))
        self.cplx_lig_encoder = nn.Sequential(nn.Linear(512, 1024, bias=False))

        self.rec_encoder = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024))
        self.cplx_rec_encoder_1 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024))
        self.cplx_lig_encoder_1 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024))

        # rmsd prediction layers
        self.rmsd_gmlp = gMLP(in_feats=1024,
                         hid_feats=1024,
                         num_layers=1)

        self.rmsd_mlp_layers = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=self.drop_rate), nn.SiLU(),
                                 nn.Linear(512, 128), nn.BatchNorm1d(128), nn.Dropout(p=self.drop_rate), nn.SiLU())

        self.decoder_hrmsd = nn.Sequential(nn.Linear(128, 1), nn.ReLU())

        # pkd prediction layers
        self.pkd_gmlp = gMLP(in_feats=1024,
                         hid_feats=1024,
                         num_layers=1)
        self.pkd_mlp_layers = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=self.drop_rate),
                                             nn.Linear(512, 128), nn.BatchNorm1d(128), nn.Dropout(p=self.drop_rate))
        self.mlp_rmsd_to_pkd = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128))

        self.mlp_pkd = nn.Sequential(nn.Linear(128, 1))
        self.mlp_weight = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def get_dist_map(self, lig_feats, rec_feats):

        """

        Args:
            lig_feats: [B, N_lig, dim]
            rec_feats: [B, N_rec, dim]

        Returns:
            lig_rec_distance_map: [B, N_lig, N_rec]
        """
        B, N_lig, dim = lig_feats.size()
        B, N_rec, dim = rec_feats.size()

        lig_feats = torch.sum(lig_feats, axis=-1).unsqueeze(-1).repeat(1, 1, N_rec)
        rec_feats = torch.sum(rec_feats, axis=-1).unsqueeze(-2).repeat(1, N_lig, 1)

        lig_rec_dist = lig_feats + rec_feats

        return lig_rec_dist


    def forward(self, rec_g, cplx_g):

        if self.rec_gnn is not None:
            rec_feats, pock_attention = self.rec_gnn(rec_g)  # 512

        if self.cplx_gnn is not None:
            cplx_rec_feats, cplx_lig_feats, rec_n_mask, lig_n_mask, cplx_rec_attention, cplx_lig_attention, \
               cplx_fw_attention, cplx_rv_attention = self.cplx_gnn(cplx_g)  # 512

        cplx_rec_feats = self.cplx_rec_encoder(cplx_rec_feats)  # [B, N_rec, 1024]
        cplx_lig_feats = self.cplx_lig_encoder(cplx_lig_feats)  # [B, N_lig, 1024]

        cplx_rec_feats = torch.sum(cplx_rec_feats, axis=1)  # [B, 1024]
        cplx_rec_feats = self.cplx_rec_encoder_1(cplx_rec_feats) + self.rec_encoder(rec_feats)
        cplx_lig_feats = self.cplx_lig_encoder_1(torch.sum(cplx_lig_feats, axis=1))
        feats = cplx_rec_feats + cplx_lig_feats

        # rmsd prediction block
        x_rmsd = self.rmsd_gmlp(feats)
        x_rmsd = self.rmsd_mlp_layers(x_rmsd)  # 128

        pred_hrmsd = self.decoder_hrmsd(x_rmsd)
    
        # pkd prediction block
        x_pkd_feats = self.pkd_gmlp(feats)
        x_pkd_feats = self.pkd_mlp_layers(x_pkd_feats) + self.mlp_rmsd_to_pkd(x_rmsd)  # 128

        pred_pkd = self.mlp_pkd(x_pkd_feats)
        W = self.mlp_weight(x_pkd_feats)

        return pred_hrmsd, pred_pkd, cplx_lig_feats, feats, W, pock_attention, cplx_rec_attention, cplx_lig_attention, cplx_fw_attention, cplx_rv_attention
