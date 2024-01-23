import os
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from ligand_features import LigandFeature
from load_receptor import ReceptorFile
from pocket_features import PocketFeatures
from cplx_graph import ComplexGraph
from dgl.dataloading import GraphDataLoader
from model import *
from torch.utils.data import DataLoader, Dataset
from utils import run_an_eval_epoch, sdf_split, mol2_split
from argparse import RawDescriptionHelpFormatter
import argparse

class MyDataset(Dataset):
    def __init__(self, rec_gs, cplx_gs):
        self.rec_gs = rec_gs
        self.cplx_gs = cplx_gs

    def __getitem__(self, idx):
        return idx, self.rec_gs[idx], self.cplx_gs[idx]
    def __len__(self):
        return len(self.rec_gs)

def scoring(prefix, pose_fpath, rec, pock_g, model, out_fpath, temp_dir):

    # parse poses
    if pose_fpath.endswith("sdf"):
        poses_content = sdf_split(pose_fpath)
    elif pose_fpath.endswith("mol2"):
        poses_content = mol2_split(pose_fpath)
    else:
        print("InputError: Please input the pose file with .sdf or .mol2 format.")
        exit()
    keys = []
    cplx_graphs = []
    pock_graphs = []
    for idx, p in enumerate(poses_content):
        print(idx)
        basename = prefix + "-" + str(idx+1)
        if pose_fpath.endswith("sdf"):
            mol = Chem.MolFromMolBlock(p)
        elif pose_fpath.endswith("mol2"):
            mol = Chem.MolFromMol2Block(p)
        else:
            print("InputError: Please input the pose file with .sdf or .mol2 format.")
            continue
      
        try:
            lig = LigandFeature(mol=mol)
            lig.lig_to_graph()

            if pock_g == None:
                rec = ReceptorFile(rec_fpath=rec_fpath, ref_lig_fpath=None, temp_dpath=temp_dir)
                rec.parse_ref_coords(lig.pose_ha_xyz)
                rec.clip_rec()
                rec.define_pocket()
                print("receptor clipped ...")
                # parse pocket
                pock_feat = PocketFeatures(rec, pock_center=rec.pock_center)
                pock_g = pock_feat.pock_to_graph()
                pock_g = dgl.add_self_loop(pock_g)
                print("receptor parsed ...")
                pock_graphs.append(pock_g)
            else:
                pock_graphs.append(pock_g)

            # cplx graph
            cplx = ComplexGraph(rec, lig)
            cplx_graph = cplx.get_cplx_graph()
            cplx_graphs.append(cplx_graph)

            keys.append(basename)
        except:
            print("Error:", idx)
    pock_graphs = [pock_g] * len(keys)
    dataset = MyDataset(pock_graphs, cplx_graphs)

    test_loader = GraphDataLoader(dataset, batch_size=32, shuffle=False)
    pred_rmsd, pred_pkd = run_an_eval_epoch(model, test_loader, device="cpu")
    output_results(keys, pred_rmsd, pred_pkd, out_fpath)

def output_results(codes, pred_rmsd, pred_pkd, file_):

    values = np.concatenate([pred_rmsd.reshape(-1, 1), pred_pkd.reshape(-1, 1)], axis=1)
    df = pd.DataFrame(values, index=codes, columns=["pred_rmsd", "pred_pkd"])
    df = df.sort_values(by="pred_rmsd", ascending=True)
    df.to_csv(file_)

if __name__ == "__main__":

    d = "Generate graphs of the protein and the ligand and scoring the complex."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-prefix", type=str, default="receptor",
                        help="Input. Specify a special identifier for the task.")
    parser.add_argument("-rec_fpath", type=str, default="receptor.pdb",
                        help="Input (.pdb). The path of the receptor.")
    parser.add_argument("-ref_lig_fpath", type=str, default=None,
                        help="Input (.sdf or .mol2). The path of the reference pose.")
    parser.add_argument("-pose_fpath", type=str, default="poses.sdf",
                        help="Input (.sdf or .mol2). The fpath of the docking poses.")
    parser.add_argument("-model", type=str, default="model.pth",
                        help="Input. The path of the GNN model.")
    parser.add_argument("-out_fpath", type=str, default="out_fpath",
                        help="Output. The path of the score file.")
    args = parser.parse_args()

    prefix = args.prefix
    rec_fpath = args.rec_fpath
    ref_lig_fpath = args.ref_lig_fpath 
    pose_fpath = args.pose_fpath
    out_fpath = args.out_fpath
    model = torch.load(args.model, map_location=torch.device('cpu'))

    temp_dir = "temp_files" # The dir storing the temp files.
    os.makedirs(temp_dir, exist_ok=True)
    
    if ref_lig_fpath != None:
        # load protein
        rec = ReceptorFile(rec_fpath=rec_fpath, ref_lig_fpath=ref_lig_fpath, temp_dpath=temp_dir)
        rec.clip_rec()
        rec.define_pocket()
        print("receptor clipped ...")

        # parse pocket
        pock_feat = PocketFeatures(rec, pock_center=rec.pock_center)
        pock_g = pock_feat.pock_to_graph()
        pock_g = dgl.add_self_loop(pock_g)
        print("receptor parsed ...")
    else:
        rec = None
        pock_g = None

    try:
        scoring(prefix, pose_fpath, rec, pock_g, model, out_fpath, temp_dir)
    except ValueError:
        print("Error:", pose_fpath)
