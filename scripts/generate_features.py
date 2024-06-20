import os
import pickle
import numpy as np
import pandas as pd
from ligand_features import LigandFeature
from load_receptor import ReceptorFile
from pocket_features import PocketFeatures
from cplx_graph import ComplexGraph
from utils import cal_rmsd, merge_graph
import dgl
from rdkit import Chem
from utils import sdf_split, mol2_split
from argparse import RawDescriptionHelpFormatter
import argparse
import json
import time

def get_feats(rec_fpath, ref_lig_fpath, pose_fpath, code):
    if not os.path.exists("temp_files"):
        os.mkdir("temp_files")

    rec = ReceptorFile(rec_fpath=rec_fpath, ref_lig_fpath=ref_lig_fpath, temp_dpath="temp_files")
    rec.clip_rec()
    rec.define_pocket()
    print("receptor clipped ...")

    # parse pocket
    pock_feat = PocketFeatures(rec, pock_center=rec.pock_center)
    pock_g = pock_feat.pock_to_graph()
    pock_g = dgl.add_self_loop(pock_g)

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
    #pock_graphs = []
    for idx, p in enumerate(poses_content):
        print(idx)
        basename = code + "-" + str(idx+1)
        try:
            if pose_fpath.endswith("sdf"):
                mol = Chem.MolFromMolBlock(p)
            elif pose_fpath.endswith("mol2"):
                mol = Chem.MolFromMol2Block(p)
            else:
                print("InputError: Please input the pose file with .sdf or .mol2 format.")
                continue
           
            lig = LigandFeature(mol=mol)
            lig.lig_to_graph()

            # cplx graph
            cplx = ComplexGraph(rec, lig)
            cplx_graph = cplx.get_cplx_graph()
            cplx_graphs.append(cplx_graph)
            keys.append(basename)
        except Exception as e:
            print("Error:", e)

    # merge poses graph
    cplx_graphs = merge_graph(keys, cplx_graphs)

    return keys, pock_g, cplx_graphs

if __name__ == "__main__":

    d = "Generate graphs of the protein and the ligand as well as deeprmsd features."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-pose_fpath", type=str, default="general",
                        help="Input. The path of datasets.")
    parser.add_argument("-code", type=str, default="pdb code",
                        help="Input. The pdb code.")
    parser.add_argument("-rec_fpath", type=str, default="receptor.pdb",
                        help="Input. The path of the receptor.")
    parser.add_argument("-ref_lig_fpath", type=str, default="ligand.mol2",
                        help="Input. The path of the ligand.")
    parser.add_argument("-out_dpath", type=str, default="out_dpath",
                        help="Output. The dpath of the output files.")
    args = parser.parse_args()

    code = args.code
    pose_fpath = args.pose_fpath
    rec_fpath = args.rec_fpath
    ref_lig_fpath = args.ref_lig_fpath
    out_dpath = args.out_dpath

    out_rec_graphs = out_dpath + "/rec_graphs/" + code + "_rec_graph.pkl"
    out_cplx_graphs = out_dpath + "/cplx_graphs/" + code + "_cplx_graph.pkl"

    if os.path.exists(out_rec_graphs) and \
            os.path.exists(out_cplx_graphs):
        print("Features are generated ...")
        print("Exit ...")
        exit()

    keys, pock_g, cplx_graphs = get_feats(rec_fpath, ref_lig_fpath, pose_fpath, code)

    if not os.path.exists(out_dpath):
        os.mkdir(out_dpath)

    dirs = ["rec_graphs", "cplx_graphs"]
    for d in dirs:
        if not os.path.exists(out_dpath + "/" + d):
            os.mkdir(out_dpath + "/" + d)

    with open(out_rec_graphs, "wb") as f:
        pickle.dump(pock_g, f)

    with open(out_cplx_graphs, "wb") as f:
        pickle.dump(cplx_graphs, f)