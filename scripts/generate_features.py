import os
import pickle
import numpy as np
import pandas as pd
from ligand_features import LigandFeature
from load_receptor import ReceptorFile, PocketToplogy
from pocket_features import PocketFeatures
from cplx_graph import ComplexGraph
from utils import cal_rmsd, merge_graph
from argparse import RawDescriptionHelpFormatter
import argparse
import json
import time

def get_feats(dpath, code):
    rec_fpath = dpath + "/" + code + "/" + code + "_protein_atom_noHETATM.pdbqt"
    poses_list = samples_dict[code]
    ref_lig_fpath = dpath + "/" + code + "/" + code + "_ligand.sdf"

    if not os.path.exists("temp_files"):
        os.mkdir("temp_files")

    rec = ReceptorFile(rec_fpath=rec_fpath, ref_lig_fpath=ref_lig_fpath)
    rec.clip_rec()
    rec.define_pocket()

    # parse pocket & pocket graph
    pock = PocketToplogy(rec.lines, rec.clip_ha_indices)
    pock.parse_pocket()

    pock_feat = PocketFeatures(rec, pock_center=rec.pock_center)
    pock_g = pock_feat.pock_to_graph()

    # parse poses
    poses_list += [code + "_ligand"]

    keys = []
    cplx_graphs = []
    real_rmsd = []
    for num, p in enumerate(poses_list):
        print(num, p)
        try:
            if p[-6:] == "ligand":
                pose_file = ref_lig_fpath
                sym_rmsd, h_rmsd = 0., 0.
            elif len(p) < 23:
                pose_file = dpath + "/" + code + "/" + p + ".sdf"
                sym_rmsd, h_rmsd = cal_rmsd(ref_lig_fpath, pose_file)
            else:
                pose_file = ledock_dpath + "/" + code + "/" + p + ".sdf"
                sym_rmsd, h_rmsd = cal_rmsd(ref_lig_fpath, pose_file)

            # ligand graph
            lig = LigandFeature(pose_fpath=pose_file)
            lig.lig_to_graph()

            # cplx graph
            cplx = ComplexGraph(rec, lig)
            cplx_graph = cplx.get_cplx_graph()
            cplx_graphs.append(cplx_graph)

            # real rmsd
            real_rmsd.append(np.array([[sym_rmsd, h_rmsd]]))
            keys.append(p)
        except Exception as e:
            print("Error:", e, num, p)

    # merge poses graph
    cplx_graphs = merge_graph(keys, cplx_graphs)

    real_rmsd = np.concatenate(real_rmsd, axis=0)
    real_rmsd_df = pd.DataFrame(real_rmsd, index=keys, columns=["sys_rmsd", "h_rmsd"])

    return keys, pock_g, cplx_graphs, real_rmsd_df

if __name__ == "__main__":

    d = "Generate graphs of the protein and the ligand as well as deeprmsd features."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-dpath", type=str, default="general",
                        help="Input. The path of datasets.")
    parser.add_argument("-ledock_dpath", type=str, default="general",
                        help="Input. The path of datasets.")
    parser.add_argument("-samples_dict", type=str, default="general",
                        help="Input. The path of datasets.")
    parser.add_argument("-code", type=str, default="pdb code",
                        help="Input. The pdb code.")
    parser.add_argument("-rec_fpath", type=str, default="receptor.pdb",
                        help="Input. The path of the receptor.")
    parser.add_argument("-lig_fpath", type=str, default="ligand.mol2",
                        help="Input. The path of the ligand.")
    parser.add_argument("-out_dpath", type=str, default="out_dpath",
                        help="Output. The dpath of the output files.")
    args = parser.parse_args()

    code = args.code
    dpath = args.dpath
    ledock_dpath = args.ledock_dpath
    out_dpath = args.out_dpath

    with open(args.samples_dict) as f:
        d = f.readlines()[0]
    samples_dict = json.loads(d)

    out_rec_graphs = out_dpath + "/rec_graphs/" + code + "_rec_graph.pkl"
    out_cplx_graphs = out_dpath + "/cplx_graphs/" + code + "_cplx_graph.pkl"
    out_real_rmsd = out_dpath + "/real_rmsd/" + code + "_real_rmsd.pkl"

    if os.path.exists(out_rec_graphs) and \
            os.path.exists(out_cplx_graphs) and \
            os.path.exists(out_real_rmsd):
        print("Features are generated ...")
        print("Exit ...")
        exit()

    keys, pock_g, cplx_graphs, real_rmsd_df = get_feats(dpath, code)

    if not os.path.exists(out_dpath):
        os.mkdir(out_dpath)

    dirs = ["rec_graphs", "cplx_graphs", "real_rmsd"]
    for d in dirs:
        if not os.path.exists(out_dpath + "/" + d):
            os.mkdir(out_dpath + "/" + d)

    with open(out_rec_graphs, "wb") as f:
        pickle.dump(pock_g, f)

    with open(out_cplx_graphs, "wb") as f:
        pickle.dump(cplx_graphs, f)

    real_rmsd_df.to_pickle(out_real_rmsd)
