#!/bin/bash

# Specify a reference ligand structure to grab the pocket
python scripts/scoring.py \
	-prefix 1bcu \
	-rec_fpath samples/1bcu/1bcu_protein_atom_noHETATM.pdb	\
	-ref_lig_fpath samples/1bcu/1bcu_ligand.sdf \
	-pose_fpath samples/1bcu/1bcu_decoys.sdf \
	-model models/saved_model.pth \
	-out_fpath scores.csv
exit

# Capture the pocket using docking poses as the reference
python scripts/scoring.py \
	-prefix 1bcu \
	-rec_fpath samples/1bcu/1bcu_protein_atom_noHETATM.pdb	\
	-pose_fpath samples/1bcu/1bcu_decoys.sdf \
	-model models/self-ref_model.pth \
	-out_fpath self-ref_scores.csv
