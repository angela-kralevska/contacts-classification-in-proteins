from Bio.PDB import DSSP, HSExposureCB, PPBuilder, is_aa, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import pandas as pd
from pathlib import Path, PurePath
import json
import argparse
import logging
import numpy as np
import os
import tensorflow as tf


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file', help='mmCIF or PDB file')
    parser.add_argument('-conf_file', help='Configuration and parameters file', default=None)
    parser.add_argument('-out_dir', help='Output directory', default='.')
    parser.add_argument('-model_file', help='final_model.h5 path', default='final_model.h5')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Set the logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    # fileHandler = logging.FileHandler("{}/info.log".format(args.out_dir))
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)

    # Load the config file
    # If not provided, set the path to "configuration.json", which is in the same folder of this Python file
    src_dir = str(PurePath(os.path.realpath(__file__)).parent)
    config_file = src_dir + "/configuration.json" if args.conf_file is None else args.configuration
    with open(config_file) as f:
        config = json.load(f)

    # Fix configuration paths (identified by the '_file' or '_dir' suffix in the field name)
    # If paths are relative it expects they refer to the absolute position of this file
    for k in config:
        if ('_file' in k or '_dir' in k) and k[0] != '/':
            config[k] = src_dir + '/' + config[k]

    # Start
    pdb_id = Path(args.pdb_file).stem
    logging.info("{} processing".format(pdb_id))

    # Ramachandran regions
    regions_matrix = []
    with open(config["rama_file"]) as f:
        for line in f:
            if line:
                regions_matrix.append([int(ele) for ele in line.strip().split()])

    # Atchely scales
    atchley_scale = {}
    with open(config["atchley_file"]) as f:
        next(f)
        for line in f:
            line = line.strip().split("\t")
            atchley_scale[line[0]] = line[1:]

    # Parse the structure
    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, args.pdb_file)

    # Get valid residues
    residues = [residue for residue in structure[0].get_residues() if is_aa(residue) and residue.id[0] == ' ']
    if not residues:
        logging.warning("{} no valid residues error  (skipping prediction)".format(pdb_id))
        raise ValueError("no valid residues")

    # Calculate DSSP
    dssp = {}
    try:
        # Read the configuration file
        with open('configuration.json', 'r') as config_file:
            config = json.load(config_file)

        # Get the dssp file path
        dssp_file_path = config["dssp_file"]

        dssp = dict(DSSP(structure[0], args.pdb_file, dssp=dssp_file_path))
    except Exception:
        logging.warning("{} DSSP error".format(pdb_id))

    # Calculate Half Sphere Exposure
    hse = {}
    try:
        hse = dict(HSExposureCB(structure[0]))
    except Exception:
        logging.warning("{} HSE error".format(pdb_id))

    # Calculate ramachandran values
    rama_dict = {}  # {(chain_id, residue_id): [phi, psi, ss_class], ...}
    ppb = PPBuilder()
    for chain in structure[0]:
        for pp in ppb.build_peptides(chain):
            phi_psi = pp.get_phi_psi_list()  # [(phi_residue_1, psi_residue_1), ...]
            for i, residue in enumerate(pp):
                phi, psi = phi_psi[i]
                ss_class = None
                if phi is not None and psi is not None:
                    for x, y, width, height, ss_c, color in config["rama_ss_ranges"]:
                        if x <= phi < x + width and y <= psi < y + height:
                            ss_class = ss_c
                            break
                rama_dict[(chain.id, residue.id)] = [phi, psi, ss_class]

    # Load the trained model
    try:
        model = tf.keras.models.load_model(args.model_file)
    except:
        raise Exception(
            "Make sure your model is present in the path specified as an argument when executing this file.")

    # Generate contacts and add features
    data = []

    dssp_dict = {
        '-': 0,
        'B': 1,
        'E': 2,
        'G': 3,
        'H': 4,
        'I': 5,
        'S': 6,
        'T': 7
    }

    ss3_dict = {'H': 0, 'L': 1}

    label_dict = {
        0: "HBOND",
        1: "IONIC",
        2: "PICATION",
        3: "PIPISTACK",
        4: "SSBOND",
        5: "VDW"
    }

    ns = NeighborSearch([atom for residue in residues for atom in residue])
    for residue_1, residue_2 in ns.search_all(config["distance_threshold"], level="R"):
        data_per_row = []
        index_1 = residues.index(residue_1)
        index_2 = residues.index(residue_2)

        if abs(index_1 - index_2) >= config["sequence_separation"]:

            aa_1 = seq1(residue_1.get_resname())
            aa_2 = seq1(residue_2.get_resname())
            chain_1 = residue_1.get_parent().id
            chain_2 = residue_2.get_parent().id
            distance = residue_1["CA"] - residue_2["CA"]
            hydrophobicity_res1 = ProteinAnalysis(aa_1).gravy()
            hydrophobicity_res2 = ProteinAnalysis(aa_2).gravy()
            charge_res1 = ProteinAnalysis(aa_1).isoelectric_point()
            charge_res2 = ProteinAnalysis(aa_2).isoelectric_point()

            aromatic_residues = ['F', 'W', 'Y', 'H']
            aromatic_res1 = aa_1 in aromatic_residues
            aromatic_res2 = aa_2 in aromatic_residues

            data_per_row.append((pdb_id,
                         chain_1,
                         *residue_1.id[1:],
                         aa_1,
                         *dssp.get((chain_1, residue_1.id), [None, None, None, None])[2:4],
                         *hse.get((chain_1, residue_1.id), [None, None])[:2],
                         *rama_dict.get((chain_1, residue_1.id), [None, None, None]),
                         *atchley_scale[aa_1],
                         chain_2,
                         *residue_2.id[1:],
                         aa_2,
                         *dssp.get((chain_2, residue_2.id), [None, None, None, None])[2:4],
                         *hse.get((chain_2, residue_2.id), [None, None])[:2],
                         *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                         *atchley_scale[aa_2],
                         distance,
                         hydrophobicity_res1,
                         hydrophobicity_res2,
                         charge_res1,
                         charge_res2,
                         aromatic_res1,
                         aromatic_res2))

            df = pd.DataFrame(data_per_row, columns=['pdb_id',
                                     's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_ss3', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                                     't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_ss3', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5',
                                     'Distance', 's_residue_hydrophobicity', 't_residue_hydrophobicity', 's_residue_isoelectric_point', 't_residue_isoelectric_point', 's_residue_aromatic', 't_residue_aromatic']).round(3)

            # Check if there are missing values in the DataFrame
            has_null_values = df.isna().any().any()

            if not has_null_values:

                # tuka preprocessing i predviduvanje pravi i zacuvaj vo data

                # Label encoding
                df['s_ss8'] = df['s_ss8'].map(dssp_dict)
                df['t_ss8'] = df['t_ss8'].map(dssp_dict)

                df['s_ss3'] = df['s_ss3'].map(ss3_dict)
                df['t_ss3'] = df['t_ss3'].map(ss3_dict)

                # dropping id columns

                X_test = list(
                    df.drop(['pdb_id', 's_ch', 's_resi', 's_ins', 's_resn', 't_ch', 't_resi', 't_ins', 't_resn', 's_residue_hydrophobicity', 't_residue_hydrophobicity'],
                            axis=1).values)
                # Convert to NumPy array
                data_array = np.array(X_test)

                # Replace string values with appropriate numerical representations
                data_array[data_array == 'True'] = 1
                data_array[data_array == 'False'] = 0
                data_array = data_array.astype(float)

                # Extract the desired columns and reshape the array
                data_array = data_array[:, :29].reshape(1, -1)

                # Verify the conversion
                yhat = model.predict(data_array)  # here in yhat we have the predicted probabiities

                yhat_classes = yhat.round()  # with this line we obtain the class one-hot encoded

                # find where we have 1 in the array, that means find which classes are predicted

                result_classes = np.array(yhat_classes)  # Example result array

                # Get the indices where the value is 1
                class_indices = np.where(result_classes == 1)[1]

                # Map each index to the corresponding class label
                classes = [label_dict[idx] for idx in class_indices]

                data.append((pdb_id,
                        chain_1,
                        *residue_1.id[1:],
                        aa_1,
                        *dssp.get((chain_1, residue_1.id), [None, None, None, None])[2:4],
                        *hse.get((chain_1, residue_1.id), [None, None])[:2],
                        *rama_dict.get((chain_1, residue_1.id), [None, None, None]),
                        *atchley_scale[aa_1],
                        chain_2,
                        *residue_2.id[1:],
                        aa_2,
                        *dssp.get((chain_2, residue_2.id), [None, None, None, None])[2:4],
                        *hse.get((chain_2, residue_2.id), [None, None])[:2],
                        *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                        *atchley_scale[aa_2],
                        distance,
                        hydrophobicity_res1,
                        hydrophobicity_res2,
                        charge_res1,
                        charge_res2,
                        aromatic_res1,
                        aromatic_res2,
                        yhat_classes[0],
                        yhat[0],
                        classes))

    if not data:
        logging.warning("{} no contacts error (skipping prediction)".format(pdb_id))
        raise ValueError("no contacts error (skipping prediction)")

    # Create a DataFrame and save to file
    df = pd.DataFrame(data, columns=['pdb_id',
                                     's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_ss3', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                                     't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_ss3', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5',
                                     'Distance', 's_residue_hydrophobicity', 't_residue_hydrophobicity', 's_residue_isoelectric_point', 't_residue_isoelectric_point',
                                     's_residue_aromatic', 't_residue_aromatic', 'model_predictions', 'prediction_scores', 'predicted_classes'])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].apply(lambda t: round(t, 3))
    df.to_csv("{}/{}.tsv".format(args.out_dir, pdb_id), sep="\t", index=False)
