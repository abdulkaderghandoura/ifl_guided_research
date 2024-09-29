import torch
import numpy as np
import argparse
import json

from pathlib import Path

import deep_sdf
import deep_sdf.workspace as ws


def interpolate_complete_motion_code(Seq, observed_phases, observed_c_m, T_N):
    # Compute PCA on the training set motion codes (Seq)
    Seq_mean = torch.mean(Seq, dim=0)  # Mean of the training set motion codes
    Seq_centered = Seq - Seq_mean  # Center the data (S, T_N * K_m)

    # Perform SVD (PCA)
    _, _, Vh = torch.linalg.svd(Seq_centered, full_matrices=False)
    V = Vh.T  # Get the right singular vectors (V) from Vh

    L, K_m = observed_c_m.shape
    K_beta = min(L, V.shape[1])
    V_m = V[:, :K_beta]  # (T_N * K_m, K_beta)

    # # Extract the rows of V_m​ corresponding to the observed phases.
    # V_m_observed0 = torch.cat([V_m[K_m * i: K_m * (i + 1), :] for i in observed_phases], dim=0)
    # V_m_observed = V_m.reshape(T_N, K_m, K_beta)[observed_phases].reshape(L * K_m, K_beta)
    # assert V_m_observed0 == V_m_observed, "Incorrect slicing"
    V_m_reduced = V_m[:L * K_m, :]

    # Solve for β using least-squares regression
    beta = torch.linalg.lstsq(V_m_reduced, observed_c_m.view(-1, 1)).solution

    # Reconstruct the complete motion sequence using β
    c_m_complete = (Seq_mean + (V_m @ beta).T).view(T_N, K_m)  # (T_N, K_m)

    return c_m_complete


def reconstruct_interpolated_phases(decoder, c_s, c_m, missing_phases, output_dir, args, seq_id, T_N):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in missing_phases:
        tau = torch.FloatTensor([i / (T_N - 1)]).unsqueeze(0)

        mesh_file_name = f"{seq_id}_{i:02d}"
        mesh_file_path = Path(output_dir) / ws.reconstruction_meshes_subdir / mesh_file_name
        mesh_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        motion_file_name = f"{seq_id}_{i:02d}"
        motion_file_path = Path(output_dir) / ws.reconstruction_motions_subdir / motion_file_name
        motion_file_path.parent.mkdir(parents=True, exist_ok=True)
        decoder.eval()

        with torch.no_grad():
            deep_sdf.mesh.create_mesh_4dsdf(
                decoder, c_s, c_m[i].unsqueeze(0), tau, str(mesh_file_path), 
                str(motion_file_path), N=args.resolution, max_batch=int(2 ** 17)
            )
        
        print(f"Reconstructed: {seq_id}_{i:02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate and reconstruct missing intermediate phases"
    )
    
    parser.add_argument(
        "--experiment", 
        "-e", 
        required=True, 
        help="Experiment directory with model specs."
    )
    
    parser.add_argument(
        "--data", 
        "-d", 
        required=True, 
        help="Directory with SDF samples."
    )

    parser.add_argument(
        "--split", 
        "-s", 
        required=True, 
        help="The split file for data."
    )

    parser.add_argument(
        "--checkpoint", 
        "-c", 
        default="latest", 
        help="Checkpoint weights to use."
    )

    parser.add_argument(
        "--resolution", 
        default=128, 
        type=int, 
        help="Marching cube resolution."
    )

    args = parser.parse_args()

    # Load specifications
    specs_filename = Path(args.experiment) / "specs.json"
    specs = json.load(open(specs_filename))
    T_N = specs["FrameNum"]  # Number of phases

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(**specs["NetworkSpecs"]).cuda()

    # Load model weights
    model_path = Path(args.experiment) / ws.model_params_subdir / f"{args.checkpoint}.pth"
    saved_model_state = torch.load(model_path)
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    # Load preserved phases in the subsequence
    subsequence_filename = Path(args.experiment) / "subsequence.json"
    subsequence = json.load(open(subsequence_filename))
    observed_phases = list(map(int, sorted(subsequence["preserve"], key=lambda x: int(x))))
    missing_phases = [i for i in range(T_N) if i not in observed_phases]

    # Count the train sequences
    train_split_path = Path(args.split).parent / "train.json"
    with open(train_split_path, "r") as f:
        train_split = json.load(f)

    S = len(deep_sdf.dataset.get_instance_filenames(args.data, train_split))
    
    # Load the motion code map of the training sequences and reshape it to (S, T_N * K^m)
    all_c_m_train_path = Path(args.experiment) / ws.latent_codes_subdir / \
                         f"{str(saved_model_epoch)}_cm.pth"
    
    Seq = torch.load(all_c_m_train_path)['latent_codes']['weight'].reshape(S, -1)

    with open(args.split, "r") as f:
        test_split = json.load(f)

    seqfiles = deep_sdf.dataset.get_instance_filenames(args.data, test_split)

    for npz in seqfiles:
        c_m_path = Path(args.experiment) / "SubSeqReconstructions" / \
                   str(saved_model_epoch) / ws.reconstruction_codes_subdir / \
                   f"{Path(npz).name}_cm.pth"
        
        observed_c_m = torch.load(c_m_path)

        c_m = interpolate_complete_motion_code(Seq, observed_phases, observed_c_m, T_N)

        c_s_path = Path(args.experiment) / "SubSeqReconstructions" / \
                   str(saved_model_epoch) / ws.reconstruction_codes_subdir / \
                   f"{Path(npz).name}_cs.pth"
        
        c_s = torch.load(c_s_path)

        # Interpolate and reconstruct phases
        seq_id = Path(npz).name
        output_dir = Path(args.experiment) / "SubSeqReconstructions" / str(saved_model_epoch)
        reconstruct_interpolated_phases(decoder, c_s, c_m, missing_phases, output_dir, args, seq_id, T_N)
