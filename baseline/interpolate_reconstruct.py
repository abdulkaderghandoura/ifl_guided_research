import torch
import numpy as np
import argparse
import json

from pathlib import Path

import deep_sdf
import deep_sdf.workspace as ws

def interpolate_latent_codes(c_m, num_phases):
    """
    Interpolates the latent codes between two phases: ED (00) and ES (01)
    Arguments:
    c_m -- latent motion vectors of shape (frame_num, Cm_size)
    num_phases -- number of intermediate phases to interpolate (k)

    Returns:
    interpolated_cms -- list of interpolated latent codes for the k intermediate phases
    """
    c_m_ed = c_m[0]  # ED phase (start)
    c_m_es = c_m[1]  # ES phase (end)
    
    # Linear interpolation for k intermediate phases
    interpolated_cms = []
    for i in range(1, num_phases + 1):
        alpha = i / (num_phases + 1)  # fraction for interpolation
        interpolated_cm = (1 - alpha) * c_m_ed + alpha * c_m_es
        interpolated_cms.append(interpolated_cm)
    
    return interpolated_cms

def reconstruct_interpolated_phases(decoder, c_s, c_m, num_interpolations, output_dir, args, seq_id):
    """
    Reconstruct the k interpolated phases and save meshes.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    interpolated_cms = interpolate_latent_codes(c_m, num_interpolations)
    
    for idx, interpolated_cm in enumerate(interpolated_cms, start=1):
        phase = torch.FloatTensor([idx / (num_interpolations + 1)]).unsqueeze(0)
        
        mesh_file_name = f"{seq_id}_{idx:02d}".format(idx)
        mesh_file_path = Path(output_dir) / ws.reconstruction_meshes_subdir / mesh_file_name
        mesh_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        motion_file_name = f"{seq_id}_{idx:02d}".format(idx)
        motion_file_path = Path(output_dir) / ws.reconstruction_motions_subdir / motion_file_name
        motion_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Reconstruct: {seq_id}_{idx:02d}".format(idx))
        
        with torch.no_grad():
            deep_sdf.mesh.create_mesh_4dsdf(
                decoder, c_s, interpolated_cm.unsqueeze(0), phase,
                str(mesh_file_path), str(motion_file_path), N=args.resolution, max_batch=int(2 ** 17)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate and reconstruct intermediate phases between ED and ES."
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

    parser.add_argument(
        "--num_interpolations", 
        "-k", 
        type=int, 
        required=True, 
        help="Number of intermediate phases to reconstruct."
    )

    args = parser.parse_args()
    assert args.num_interpolations // 100 == 0, 'Too many phases!'

    # Load specifications and model
    specs_filename = Path(args.experiment) / "specs.json"
    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(**specs["NetworkSpecs"]).cuda()

    # Load model weights
    model_path = Path(args.experiment) / ws.model_params_subdir / f"{args.checkpoint}.pth"
    saved_model_state = torch.load(model_path)
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    with open(args.split, "r") as f:
        split = json.load(f)

    seqfiles = deep_sdf.dataset.get_instance_filenames(args.data, split)

    for npz in seqfiles:
        phase_list = sorted(Path(npz).iterdir(), key=lambda x:int(x.stem))

        # print(f"Debugging {Path(npz).name}_00")
        # data = np.load(phase_list[0])
        # print("Ti:", data['Ti'])
        # print("Offset:", data['offset'])
        # print("Scale:", data['scale'])
        
        # c_s = torch.ones(1, specs["CsLength"]).normal_(mean=0, std=0.1).cuda()  # [1, Cs_size]
        # c_m = torch.ones(len(phase_list), specs["CmLength"]).normal_(mean=0, std=1.0 / np.sqrt(specs["CmLength"])).cuda()

        c_s_path = Path(args.experiment) / ws.reconstructions_subdir / \
                   str(saved_model_epoch) / ws.reconstruction_codes_subdir / \
                   f"{Path(npz).name}_cs.pth"
        
        c_s = torch.load(c_s_path)

        c_m_path = Path(args.experiment) / ws.reconstructions_subdir / \
                   str(saved_model_epoch) / ws.reconstruction_codes_subdir / \
                   f"{Path(npz).name}_cm.pth"
        
        c_m = torch.load(c_m_path)

        # Interpolate and reconstruct phases
        output_dir = Path(args.experiment) / "Interpolation" / str(saved_model_state["epoch"])
        reconstruct_interpolated_phases(decoder, c_s, c_m, args.num_interpolations, output_dir, args, Path(npz).name)
