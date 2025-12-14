import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.configs import Config
from networks.model import LLMVS
from utils.generate_summary import generate_summary

def save_frame_indices(args):
    # Setup Config
    kwargs = vars(args)
    config = Config(**kwargs)

    # Load Dataset
    print(f"Loading dataset: {config.dataset}...")
    if 'summe' in config.dataset:
        from utils.summe_dataset import SumMeLLaMADataset, ValBatchCollator
        test_dataset = SumMeLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding=config.pt_path)
    elif 'tvsum' in config.dataset:
        from utils.tvsum_dataset import TVSumLLaMADataset, ValBatchCollator
        test_dataset = TVSumLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding=config.pt_path)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=ValBatchCollator(), pin_memory=True)

    # Load Model
    print(f"Loading model from {config.weights}...")
    if not os.path.exists(config.weights):
        print(f"Error: Checkpoint not found at {config.weights}")
        return

    model = LLMVS.load_from_checkpoint(config.weights, config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    frame_indices_dict = {}

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move inputs to device
            x1 = batch['llama_embedding_userprompt'].squeeze(0).to(device).float()
            x2 = batch['llama_embedding_generation'].squeeze(0).to(device).float()
            mask = batch['mask'].to(device)
            
            x = torch.cat((x1, x2), dim=1)
            
            # Forward pass
            score = model(x, mask=mask).squeeze(1).unsqueeze(0)
            score = score.clamp(0.0, 1.0).squeeze()
            
            # Metadata for summary generation
            cps = batch['change_points'][0]
            n_frames = batch['n_frames']
            nfps = batch['n_frame_per_seg'][0].tolist()
            picks = batch['picks'][0]
            video_name = batch['video_name'][0]

            # Generate binary summary
            # generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack')
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            
            # Convert to numpy and get indices
            summary_np = machine_summary.cpu().numpy()
            indices = np.where(summary_np == 1)[0].tolist()
            
            # Reconstruct frame-level scores (logic from generate_summary.py)
            positions = picks
            # batch['n_frames'] is a list of tensors (batch_size=1)
            n_frames_item = n_frames[0]
            n_frames_val = n_frames_item.item() if isinstance(n_frames_item, torch.Tensor) else n_frames_item
            n_frames_tensor = torch.tensor([n_frames_val], device=device)
            
            frame_scores = torch.zeros((n_frames_val), dtype=torch.float32, device=device)
            if positions.dtype != torch.int32:
                positions = positions.to(torch.int32)
            
            # Ensure positions includes the end frame
            if len(positions) == 0 or positions[-1] != n_frames_val:
                positions = torch.cat([positions, n_frames_tensor.to(positions.device)])
                
            for i in range(len(positions) - 1):
                pos_left, pos_right = positions[i], positions[i+1]
                if i < len(score):
                    frame_scores[pos_left:pos_right] = score[i]
                else:
                    frame_scores[pos_left:pos_right] = 0

            # Get scores for selected frames
            scores = frame_scores[indices].cpu().tolist()
            
            frame_indices_dict[video_name] = {
                "indices": indices,
                "scores": scores
            }

    # Save to JSON
    output_path = args.output_json
    with open(output_path, 'w') as f:
        json.dump(frame_indices_dict, f, indent=4)
    
    print(f"Successfully saved frame indices to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='summe', help='the name of the dataset')
    parser.add_argument('--model', type=str, default='summe_head2_layer3', help='the name of the model')
    parser.add_argument('--tag', type=str, default='summe_split0', help='tag for the experiment')
    parser.add_argument('--split_idx', type=int, default=0, help='the split index')
    parser.add_argument('--reduced_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    # Default to the checkpoint we found
    parser.add_argument('--weights', default='Summaries/best_rho_model/epoch=69-val_sRho=0.228.ckpt', type=str, help='Path to weights')
    parser.add_argument('--pt_path', type=str, default='llama_emb/summe_sum/')
    parser.add_argument('--output_json', type=str, default='frame_indices.json', help='Output JSON file path')

    args = parser.parse_args()
    
    # Adjust paths if running from root but script assumes relative to Colab-LLMVS or vice versa
    # The user seems to be running from root, but paths in code (like 'llama_emb/summe_sum/') are relative.
    # We should ensure we are in the right directory or paths are correct.
    # The default args assume we are inside Colab-LLMVS or paths are relative to it?
    # Let's assume the user runs this from the project root (parent of Colab-LLMVS) and adjusts paths, 
    # OR runs from Colab-LLMVS.
    # Given the previous context, the user is likely at the root.
    # But the default paths in test.py were relative.
    # I will prefix defaults with Colab-LLMVS/ if they don't exist, or let the user handle it.
    # Actually, let's just use the defaults provided and assume the user runs it correctly or we adjust in the command line.
    
    save_frame_indices(args)
