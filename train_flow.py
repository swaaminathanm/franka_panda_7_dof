import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from policy.dataset import FrankaLeRobotDataset
from policy.flow_matching import FlowMatchingPolicy


def set_seed(seed: int = 42):
    """Sets random seeds across libraries for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """Main training routine for Flow Matching Policy."""
    set_seed(args.seed)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using execution device: {device}")

    # 2. Initialize Dataset & DataLoader
    dataset = FrankaLeRobotDataset(
        data_dir=args.data_dir,
        pred_horizon=args.pred_horizon,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Initialize Flow Matching Policy
    policy = FlowMatchingPolicy(
        action_dim=4,
        state_dim=30,
        pred_horizon=args.pred_horizon,
        cond_dim=args.cond_dim,
        stats_path=os.path.join(args.data_dir, "meta", "stats.json"),
    ).to(device)

    # 4. Initialize Optimizer & Cosine Annealing Learning Rate Scheduler
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Checkpoint directory setup
    os.makedirs(args.save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(args.save_dir, "flow_policy_best.pt")
    latest_checkpoint_path = os.path.join(args.save_dir, "flow_policy_latest.pt")

    best_loss = float("inf")
    start_time = time.time()

    print(f"\n[Train] Starting Flow Matching Policy Training...")
    print(f"        Total Epochs:  {args.epochs}")
    print(f"        Batch Size:    {args.batch_size}")
    print(f"        Learning Rate: {args.lr}")
    print(f"        Save Directory: {args.save_dir}\n")

    # 6. Main Training Loop
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            state = batch["state"].to(device)    # (Batch_Size, 30)
            action = batch["action"].to(device)  # (Batch_Size, 16, 4)

            optimizer.zero_grad()

            # Compute OT-CFM velocity matching MSE loss
            loss = policy.compute_loss(state, action)

            # Backpropagation & Gradient Clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(state)
            num_batches += 1

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        current_lr = scheduler.get_last_lr()[0]

        # Checkpoint saving
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

        # Log epoch progress
        best_tag = " [BEST]" if is_best else ""
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:3d}/{args.epochs:3d}] | "
                f"Loss: {epoch_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s{best_tag}"
            )

    # Save final latest checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "args": vars(args),
        },
        latest_checkpoint_path,
    )

    total_time = time.time() - start_time
    print(f"\n[Train] Training Completed in {total_time:.1f}s!")
    print(f"[Train] Best Loss: {best_loss:.6f}")
    print(f"[Train] Best checkpoint saved to:   {best_checkpoint_path}")
    print(f"[Train] Latest checkpoint saved to: {latest_checkpoint_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Flow Matching Policy for Franka Panda 7-DoF")
    parser.add_argument("--data-dir", type=str, default="data/lerobot", help="Path to LeRobot dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--pred-horizon", type=int, default=16, help="Action prediction horizon (default: 16)")
    parser.add_argument("--cond-dim", type=int, default=256, help="Condition embedding dimension (default: 256)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-interval", type=int, default=5, help="Epoch logging interval (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
