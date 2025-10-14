"""
1D CNN training script for affective prediction
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

from dataset import create_dataloaders
from models import get_model


def train_epoch(model, train_loader, criterion, optimizer, device, target_type="both"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward
        if target_type == "both":
            # ������
            arousal_out, valence_out = model(data)
            arousal_target = target[:, 0]
            valence_target = target[:, 1]
            loss = criterion(arousal_out, arousal_target) + criterion(valence_out, valence_target)

            # Accuracy
            arousal_pred = arousal_out.argmax(dim=1)
            valence_pred = valence_out.argmax(dim=1)
            correct += (arousal_pred == arousal_target).sum().item()
            correct += (valence_pred == valence_target).sum().item()
            total += arousal_target.size(0) * 2
        else:
            # ��뿹�
            output = model(data)
            target = target.squeeze(1)
            loss = criterion(output, target)

            # Accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, target_type="both"):
    """<"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data = data.to(device)
            target = target.to(device)

            # Forward
            if target_type == "both":
                # ������
                arousal_out, valence_out = model(data)
                arousal_target = target[:, 0]
                valence_target = target[:, 1]
                loss = criterion(arousal_out, arousal_target) + criterion(valence_out, valence_target)

                # Accuracy
                arousal_pred = arousal_out.argmax(dim=1)
                valence_pred = valence_out.argmax(dim=1)
                correct += (arousal_pred == arousal_target).sum().item()
                correct += (valence_pred == valence_target).sum().item()
                total += arousal_target.size(0) * 2
            else:
                # ��뿹�
                output = model(data)
                target = target.squeeze(1)
                loss = criterion(output, target)

                # Accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # multitaskモデルの場合はtarget_type="both"にする
    target_type = "both" if args.model == "multitask" else args.target_type

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        search_conditions={"ex-term": args.term},
        target_type=target_type,
        use_eda=args.use_eda,
        use_eda_masked=args.use_eda_masked,
        use_scr=args.use_scr,
        use_scl=args.use_scl,
        use_acc_x=args.use_acc_x,
        use_acc_y=args.use_acc_y,
        use_acc_z=args.use_acc_z,
        use_acc_mag=args.use_acc_mag,
        use_temp=args.use_temp,
        use_rri=args.use_rri
    )

    # チャンネル数カウント（シンプルな加算）
    in_channels = 0
    if args.use_eda:
        in_channels += 1
    if args.use_eda_masked:
        in_channels += 1
    if args.use_scr:
        in_channels += 1
    if args.use_scl:
        in_channels += 1
    if args.use_acc_x:
        in_channels += 1
    if args.use_acc_y:
        in_channels += 1
    if args.use_acc_z:
        in_channels += 1
    if args.use_acc_mag:
        in_channels += 1
    if args.use_temp:
        in_channels += 1
    if args.use_rri:
        in_channels += 1

    print(f"Creating model: {args.model}")
    model = get_model(
        args.model,
        in_channels=in_channels,
        num_classes=9,
        dropout=args.dropout
    )
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.model}_{timestamp}")
    writer = SummaryWriter(log_dir)

    # �Xǣ���
    save_dir = os.path.join(args.save_dir, f"{args.model}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # -���X
    config = vars(args)
    config['in_channels'] = in_channels
    config['timestamp'] = timestamp
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, target_type
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, target_type
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # ��������
        scheduler.step(val_loss)

        # ��ïݤ���X
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, final_path)
    print(f"Saved final model to {final_path}")

    # ƹ�
    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60)

    # ٹ�������
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device, target_type)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # P���X
    results = {
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\nTraining completed. Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D CNN for affective prediction")

    # ����#
    parser.add_argument("--term", type=str, default="term1", help="Data term (term1/term2/term3)")
    parser.add_argument("--target_type", type=str, default="arousal",
                        choices=["arousal", "valence", "both"], help="Target type")

    # EDA関連
    parser.add_argument("--use_eda", type=int, default=0, help="Use raw EDA data")
    parser.add_argument("--use_eda_masked", type=int, default=0, help="Use masked EDA data")
    parser.add_argument("--use_scr", type=int, default=0, help="Use SCR (Phasic) data")
    parser.add_argument("--use_scl", type=int, default=0, help="Use SCL (Tonic) data")

    # 加速度関連
    parser.add_argument("--use_acc_x", type=int, default=0, help="Use accelerometer X-axis")
    parser.add_argument("--use_acc_y", type=int, default=0, help="Use accelerometer Y-axis")
    parser.add_argument("--use_acc_z", type=int, default=0, help="Use accelerometer Z-axis")
    parser.add_argument("--use_acc_mag", type=int, default=1, help="Use accelerometer magnitude")

    # その他
    parser.add_argument("--use_temp", type=int, default=1, help="Use temperature data")
    parser.add_argument("--use_rri", type=int, default=1, help="Use RRI data")

    # ���#
    parser.add_argument("--model", type=str, default="simple",
                        choices=["simple", "resnet", "deep", "multitask"], help="Model type")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # ]n�
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")

    args = parser.parse_args()

    main(args)
