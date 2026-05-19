import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
#  Load Best Model
# ─────────────────────────────────────────────

def load_best_model(model, save_path: str = "best_model.pth", device=None):
    """
    Loads the best saved model weights.

    Args:
        model     : UNet instance (already initialized)
        save_path : path to saved .pth file
        device    : 'cuda' or 'cpu'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {save_path}")
    print(f"   Running on: {device}")
    return model


# ─────────────────────────────────────────────
#  Final Evaluation on Test Set
# ─────────────────────────────────────────────

def evaluate_model(model, test_loader, device=None):
    """
    Runs full evaluation on the test set.
    Computes:
        - Pixel Accuracy
        - IoU per class (background + foreground) + mean IoU
        - Dice Coefficient

    Returns a results dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    total_acc   = 0.0
    total_miou  = {"iou_background": 0.0, "iou_foreground": 0.0, "miou": 0.0}
    total_dice  = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)           # [B, 1, H, W] — probabilities (sigmoid applied)

            total_acc += pixel_accuracy(preds, masks)

            batch_miou = mean_iou(preds, masks)
            total_miou["iou_background"] += batch_miou["iou_background"]
            total_miou["iou_foreground"] += batch_miou["iou_foreground"]
            total_miou["miou"]           += batch_miou["miou"]

            total_dice += dice_score(preds, masks)

    n = len(test_loader)
    results = {
        "pixel_accuracy":  total_acc  / n,
        "iou_background":  total_miou["iou_background"] / n,
        "iou_foreground":  total_miou["iou_foreground"] / n,
        "mean_iou":        total_miou["miou"] / n,
        "dice_coefficient": total_dice / n,
    }

    return results


def print_results(results: dict):
    """Prints evaluation results in a clean table format."""
    print("\n" + "═" * 45)
    print("       EVALUATION RESULTS — TEST SET")
    print("═" * 45)
    print(f"  Pixel Accuracy       : {results['pixel_accuracy']:.4f}  ({results['pixel_accuracy']*100:.2f}%)")
    print(f"  IoU — Background     : {results['iou_background']:.4f}")
    print(f"  IoU — Foreground     : {results['iou_foreground']:.4f}")
    print(f"  Mean IoU (mIoU)      : {results['mean_iou']:.4f}")
    print(f"  Dice Coefficient     : {results['dice_coefficient']:.4f}")
    print("═" * 45)


# ─────────────────────────────────────────────
#  Qualitative Visualisation
# ─────────────────────────────────────────────

def visualize_predictions(model, test_loader, device=None, num_samples: int = 6,
                           save_path: str = "predictions.png"):
    """
    Visualises model predictions vs ground truth on random test samples.
    Each row shows: Input Image | Ground Truth Mask | Predicted Mask | Overlay

    Args:
        model       : trained UNet
        test_loader : test DataLoader
        num_samples : number of samples to visualise
        save_path   : where to save the figure
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Collect one batch
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks  = masks.to(device)

    with torch.no_grad():
        preds = model(images)                              # [B, 1, H, W]
        preds_binary = (preds.squeeze(1) > 0.5).long()    # [B, H, W]

    # Move to CPU for plotting
    images       = images.cpu()
    masks        = masks.cpu()
    preds_binary = preds_binary.cpu()

    num_samples = min(num_samples, images.shape[0])

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))

    # ImageNet denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    col_titles = ["Input Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(num_samples):
        # Denormalize image for display
        img_vis = images[i] * std + mean
        img_vis = img_vis.permute(1, 2, 0).clamp(0, 1).numpy()

        gt_mask   = masks[i].numpy()
        pred_mask = preds_binary[i].numpy()

        # Col 0 — Input image
        axes[i, 0].imshow(img_vis)
        axes[i, 0].axis("off")

        # Col 1 — Ground truth mask
        axes[i, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

        # Col 2 — Predicted mask
        axes[i, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].axis("off")

        # Col 3 — Overlay (image + prediction + ground truth difference)
        overlay = img_vis.copy()
        # Green  = correct foreground (TP)
        # Red    = missed foreground  (FN)
        # Blue   = false positive     (FP)
        tp = (pred_mask == 1) & (gt_mask == 1)
        fn = (pred_mask == 0) & (gt_mask == 1)
        fp = (pred_mask == 1) & (gt_mask == 0)

        overlay[tp] = overlay[tp] * 0.5 + np.array([0.0, 1.0, 0.0]) * 0.5   # green
        overlay[fn] = overlay[fn] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5   # red
        overlay[fp] = overlay[fp] * 0.5 + np.array([0.0, 0.0, 1.0]) * 0.5   # blue

        axes[i, 3].imshow(overlay)
        axes[i, 3].axis("off")

        # Per-sample IoU label
        sample_iou = mean_iou(
            preds[i:i+1].cpu(), masks[i:i+1].cpu()
        )["miou"]
        axes[i, 0].set_ylabel(f"mIoU: {sample_iou:.3f}", fontsize=10, rotation=0,
                               labelpad=60, va="center")

    # Legend for overlay
    legend_elements = [
        mpatches.Patch(color="green", label="True Positive (correct pet)"),
        mpatches.Patch(color="red",   label="False Negative (missed pet)"),
        mpatches.Patch(color="blue",  label="False Positive (wrong background)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("U-Net Predictions — Oxford-IIIT Pet Dataset", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved predictions to {save_path}")


# ─────────────────────────────────────────────
#  Plot Training Curves
# ─────────────────────────────────────────────

def plot_history(history: dict, save_path: str = "training_curves.png"):
    """
    Plots loss, pixel accuracy, IoU per class, and Dice curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0, 0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Pixel Accuracy
    axes[0, 1].plot(epochs, history["train_acc"], label="Train Acc", color="royalblue")
    axes[0, 1].plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato")
    axes[0, 1].set_title("Pixel Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # IoU per class + mean
    axes[1, 0].plot(epochs, history["train_iou_bg"], label="Train IoU bg", color="royalblue", linestyle="dashed")
    axes[1, 0].plot(epochs, history["train_iou_fg"], label="Train IoU fg", color="royalblue")
    axes[1, 0].plot(epochs, history["train_miou"],   label="Train mIoU",   color="royalblue", linestyle="dotted")
    axes[1, 0].plot(epochs, history["val_iou_bg"],   label="Val IoU bg",   color="tomato",    linestyle="dashed")
    axes[1, 0].plot(epochs, history["val_iou_fg"],   label="Val IoU fg",   color="tomato")
    axes[1, 0].plot(epochs, history["val_miou"],     label="Val mIoU",     color="tomato",    linestyle="dotted")
    axes[1, 0].set_title("IoU per Class + Mean IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True)

    # Dice
    axes[1, 1].plot(epochs, history["train_dice"], label="Train Dice", color="royalblue")
    axes[1, 1].plot(epochs, history["val_dice"],   label="Val Dice",   color="tomato")
    axes[1, 1].set_title("Dice Coefficient")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle("U-Net Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved training curves to {save_path}")

