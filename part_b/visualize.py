"""
B3. Vizualizasiya — Training Qrafiklər

Training gedişatını (loss, WER per epoch) qrafik şəklində vizualizasiya edir.
Overfitting-i izləmək üçün train vs validation loss qrafiki çəkir.

İstifadə:
    python visualize.py [--history_path PATH] [--comparison_path PATH]
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")  # GUI olmadan
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# Stil tənzimləmələri
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def plot_training_loss(history, save_path):
    """
    Training və Validation loss qrafiki.
    Overfitting-i izləmək üçün əsas qrafik.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Training loss
    if history.get("train_losses"):
        train_steps = [x["step"] for x in history["train_losses"]]
        train_loss_vals = [x["loss"] for x in history["train_losses"]]
        ax.plot(train_steps, train_loss_vals, "b-", alpha=0.3, linewidth=1, label="Train Loss (step)")

        # Smoothed training loss
        if len(train_loss_vals) > 5:
            window = min(10, len(train_loss_vals) // 3)
            smoothed = np.convolve(train_loss_vals, np.ones(window)/window, mode="valid")
            smooth_steps = train_steps[window-1:]
            ax.plot(smooth_steps, smoothed, "b-", linewidth=2, label="Train Loss (smoothed)")

    # Validation loss
    if history.get("eval_losses"):
        eval_steps = [x["step"] for x in history["eval_losses"]]
        eval_loss_vals = [x["eval_loss"] for x in history["eval_losses"] if x["eval_loss"] is not None]
        eval_steps_filtered = [x["step"] for x in history["eval_losses"] if x["eval_loss"] is not None]

        ax.plot(eval_steps_filtered, eval_loss_vals, "ro-", linewidth=2,
                markersize=8, label="Validation Loss")

        # Overfitting zone annotation
        if len(eval_loss_vals) >= 2:
            min_val_loss = min(eval_loss_vals)
            min_idx = eval_loss_vals.index(min_val_loss)
            ax.axvline(x=eval_steps_filtered[min_idx], color="green", linestyle="--",
                       alpha=0.5, label=f"Ən yaxşı checkpoint (step {eval_steps_filtered[min_idx]})")

            if min_idx < len(eval_loss_vals) - 1:
                # Overfitting zonasını qeyd et
                overfitting_start = eval_steps_filtered[min_idx]
                ax.axvspan(overfitting_start, eval_steps_filtered[-1],
                           alpha=0.1, color="red", label="Potensial Overfitting Zonası")

    ax.set_xlabel("Addım (Step)")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss — Overfitting Monitorinqi")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Loss qrafiki saxlandı: {save_path}")


def plot_wer_per_epoch(history, save_path):
    """
    WER per epoch qrafiki.
    """
    if not history.get("eval_wers"):
        print("[!] WER data tapılmadı, qrafik çəkilmir.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [x["epoch"] for x in history["eval_wers"]]
    wer_vals = [x["eval_wer"] for x in history["eval_wers"]]

    bars = ax.bar(range(len(epochs)), wer_vals, color=sns.color_palette("viridis", len(epochs)),
                  edgecolor="white", linewidth=1.5)
    ax.plot(range(len(epochs)), wer_vals, "ko-", markersize=8, linewidth=2)

    # Min WER-i vurğula
    min_wer = min(wer_vals)
    min_idx = wer_vals.index(min_wer)
    bars[min_idx].set_edgecolor("gold")
    bars[min_idx].set_linewidth(3)
    ax.annotate(
        f"Ən yaxşı: {min_wer:.1f}%",
        xy=(min_idx, min_wer),
        xytext=(min_idx + 0.3, min_wer + max(wer_vals) * 0.05),
        arrowprops=dict(arrowstyle="->", color="gold"),
        fontsize=11, fontweight="bold", color="darkgreen",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER (%)")
    ax.set_title("Validation WER — Epoch-lar üzrə")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"Epoch {e:.0f}" for e in epochs], rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] WER qrafiki saxlandı: {save_path}")


def plot_comparison_bar(comparison, save_path):
    """
    Base vs Fine-Tuned WER/CER müqayisə bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # WER müqayisə
    models = ["Baza Model\n(whisper-small)", "Fine-Tuned\nModel"]
    wer_vals = [comparison["base_wer"], comparison["fine_tuned_wer"]]
    cer_vals = [comparison["base_cer"], comparison["fine_tuned_cer"]]

    colors_wer = ["#e74c3c", "#2ecc71"] if comparison["wer_diff"] < 0 else ["#2ecc71", "#e74c3c"]
    colors_cer = ["#e74c3c", "#2ecc71"] if comparison["cer_diff"] < 0 else ["#2ecc71", "#e74c3c"]

    # WER
    bars1 = axes[0].bar(models, wer_vals, color=colors_wer, edgecolor="white", linewidth=2, width=0.5)
    axes[0].set_ylabel("WER (%)")
    axes[0].set_title("Word Error Rate (WER) Müqayisəsi")
    for bar, val in zip(bars1, wer_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    diff_text = f"Fərq: {comparison['wer_diff']:+.2f}%"
    axes[0].text(0.5, 0.95, diff_text, transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    # CER
    bars2 = axes[1].bar(models, cer_vals, color=colors_cer, edgecolor="white", linewidth=2, width=0.5)
    axes[1].set_ylabel("CER (%)")
    axes[1].set_title("Character Error Rate (CER) Müqayisəsi")
    for bar, val in zip(bars2, cer_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    diff_text = f"Fərq: {comparison['cer_diff']:+.2f}%"
    axes[1].text(0.5, 0.95, diff_text, transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(max(wer_vals), max(cer_vals)) * 1.2)

    plt.suptitle("Model Müqayisəsi: Base vs Fine-Tuned", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Müqayisə qrafiki saxlandı: {save_path}")


def plot_per_sample_comparison(comparison, save_path):
    """
    Nümunə-nümunə WER müqayisəsi scatter plot.
    """
    if not comparison.get("per_sample"):
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    samples = comparison["per_sample"]
    base_wers = [s["base_wer"] for s in samples]
    ft_wers = [s["ft_wer"] for s in samples]

    # Diagonal xətt (heç bir fərq olmadıqda)
    max_val = max(max(base_wers), max(ft_wers)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Fərq yox xətti")

    # Nöqtələr
    improved = [(b, f) for b, f in zip(base_wers, ft_wers) if f < b]
    worsened = [(b, f) for b, f in zip(base_wers, ft_wers) if f > b]
    same = [(b, f) for b, f in zip(base_wers, ft_wers) if f == b]

    if improved:
        ax.scatter(*zip(*improved), c="#2ecc71", s=60, alpha=0.7, label=f"Yaxşılaşma ({len(improved)})", zorder=5)
    if worsened:
        ax.scatter(*zip(*worsened), c="#e74c3c", s=60, alpha=0.7, label=f"Pisləşmə ({len(worsened)})", zorder=5)
    if same:
        ax.scatter(*zip(*same), c="#95a5a6", s=60, alpha=0.7, label=f"Eyni ({len(same)})", zorder=5)

    ax.set_xlabel("Baza Model WER (%)")
    ax.set_ylabel("Fine-Tuned Model WER (%)")
    ax.set_title("Nümunə-Nümunə WER Müqayisəsi")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Annotasiya: xəttin altı = yaxşılaşma
    ax.fill_between([0, max_val], [0, max_val], [0, 0],
                    alpha=0.05, color="green", label="_nolegend_")
    ax.fill_between([0, max_val], [max_val, max_val], [0, max_val],
                    alpha=0.05, color="red", label="_nolegend_")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Scatter plot saxlandı: {save_path}")


def generate_all_plots(results_dir=None):
    """
    Bütün qrafikləri yaradır.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              QRAFİKLƏR YARADILIR                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Training history
    history_path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        plot_training_loss(history, os.path.join(results_dir, "training_loss.png"))
        plot_wer_per_epoch(history, os.path.join(results_dir, "wer_per_epoch.png"))
    else:
        print(f"[!] Training history tapılmadı: {history_path}")

    # Comparison
    comp_path = os.path.join(results_dir, "comparison_results.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)

        plot_comparison_bar(comparison, os.path.join(results_dir, "comparison_bar.png"))
        plot_per_sample_comparison(comparison, os.path.join(results_dir, "comparison_scatter.png"))
    else:
        print(f"[!] Müqayisə nəticələri tapılmadı: {comp_path}")

    print(f"\n[✓] Bütün qrafiklər saxlandı: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qrafik yaradıcı")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()
    generate_all_plots(results_dir=args.results_dir)
