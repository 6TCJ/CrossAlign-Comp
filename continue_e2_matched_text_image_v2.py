#!/usr/bin/env python3
"""
CrossAlign-Comp: step-resumable continuation of matched Text+Image Adapter.

Starting point:
e2_data/e2_matched_clean_full_text_image_best.pt

Features:
1. Default CHECK-ONLY mode.
2. Step-level checkpoint and resume.
3. Restore AdamW optimizer state.
4. Restore TROHN order and Flickr sampler state.
5. Save latest.pt and best checkpoints independently.
6. Clean validation is used for model selection.
7. Original training files are not overwritten.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import train_e2_matched_clean as base


RUN_DIR = base.DATA / "continuation_full_text_image_v2"

SOURCE_CHECKPOINT = (
    base.DATA / "e2_matched_clean_full_text_image_best.pt"
)

DEFAULT_LR = 5e-5
DEFAULT_SAVE_EVERY_STEPS = 100
RUN_VERSION = "2.0.0"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def prepare_data():

    loaded = base.preflight()

    train_cpu, val_cpu, train_cache, dev_cache, _ = loaded

    train_data = base.move_feature_dict(train_cpu)
    val_data = base.move_feature_dict(val_cpu)

    train_images = F.normalize(
        train_cache["image_features"].float(),
        dim=-1
    ).to(base.DEVICE)

    train_texts = F.normalize(
        train_cache["text_features"].float(),
        dim=-1
    ).to(base.DEVICE)

    dev_images = F.normalize(
        dev_cache["image_features"].float(),
        dim=-1
    ).to(base.DEVICE)

    dev_texts = F.normalize(
        dev_cache["text_features"].float(),
        dim=-1
    ).to(base.DEVICE)

    del train_cpu, val_cpu
    del train_cache, dev_cache

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        train_data,
        val_data,
        train_images,
        train_texts,
        dev_images,
        dev_texts
    )


def evaluate_model(
    label,
    image_adapter,
    text_adapter,
    dev_images,
    dev_texts,
    val_data
):

    flickr_i2t, flickr_t2i = base.evaluate_retrieval(
        dev_images,
        dev_texts,
        image_adapter,
        text_adapter
    )

    trohn_i2t, trohn_t2i, trohn_group = base.evaluate_trohn(
        val_data,
        image_adapter,
        text_adapter
    )

    metrics = {
        "flickr_i2t": float(flickr_i2t),
        "flickr_t2i": float(flickr_t2i),
        "trohn_i2t": float(trohn_i2t),
        "trohn_t2i": float(trohn_t2i),
        "trohn_group": float(trohn_group),
    }

    print_metrics(label, metrics)

    return metrics


def print_metrics(label, m):

    print(
        f"{label:30s} | "
        f"Flickr {m['flickr_i2t']:.2%}/"
        f"{m['flickr_t2i']:.2%} | "
        f"TROHN {m['trohn_i2t']:.2%}/"
        f"{m['trohn_t2i']:.2%} | "
        f"Group {m['trohn_group']:.2%}",
        flush=True
    )


def trohn_key(m):

    return (
        m["trohn_group"],
        (m["trohn_i2t"] + m["trohn_t2i"]) / 2.0
    )


def flickr_key(m):

    return (
        m["flickr_i2t"] + m["flickr_t2i"],
        m["trohn_group"]
    )


def inference_payload(
    model_state,
    metrics,
    epoch,
    label,
    selection_metric
):

    return {
        "image_adapter": model_state["image_adapter"],
        "text_adapter": model_state["text_adapter"],
        "alpha": float(base.ALPHA),
        "epoch": int(epoch),
        "variant": "full_text_image_continuation_v2",
        "checkpoint_label": label,
        "metrics_clean_validation": metrics,
        "selection_metric": selection_metric,
    }


def initialize_best_from_baseline(
    source,
    baseline_metrics,
    source_epoch,
    summary_path
):

    summary = {
        "project": "CrossAlign-Comp",
        "run": "matched_full_text_image_continuation_v2",
        "run_version": RUN_VERSION,

        "selection_policy": {
            "primary": "max clean-validation TROHN_Group",
            "tie_break": "max mean(TROHN_I2T, TROHN_T2I)",
            "separate_guardrail": (
                "max Flickr_I2T@1 + Flickr_T2I@1"
            ),
        },

        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_epoch": int(source_epoch),
        "alpha": float(base.ALPHA),
        "learning_rate": DEFAULT_LR,

        "baseline_epoch5": baseline_metrics,

        "best_trohn_group": {
            "checkpoint": "best_trohn_group.pt",
            "epoch": int(source_epoch),
            "metrics": baseline_metrics,
            "selection_key": list(trohn_key(baseline_metrics)),
        },

        "best_flickr": {
            "checkpoint": "best_flickr.pt",
            "epoch": int(source_epoch),
            "metrics": baseline_metrics,
            "selection_key": list(flickr_key(baseline_metrics)),
        },
    }

    state = {
        "image_adapter": source["image_adapter"],
        "text_adapter": source["text_adapter"],
    }

    atomic_torch_save(
        inference_payload(
            state,
            baseline_metrics,
            source_epoch,
            "best_trohn_group_baseline",
            "TROHN_Group primary"
        ),
        RUN_DIR / "best_trohn_group.pt"
    )

    atomic_torch_save(
        inference_payload(
            state,
            baseline_metrics,
            source_epoch,
            "best_flickr_baseline",
            "Flickr primary"
        ),
        RUN_DIR / "best_flickr.pt"
    )

    atomic_json_save(summary, summary_path)

    return summary


def update_best(
    model_state,
    metrics,
    global_epoch,
    summary,
    summary_path
):

    changed = False

    # Best TROHN checkpoint

    if trohn_key(metrics) > tuple(
        summary["best_trohn_group"]["selection_key"]
    ):

        atomic_torch_save(
            inference_payload(
                model_state,
                metrics,
                global_epoch,
                "best_trohn_group",
                "TROHN_Group primary"
            ),
            RUN_DIR / "best_trohn_group.pt"
        )

        summary["best_trohn_group"] = {
            "checkpoint": "best_trohn_group.pt",
            "epoch": int(global_epoch),
            "metrics": metrics,
            "selection_key": list(trohn_key(metrics)),
        }

        print(
            "[BEST TROHN GROUP UPDATED]",
            flush=True
        )

        changed = True

    # Best Flickr checkpoint

    if flickr_key(metrics) > tuple(
        summary["best_flickr"]["selection_key"]
    ):

        atomic_torch_save(
            inference_payload(
                model_state,
                metrics,
                global_epoch,
                "best_flickr",
                "Flickr primary"
            ),
            RUN_DIR / "best_flickr.pt"
        )

        summary["best_flickr"] = {
            "checkpoint": "best_flickr.pt",
            "epoch": int(global_epoch),
            "metrics": metrics,
            "selection_key": list(flickr_key(metrics)),
        }

        print(
            "[BEST FLICKR UPDATED]",
            flush=True
        )

        changed = True

    if changed:
        atomic_json_save(summary, summary_path)


def save_history_row(row, path):

    existing = set()

    if path.exists():

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            for old in csv.DictReader(f):

                value = old.get("global_epoch", "")

                if value:
                    existing.add(int(value))

    if int(row["global_epoch"]) in existing:
        return

    fields = [
        "stage",
        "continuation_epoch",
        "global_epoch",
        "train_loss",
        "base_loss",
        "trohn_loss",
        "flickr_i2t",
        "flickr_t2i",
        "trohn_i2t",
        "trohn_t2i",
        "trohn_group",
        "learning_rate",
        "source_checkpoint",
    ]

    new_file = not path.exists()

    with path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        if new_file:
            writer.writeheader()

        writer.writerow(row)

        f.flush()
        os.fsync(f.fileno())


def sampler_state(sampler):

    return {
        "n_images": int(sampler.n_images),
        "generator_state": sampler.generator.get_state(),
        "order": sampler.order.clone(),
        "cursor": int(sampler.cursor),
    }


def restore_sampler(state):

    sampler = base.FlickrBatchSampler(
        int(state["n_images"]),
        seed=0
    )

    sampler.generator.set_state(
        state["generator_state"].cpu()
    )

    sampler.order = state["order"].cpu().clone()
    sampler.cursor = int(state["cursor"])

    return sampler


def build_checkpoint(
    image_adapter,
    text_adapter,
    optimizer,
    *,
    status,
    completed_epochs,
    source_epoch,
    source_checkpoint,
    source_hash,
    script_hash,
    base_hash,
    learning_rate,
    current_epoch=None,
    step_in_epoch=0,
    trohn_order=None,
    flickr_sampler=None,
    total_loss=0.0,
    total_base=0.0,
    total_trohn=0.0,
):

    payload = {

        "image_adapter": image_adapter.state_dict(),

        "text_adapter": text_adapter.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "alpha": float(base.ALPHA),

        "variant": "full_text_image_continuation_v2",

        "run_version": RUN_VERSION,

        "status": status,

        "completed_epochs": int(completed_epochs),

        "source_epoch": int(source_epoch),

        "source_checkpoint": str(source_checkpoint),

        "source_checkpoint_sha256": source_hash,

        "runner_sha256": script_hash,

        "base_training_script_sha256": base_hash,

        "configured_lr": float(learning_rate),

        "trohn_objective": "paired_text_image_contrastive",

        "trohn_weight": float(base.TROHN_WEIGHT),

        "trohn_batch_size": int(base.TROHN_BATCH_SIZE),

        "flickr_batch_size": int(base.FLICKR_BATCH_SIZE),

        "temperature": float(base.TEMPERATURE),

        "torch_rng_state": torch.get_rng_state(),

        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }

    if status == "in_progress":

        payload.update({

            "epoch_in_progress": int(current_epoch),

            "step_in_epoch": int(step_in_epoch),

            "trohn_order": trohn_order.cpu().clone(),

            "flickr_sampler_state": sampler_state(flickr_sampler),

            "total_loss": float(total_loss),

            "total_base": float(total_base),

            "total_trohn": float(total_trohn),
        })

    return payload


def main():

    parser = argparse.ArgumentParser(
        description="Safe matched Text+Image Adapter continuation."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually start training."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None
    )

    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=DEFAULT_SAVE_EVERY_STEPS
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.save_every_steps < 1:
        raise ValueError("Invalid training parameters.")

    if args.resume and not args.train:
        raise ValueError("Use --train --resume.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    script_path = Path(__file__).resolve()
    base_path = Path(base.__file__).resolve()

    script_hash = sha256_file(script_path)
    base_hash = sha256_file(base_path)

    latest_path = RUN_DIR / "latest.pt"
    history_path = RUN_DIR / "history.csv"
    summary_path = RUN_DIR / "metrics_summary.json"

    print("=" * 72)
    print("CrossAlign Text+Image Adapter continuation V2")
    print("=" * 72)

    print("Project root:", base.ROOT)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Run directory:", RUN_DIR)
    print("Alpha:", base.ALPHA)
    print("TROHN weight:", base.TROHN_WEIGHT)

    if args.resume:

        if not latest_path.exists():
            raise FileNotFoundError(latest_path)

        resume_ckpt = base.load_pt(latest_path)

        if resume_ckpt.get("run_version") != RUN_VERSION:
            raise RuntimeError("Version mismatch.")

        if resume_ckpt.get("runner_sha256") != script_hash:
            raise RuntimeError("Training script changed.")

        if resume_ckpt.get("base_training_script_sha256") != base_hash:
            raise RuntimeError("Base training script changed.")

        if (
            not SOURCE_CHECKPOINT.exists()
            or sha256_file(SOURCE_CHECKPOINT)
            != resume_ckpt.get("source_checkpoint_sha256")
        ):
            raise RuntimeError("Source checkpoint changed or missing.")

        learning_rate = float(
            resume_ckpt["configured_lr"]
        )

        if (
            args.lr is not None
            and abs(args.lr - learning_rate) > 1e-15
        ):
            raise RuntimeError("Cannot change LR during exact resume.")

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    else:

        resume_ckpt = None

        learning_rate = (
            DEFAULT_LR if args.lr is None else float(args.lr)
        )

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if args.train and RUN_DIR.exists() and any(RUN_DIR.iterdir()):
            raise FileExistsError(
                "Run already exists. Use --train --resume."
            )

    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    source_hash = sha256_file(SOURCE_CHECKPOINT)

    (
        train_data,
        val_data,
        train_images,
        train_texts,
        dev_images,
        dev_texts
    ) = prepare_data()

    print(
        "Clean validation rows:",
        len(val_data["image_features"])
    )

    # CHECK-ONLY MODE

    if not args.train:

        source = base.load_pt(SOURCE_CHECKPOINT)

        print("Source checkpoint:", SOURCE_CHECKPOINT)
        print("Source epoch:", source.get("epoch"))
        print("Variant:", source.get("variant"))
        print("Alpha:", source.get("alpha"))

        print("\nCHECK-ONLY COMPLETE.")
        print("No model weights modified.")

        return

    # LOAD MODEL

    if args.resume:

        checkpoint = resume_ckpt

        image_adapter, text_adapter = base.build_adapters(
            checkpoint
        )

        optimizer = torch.optim.AdamW(
            list(image_adapter.parameters())
            + list(text_adapter.parameters()),
            lr=learning_rate
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        completed_epochs = int(
            checkpoint["completed_epochs"]
        )

        source_epoch = int(
            checkpoint["source_epoch"]
        )

        source_checkpoint = checkpoint["source_checkpoint"]

        print("Optimizer restored.")

    else:

        source = base.load_pt(SOURCE_CHECKPOINT)

        if source.get("variant") != "full_text_image":
            raise RuntimeError("Unexpected checkpoint variant.")

        if (
            source.get("alpha") is None
            or abs(float(source["alpha"]) - base.ALPHA) > 1e-10
        ):
            raise RuntimeError("Alpha mismatch.")

        source_epoch = int(source.get("epoch", 5))

        source_checkpoint = str(SOURCE_CHECKPOINT)

        image_adapter, text_adapter = base.build_adapters(
            source
        )

        optimizer = torch.optim.AdamW(
            list(image_adapter.parameters())
            + list(text_adapter.parameters()),
            lr=learning_rate
        )

        print(
            "Initialized from epoch-5 checkpoint.",
            "Fresh AdamW LR:",
            learning_rate
        )

        baseline_metrics = evaluate_model(
            "Original epoch-5 baseline",
            image_adapter,
            text_adapter,
            dev_images,
            dev_texts,
            val_data
        )

        summary = initialize_best_from_baseline(
            source,
            baseline_metrics,
            source_epoch,
            summary_path
        )

        summary["learning_rate"] = learning_rate

        summary["clean_validation_rows"] = int(
            len(val_data["image_features"])
        )

        summary["runner_sha256"] = script_hash
        summary["base_training_script_sha256"] = base_hash
        summary["source_checkpoint_sha256"] = source_hash

        atomic_json_save(summary, summary_path)

        save_history_row(
            {
                "stage": "baseline_epoch5",
                "continuation_epoch": 0,
                "global_epoch": source_epoch,
                "train_loss": "",
                "base_loss": "",
                "trohn_loss": "",
                **baseline_metrics,
                "learning_rate": "",
                "source_checkpoint": source_checkpoint,
            },
            history_path
        )

        del source

    # RESUME STATE

    if (
        args.resume
        and completed_epochs >= args.epochs
        and resume_ckpt.get("status") == "epoch_complete"
    ):

        print("Requested epochs already completed.")
        return

    n_trohn = len(train_data["image_features"])

    steps_per_epoch = math.ceil(
        n_trohn / base.TROHN_BATCH_SIZE
    )

    if args.resume and resume_ckpt.get("status") == "in_progress":

        current_epoch = int(
            resume_ckpt["epoch_in_progress"]
        )

        start_step = int(
            resume_ckpt["step_in_epoch"]
        ) + 1

        trohn_order = resume_ckpt["trohn_order"].cpu().clone()

        flickr_sampler = restore_sampler(
            resume_ckpt["flickr_sampler_state"]
        )

        total_loss = float(resume_ckpt["total_loss"])
        total_base = float(resume_ckpt["total_base"])
        total_trohn = float(resume_ckpt["total_trohn"])

        torch.set_rng_state(
            resume_ckpt["torch_rng_state"].cpu()
        )

        torch.cuda.set_rng_state_all(
            [
                x.cpu()
                for x in resume_ckpt["cuda_rng_state_all"]
            ]
        )

        print(
            f"Resuming epoch {current_epoch}, "
            f"step {start_step}"
        )

    else:

        current_epoch = (
            completed_epochs + 1 if args.resume else 1
        )

        start_step = 1

        trohn_order = None
        flickr_sampler = None

        total_loss = 0.0
        total_base = 0.0
        total_trohn = 0.0

    # TRAINING LOOP

    while current_epoch <= args.epochs:

        if start_step == 1:

            epoch_seed = base.SEED + 50000 + current_epoch

            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)

            trohn_generator = torch.Generator().manual_seed(
                epoch_seed + 1
            )

            trohn_order = torch.randperm(
                n_trohn,
                generator=trohn_generator
            )

            flickr_sampler = base.FlickrBatchSampler(
                len(train_images),
                seed=epoch_seed + 2
            )

            total_loss = 0.0
            total_base = 0.0
            total_trohn = 0.0

            initial = build_checkpoint(
                image_adapter,
                text_adapter,
                optimizer,
                status="in_progress",
                completed_epochs=current_epoch - 1,
                source_epoch=source_epoch,
                source_checkpoint=source_checkpoint,
                source_hash=source_hash,
                script_hash=script_hash,
                base_hash=base_hash,
                learning_rate=learning_rate,
                current_epoch=current_epoch,
                step_in_epoch=0,
                trohn_order=trohn_order,
                flickr_sampler=flickr_sampler
            )

            atomic_torch_save(initial, latest_path)

        image_adapter.train()
        text_adapter.train()

        print(
            f"\nEpoch {current_epoch}/{args.epochs} "
            f"(global epoch {source_epoch + current_epoch})",
            flush=True
        )

        for step, start in enumerate(
            range(0, n_trohn, base.TROHN_BATCH_SIZE),
            start=1
        ):

            if step < start_step:
                continue

            trohn_ids = trohn_order[
                start:start + base.TROHN_BATCH_SIZE
            ].to(base.DEVICE)

            flickr_ids_cpu, caption_ids_cpu = (
                flickr_sampler.next(base.FLICKR_BATCH_SIZE)
            )

            flickr_ids = flickr_ids_cpu.to(base.DEVICE)
            caption_ids = caption_ids_cpu.to(base.DEVICE)

            # Flickr contrastive loss

            base_image_f = base.adapt(
                train_images[flickr_ids],
                image_adapter
            )

            base_text_f = base.adapt(
                train_texts[caption_ids],
                text_adapter
            )

            base_loss = base.contrastive_loss(
                base_image_f,
                base_text_f
            )

            # TROHN paired image-text loss

            i0 = base.adapt(
                train_data["image_features"][trohn_ids],
                image_adapter
            )

            i1 = base.adapt(
                train_data["negative_image_features"][trohn_ids],
                image_adapter
            )

            t0 = base.adapt(
                train_data["positive_features"][trohn_ids],
                text_adapter
            )

            t1 = base.adapt(
                train_data["negative_features"][trohn_ids],
                text_adapter
            )

            paired_images = torch.cat(
                [i0, i1],
                dim=0
            )

            paired_texts = torch.cat(
                [t0, t1],
                dim=0
            )

            trohn_loss = base.contrastive_loss(
                paired_images,
                paired_texts
            )

            loss = (
                base_loss
                + base.TROHN_WEIGHT * trohn_loss
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            total_loss += float(loss.item())
            total_base += float(base_loss.item())
            total_trohn += float(trohn_loss.item())

            # STEP-LEVEL CHECKPOINT

            if (
                step % args.save_every_steps == 0
                or step == steps_per_epoch
            ):

                progress = build_checkpoint(
                    image_adapter,
                    text_adapter,
                    optimizer,
                    status="in_progress",
                    completed_epochs=current_epoch - 1,
                    source_epoch=source_epoch,
                    source_checkpoint=source_checkpoint,
                    source_hash=source_hash,
                    script_hash=script_hash,
                    base_hash=base_hash,
                    learning_rate=learning_rate,
                    current_epoch=current_epoch,
                    step_in_epoch=step,
                    trohn_order=trohn_order,
                    flickr_sampler=flickr_sampler,
                    total_loss=total_loss,
                    total_base=total_base,
                    total_trohn=total_trohn
                )

                atomic_torch_save(
                    progress,
                    latest_path
                )

            if step % 200 == 0 or step == steps_per_epoch:

                print(
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss={total_loss / step:.4f} | "
                    f"Base={total_base / step:.4f} | "
                    f"TROHN={total_trohn / step:.4f}",
                    flush=True
                )

        # EPOCH EVALUATION

        metrics = evaluate_model(
            f"Continuation epoch {source_epoch + current_epoch}",
            image_adapter,
            text_adapter,
            dev_images,
            dev_texts,
            val_data
        )

        global_epoch = source_epoch + current_epoch

        epoch_result = {

            "stage": "continuation",

            "continuation_epoch": current_epoch,

            "global_epoch": global_epoch,

            "train_loss": total_loss / steps_per_epoch,

            "base_loss": total_base / steps_per_epoch,

            "trohn_loss": total_trohn / steps_per_epoch,

            **metrics,

            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),

            "source_checkpoint": source_checkpoint,
        }

        model_state = {

            "image_adapter": image_adapter.state_dict(),

            "text_adapter": text_adapter.state_dict(),
        }

        # Update BEST before marking epoch complete.

        update_best(
            model_state,
            metrics,
            global_epoch,
            summary,
            summary_path
        )

        save_history_row(
            epoch_result,
            history_path
        )

        # EPOCH CHECKPOINT

        epoch_ckpt = build_checkpoint(
            image_adapter,
            text_adapter,
            optimizer,
            status="epoch_complete",
            completed_epochs=current_epoch,
            source_epoch=source_epoch,
            source_checkpoint=source_checkpoint,
            source_hash=source_hash,
            script_hash=script_hash,
            base_hash=base_hash,
            learning_rate=learning_rate
        )

        epoch_ckpt.update({

            "epoch": global_epoch,

            "metrics_clean_validation": metrics,

            "train_loss": epoch_result["train_loss"],

            "base_loss": epoch_result["base_loss"],

            "trohn_loss": epoch_result["trohn_loss"],
        })

        epoch_path = RUN_DIR / (
            f"epoch_{current_epoch:02d}.pt"
        )

        atomic_torch_save(
            epoch_ckpt,
            epoch_path
        )

        atomic_torch_save(
            epoch_ckpt,
            latest_path
        )

        print("[SAVED EPOCH]", epoch_path, flush=True)

        current_epoch += 1
        start_step = 1

        trohn_order = None
        flickr_sampler = None

        total_loss = 0.0
        total_base = 0.0
        total_trohn = 0.0

    print("\n" + "=" * 72)
    print("CONTINUATION COMPLETE")
    print("=" * 72)

    print("History:", history_path)
    print("Latest:", latest_path)

    print(
        "Best TROHN:",
        RUN_DIR / "best_trohn_group.pt"
    )

    print(
        "Best Flickr:",
        RUN_DIR / "best_flickr.pt"
    )

    print("Summary:", summary_path)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()