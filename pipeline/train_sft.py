"""LoRA supervised fine-tuning on Tinker: one epoch over runs/<name>/train.jsonl.

Usage:
    python -m pipeline.train_sft --name smoke --no_publish
    python -m pipeline.train_sft --name v1
"""

import argparse
import json
import time

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import hyperparam_utils, model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from pipeline import jsonl
from pipeline.config import MAX_TRAIN_TOKENS, RUNS, STUDENT

SAVE_EVERY = 250
STATE_TTL_SECONDS = 7 * 24 * 3600


def lr_at(step: int, total: int, base_lr: float) -> float:
    return base_lr * (1.0 - step / total)


def batches(n: int, batch_size: int) -> list[range]:
    return [range(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_publish", action="store_true")
    return p.parse_args(argv)


def load_train_rows(run_dir) -> list[dict]:
    """Rows of runs/<name>/train.jsonl; a missing or empty file is an error, raised before any client exists."""
    path = run_dir / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Build it first with pipeline.select --name {run_dir.name}.")
    rows = jsonl.read(path)
    if not rows:
        raise ValueError(f"{path} has no training examples. Rebuild it with pipeline.select.")
    return rows


def build_renderer():
    tokenizer = get_tokenizer(STUDENT)
    return renderers.get_renderer(model_info.get_recommended_renderer_name(STUDENT), tokenizer)


def render_examples(rows: list[dict], renderer) -> list:
    """Render untruncated (max_length=None); over-long examples are rejected, never cut."""
    data = [
        conversation_to_datum(r["messages"], renderer, max_length=None,
                              train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES)
        for r in rows
    ]
    over = [(r.get("id", i), len(d.model_input.to_ints()))
            for i, (r, d) in enumerate(zip(rows, data))
            if len(d.model_input.to_ints()) > MAX_TRAIN_TOKENS]
    if over:
        shown = ", ".join(f"{i} ({n} tokens)" for i, n in over[:5])
        raise ValueError(
            f"{len(over)} of {len(rows)} training examples exceed {MAX_TRAIN_TOKENS} tokens "
            f"(first: {shown}). Truncation would cut off the answer and the stop sequence. "
            "Rebuild the training file with pipeline.select, which drops over-long examples.")
    return data


def count_tokens(data: list) -> int:
    return sum(len(d.model_input.to_ints()) for d in data)


def batch_loss(loss_fn_outputs, batch_inputs) -> float:
    """Mean negative log-likelihood over supervised (weight > 0) tokens of one batch."""
    logprobs = np.concatenate([o["logprobs"].tolist() for o in loss_fn_outputs])
    weights = np.concatenate([w["weights"].tolist() for w in batch_inputs])
    return float(-np.dot(logprobs, weights) / max(weights.sum(), 1e-9))


def write_checkpoint_info(run_dir, info: dict, published: bool) -> None:
    (run_dir / "checkpoint.json").write_text(json.dumps({**info, "published": published}, indent=2))


def save_and_publish(client, publish, run_dir, name: str, info: dict, no_publish: bool) -> dict:
    """Save sampler weights, record the path at once, then publish (checkpoint.json updated after)."""
    checkpoint = client.save_weights_for_sampler(name=name).result()
    info = {"checkpoint_path": checkpoint.path, **info}
    write_checkpoint_info(run_dir, info, False)
    print(f"Saved sampler checkpoint: {checkpoint.path}")
    if not no_publish:
        try:
            publish(checkpoint.path)
        except Exception:
            print(f"Checkpoint exists at {checkpoint.path} and still needs publishing "
                  "(checkpoint.json says published: false).")
            raise
        write_checkpoint_info(run_dir, info, True)
    return {**info, "published": not no_publish}


def main() -> None:
    args = parse_args()

    run_dir = RUNS / args.name
    rows = load_train_rows(run_dir)
    base_lr = args.lr if args.lr is not None else hyperparam_utils.get_lr(STUDENT)
    renderer = build_renderer()

    print(f"Rendering {len(rows)} examples...")
    data = render_examples(rows, renderer)  # raises on over-long examples, before any client exists
    train_tokens = count_tokens(data)
    plan = batches(len(data), args.batch_size)
    print(f"{len(plan)} steps, {train_tokens / 1e6:.1f}M tokens, lr {base_lr:.2e}, rank {args.rank}")

    log_path = run_dir / "train_log.jsonl"
    log_path.write_text("")
    (run_dir / "checkpoint.json").unlink(missing_ok=True)

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=STUDENT, rank=args.rank).result()
    started = time.time()

    for step, idx in enumerate(plan):
        batch = [data[i] for i in idx]
        adam = types.AdamParams(learning_rate=lr_at(step, len(plan), base_lr),
                                beta1=0.9, beta2=0.95, eps=1e-8)
        fwd_future = client.forward_backward(batch, loss_fn="cross_entropy")
        opt_future = client.optim_step(adam)
        result = fwd_future.result()
        opt_future.result()

        loss = batch_loss(result.loss_fn_outputs, [d.loss_fn_inputs for d in batch])
        jsonl.append(log_path, {"step": step, "loss": loss, "lr": adam.learning_rate,
                                "elapsed_s": round(time.time() - started, 1)})
        if step % 20 == 0:
            print(f"step {step + 1}/{len(plan)} loss {loss:.4f}")
        if step > 0 and step % SAVE_EVERY == 0:
            client.save_state(name=f"{args.name}-step{step}", ttl_seconds=STATE_TTL_SECONDS).result()

    info = {"base_model": STUDENT, "rank": args.rank, "lr": base_lr,
            "batch_size": args.batch_size, "steps": len(plan), "train_tokens": train_tokens}

    def publish(path: str) -> None:
        tinker.RestClient().publish_checkpoint_from_tinker_path(path).result()

    final = save_and_publish(client, publish, run_dir, args.name, info, args.no_publish)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
