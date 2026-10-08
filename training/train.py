"""Train a (small) VGGT on the procedural multi-view data.

Example:
    python -m training.train --out runs/aa --aa-order frame global --steps 6000
"""

import argparse
import json
import math
import os
import time

import torch

from training.losses import vggt_loss
from training.synthetic import MultiViewDataset, generate_dataset
from vggt import VGGT


def get_data(path, num_scenes, img_size, seed):
    if os.path.exists(path):
        return torch.load(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = time.time()
    data = generate_dataset(path, num_scenes, img_size=img_size, seed=seed)
    print(f"rendered {num_scenes} scenes to {path} in {time.time() - t:.0f}s", flush=True)
    return data


def build_model(cfg):
    return VGGT.small(img_size=cfg["img_size"], embed_dim=cfg["embed_dim"], depth=cfg["depth"],
                      num_heads=cfg["num_heads"], aa_order=tuple(cfg["aa_order"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/train.pt")
    ap.add_argument("--num-scenes", type=int, default=3000)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--aa-order", nargs="+", default=["frame", "global"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default=None, help="checkpoint to initialise the weights from (fine-tune / resume)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(img_size=args.img_size, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
               aa_order=args.aa_order)
    json.dump(dict(cfg, **vars(args)), open(os.path.join(args.out, "config.json"), "w"), indent=1)

    ds = MultiViewDataset(get_data(args.data, args.num_scenes, args.img_size, seed=1234), seed=args.seed)
    model = build_model(cfg)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  aa_order={args.aa_order}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "token" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):  # linear warm-up + cosine decay (paper Sec. 3.5)
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        S = int(torch.randint(args.min_frames, args.max_frames + 1, (1,), generator=ds.gen))
        batch = {k: v.to(args.device) for k, v in ds.batch(args.batch, S).items()}
        preds = model(batch["images"])
        loss, logs = vggt_loss(preds, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            rec = {k: round(float(v), 4) for k, v in logs.items()}
            rec.update(step=step, S=S, lr=lr_at(step), gnorm=round(float(gnorm), 3), sec=round(time.time() - t0))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if (step + 1) % 1000 == 0 or step == args.steps - 1:
            torch.save({"cfg": cfg, "model": model.state_dict(), "step": step + 1},
                       os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
