#!/usr/bin/env python3
"""Realtime web UI for the Pacman latent-diffusion world model.

Loads the epoch11 PacmanDiffusionModel + fine-tuned TAESD decoder once, then
serves a single-page canvas app over FastAPI + a WebSocket game loop.

Protocol (single WS at /ws):
  client → server : text message  — "reset", "reset:42" (seed), or an action
                    name (LEFT, RIGHT, UP, DOWN, NO_ACTION)
  server → client : text message  — JSON metadata {frame, action, latency_ms,
                    episode, fps}
                  : binary frame — JPEG bytes of the generated/seed frame

Run:
    python app/webui_pacman.py \
        --config configs/sana_config/512ms/Sana_pacman.yaml \
        --ckpt output/pacman_latent_v2/checkpoints/epoch_11_step_282445.pth \
        --ft_decoder output/vae_decoder_ft/vae_decoder_ft_best.pth
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import os.path as osp
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

# ── repo imports (must come after sys.path setup) ──────────────────────────
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "scripts"))

from dotenv import load_dotenv

load_dotenv(Path(REPO_ROOT) / ".env")

from diffusion import FlowEuler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusion.model.builder import build_model, get_vae
from diffusion.utils.config import SanaConfig

import pyrallis

# ── constants ──────────────────────────────────────────────────────────────
ACTION_NAMES = ["LEFT", "RIGHT", "UP", "DOWN", "NO_ACTION"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


# ── helpers (mirrors scripts/inference_pacman.py) ──────────────────────────
def load_config(config_path: str) -> SanaConfig:
    return pyrallis.load(SanaConfig, open(config_path))


def build_inference_model(config: SanaConfig, device: str):
    image_size = config.model.image_size
    latent_size = image_size // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma

    vae = get_vae(
        config.vae.vae_type,
        config.vae.vae_pretrained,
        device,
        finetuned_decoder=getattr(config.vae, "finetuned_decoder", None),
    ).to(torch.float32)

    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "y_norm": True,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.model.in_channels,
        "y_norm_scale_factor": 1.0,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "caption_channels": config.model.num_classes,
        "model_max_length": config.data.sequence_length - 1,
        "seq_length": config.data.sequence_length,
        "vae": vae,
        "accelerator": None,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).eval().to(device)
    return model, vae


def load_checkpoint_into_model(model, checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    for key in ["pos_embed", "base_model.pos_embed", "model.pos_embed", "sana.pos_embed"]:
        if key in state_dict:
            del state_dict[key]

    # Drop VAE weights baked into the checkpoint: the VAE (incl. the fine-tuned
    # decoder) is loaded separately via get_vae. Otherwise these STOCK weights
    # overwrite the fine-tuned decoder on load -> blobby output.
    for k in [k for k in state_dict if k.startswith("vae.")]:
        del state_dict[k]

    null_embed_root = config.train.null_embed_root
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    null_embed_path = osp.join(
        null_embed_root,
        f"null_embed_diffusers_{config.vae.vae_type}_{latent_size}.pth",
    )
    if osp.exists(null_embed_path):
        null_embed = torch.load(null_embed_path, map_location="cpu")
        if null_embed is not None and "y_embedder.y_embedding" not in state_dict:
            state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
            print(f"[Checkpoint] Loaded null_embed from {null_embed_path}")
    else:
        try:
            import huggingface_hub

            token = os.environ.get("HF_TOKEN")
            if token:
                huggingface_hub.login(token=token)
                null_embed_filename = osp.basename(null_embed_path)
                null_embed_file = huggingface_hub.hf_hub_download(
                    repo_id="Tahahah/pacman-sana-3.2m-taesd-v2",
                    filename=f"pretrained_models/{null_embed_filename}",
                    repo_type="model",
                    token=token,
                )
                null_embed = torch.load(null_embed_file, map_location="cpu")
                state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
                print("[Checkpoint] Loaded null_embed from HF")
        except Exception as e:
            print(f"[Checkpoint] Could not load null_embed: {e}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected}")
    return model


# ── world-model session ───────────────────────────────────────────────────
class WorldSession:
    """Holds the autoregressive state for one user session."""

    def __init__(self, config, model, vae, latents, episodes, dones, device, cfg_scale=4.5, steps=2):
        self.config = config
        self.model = model
        self.vae = vae
        self.latents = latents
        self.episodes = episodes
        self.dones = dones
        self.device = device
        self.cfg_scale = cfg_scale
        self.steps = steps
        self.seq_len = config.data.sequence_length
        self.S = self.seq_len - 1  # number of obs frames
        self.Lc = config.vae.vae_latent_dim
        self.Ls = config.model.image_size // config.vae.vae_downsample_rate
        self.hw = torch.tensor(
            [[config.model.image_size, config.model.image_size]], dtype=torch.float, device=device
        )
        self.ar = torch.tensor([[1.0]], device=device)
        self.null_y = torch.zeros(1, 1, self.S, 5, device=device)
        self.mdtype = next(model.parameters()).dtype
        self._one_hot = {
            i: torch.zeros(5, dtype=torch.float16).scatter_(0, torch.tensor(i), 1)
            for i in range(5)
        }
        self.episode_id = -1
        self.gen_count = 0
        self.buf: list[torch.Tensor] = []

    def reset(self, episode_idx: int | None = None, seed: int | None = None):
        """Seed the obs buffer from a random (or chosen) episode's first S frames."""
        N = len(self.latents)
        episodes = self.episodes
        dones = self.dones

        if episode_idx is None:
            rng = np.random.default_rng(seed)
            for _ in range(200):
                idx = int(rng.integers(self.S, N - 1))
                e = episodes[idx]
                start = idx
                while start > 0 and episodes[start - 1] == e and not bool(dones[start - 1]):
                    start -= 1
                if idx - start >= self.S:
                    episode_idx = start
                    break
            if episode_idx is None:
                episode_idx = 0

        self.episode_id = int(episodes[episode_idx])
        self.buf = [
            self.latents[episode_idx + k].unsqueeze(0).to(self.device).float()
            for k in range(self.S)
        ]
        self.gen_count = 0
        # Return the last seed frame as JPEG so the UI has something to show
        with torch.no_grad():
            px = self.vae.decoder(self.buf[-1].float()).clamp(0, 1)
        return self._tensor_to_jpeg(px[0])

    def step(self, action_name: str, seed: int | None = None):
        """Generate one next frame given the user's action. Returns (jpeg_bytes, meta)."""
        action_idx = ACTION_TO_IDX.get(action_name, 4)
        one_hot = self._one_hot[action_idx].to(self.device).float()

        # Broadcast the user's single action across all S obs slots
        y = one_hot.view(1, 1, 1, 5).expand(1, 1, self.S, 5).contiguous()
        obs_cat = torch.cat(self.buf, dim=1)  # [1, S*4, 32, 32]

        t0 = time.time()
        with torch.inference_mode():
            obs_latent = self.model.encode_obs(obs_cat.to(self.mdtype)).float()
            mk = dict(
                data_info={"img_hw": self.hw, "aspect_ratio": self.ar},
                mask=None,
                obs_latent=obs_latent,
            )
            g = torch.Generator(device=self.device)
            g.manual_seed(seed if seed is not None else (self.gen_count + 1))
            z = torch.randn(1, self.Lc, self.Ls, self.Ls, device=self.device, generator=g)
            gl = FlowEuler(
                self.model,
                condition=y,
                uncondition=self.null_y,
                cfg_scale=self.cfg_scale,
                model_kwargs=mk,
            ).sample(z, steps=self.steps).float()
            px = self.vae.decoder(gl).clamp(0, 1)
        dt = time.time() - t0

        self.buf = self.buf[1:] + [gl]
        self.gen_count += 1
        meta = {
            "frame": self.gen_count,
            "action": action_name,
            "latency_ms": round(dt * 1000, 1),
            "episode": self.episode_id,
        }
        return self._tensor_to_jpeg(px[0]), meta

    @staticmethod
    def _tensor_to_jpeg(t: torch.Tensor, quality: int = 90) -> bytes:
        arr = (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


class TRTWorldSession(WorldSession):
    """WorldSession backed by ONNX Runtime + TensorRT EP for the diffusion forward
    and VAE decode.  encode_obs still runs in PyTorch (tiny conv3d, not the bottleneck).

    The ONNX engines are static-shape (batch=2 for CFG, batch=1 for VAE decode),
    built and cached on first inference.  fp32 precision — fp16 was verified to
    diverge >1e-2 per step; fp32 stays within 3.6e-3 and is visually equivalent.
    """

    def __init__(self, *args, trt_diff_sess, trt_vae_sess, **kwargs):
        super().__init__(*args, **kwargs)
        self.trt_diff = trt_diff_sess
        self.trt_vae = trt_vae_sess
        self._hw_np = np.array(
            [[self.config.model.image_size, self.config.model.image_size]] * 2,
            dtype=np.float32,
        )
        self._ar_np = np.array([[1.0]] * 2, dtype=np.float32)
        self._scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    def step(self, action_name: str, seed: int | None = None):
        """Generate one next frame via ONNX/TRT. Returns (jpeg_bytes, meta)."""

        action_idx = ACTION_TO_IDX.get(action_name, 4)
        one_hot = self._one_hot[action_idx].to(self.device).float()
        y = one_hot.view(1, 1, 1, 5).expand(1, 1, self.S, 5).contiguous()
        obs_cat = torch.cat(self.buf, dim=1)

        t0 = time.time()
        with torch.inference_mode():
            obs_latent = self.model.encode_obs(obs_cat.to(self.mdtype)).float()

            g = torch.Generator(device=self.device)
            g.manual_seed(seed if seed is not None else (self.gen_count + 1))
            z = torch.randn(1, self.Lc, self.Ls, self.Ls, device=self.device, generator=g)

            self._scheduler.set_timesteps(self.steps, device=self.device)
            timesteps = self._scheduler.timesteps
            latents = z

            for t in timesteps:
                x_np = torch.cat([latents] * 2).cpu().numpy().astype(np.float32)
                t_np = np.array([t.item()] * 2, dtype=np.float32)
                y_np = torch.cat([self.null_y, y], dim=0).cpu().numpy().astype(np.float32)
                obs_np = torch.cat([obs_latent] * 2).cpu().numpy().astype(np.float32)

                inputs = {
                    "x": x_np, "timestep": t_np, "y": y_np,
                    "img_hw": self._hw_np, "aspect_ratio": self._ar_np,
                    "obs_latent": obs_np,
                }
                noise_pred = self.trt_diff.run(None, inputs)[0]

                v = noise_pred[0:1] + self.cfg_scale * (noise_pred[1:2] - noise_pred[0:1])
                v_tensor = torch.from_numpy(v).to(self.device)
                latents = self._scheduler.step(v_tensor, t, latents, return_dict=False)[0]

            gl = latents.float()
            gl_np = gl.cpu().numpy().astype(np.float32)
            px_np = self.trt_vae.run(None, {"latent": gl_np})[0]
            px = torch.from_numpy(px_np).clamp(0, 1)
        dt = time.time() - t0

        self.buf = self.buf[1:] + [gl]
        self.gen_count += 1
        meta = {
            "frame": self.gen_count,
            "action": action_name,
            "latency_ms": round(dt * 1000, 1),
            "episode": self.episode_id,
        }
        return self._tensor_to_jpeg(px[0]), meta


# ── global state (loaded once at startup) ──────────────────────────────────
app = FastAPI(title="Pacman World Model")
STATE: dict = {}


def init_state(config: SanaConfig, ckpt_path: str, ckpt_name: str, cfg_scale: float, steps: int, seed_data_path: str, use_trt: bool = False, onnx_dir: str = "output/onnx"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.train.null_embed_root, exist_ok=True)

    print("[Init] Building model...")
    model, vae = build_inference_model(config, device)
    print(f"[Init] Loading checkpoint from {ckpt_path}")
    model = load_checkpoint_into_model(model, ckpt_path, config, device)
    model = model.eval()
    vae = vae.to(device).to(torch.float32).eval()

    print(f"[Init] Loading seed latents from {seed_data_path}...")
    latent_data = torch.load(seed_data_path, map_location="cpu", weights_only=False)
    latents = latent_data["latents"]
    episodes = latent_data["episodes"]
    dones = latent_data["dones"]
    print(f"[Init] {len(latents)} seed latents loaded")

    trt_diff_sess = None
    trt_vae_sess = None
    if use_trt:
        import onnxruntime as ort
        diff_onnx = osp.join(onnx_dir, "diffusion_b2.onnx")
        vae_onnx = osp.join(onnx_dir, "vae_decoder_b1.onnx")
        if not osp.isfile(diff_onnx) or not osp.isfile(vae_onnx):
            raise FileNotFoundError(
                f"ONNX models not found in {onnx_dir}. Run export_trt.py first."
            )
        trt_cache = "/tmp/trt_cache_webui"
        os.makedirs(trt_cache, exist_ok=True)
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                "trt_fp16_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
            }),
            "CUDAExecutionProvider",
        ]
        print(f"[Init] Loading ONNX/TRT sessions from {onnx_dir}...")
        trt_diff_sess = ort.InferenceSession(diff_onnx, providers=providers)
        trt_vae_sess = ort.InferenceSession(vae_onnx, providers=providers)
        print(f"[Init] TRT providers: {trt_diff_sess.get_providers()}")

    STATE["config"] = config
    STATE["model"] = model
    STATE["vae"] = vae
    STATE["device"] = device
    if use_trt:
        STATE["session"] = TRTWorldSession(
            config, model, vae, latents, episodes, dones, device, cfg_scale, steps,
            trt_diff_sess=trt_diff_sess, trt_vae_sess=trt_vae_sess,
        )
        print("[Init] Warming up TRT engines (first inference builds the engine)...")
        STATE["session"].reset()
        # Warmup with a dummy step
        import numpy as np
        _ = STATE["session"].step("NO_ACTION", seed=0)
        print("[Init] TRT warmup done")
    else:
        STATE["session"] = WorldSession(
            config, model, vae, latents, episodes, dones, device, cfg_scale, steps
        )
    STATE["ckpt_name"] = ckpt_name
    print(f"[Init] Ready. {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}")


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = osp.join(SCRIPT_DIR, "static", "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/api/info")
async def info():
    cfg = STATE["config"]
    return JSONResponse(
        {
            "model": cfg.model.model,
            "image_size": cfg.model.image_size,
            "checkpoint": STATE["ckpt_name"],
            "actions": ACTION_NAMES,
        }
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Server-side streaming loop.

    The server runs the autoregressive loop continuously, pushing JPEG frames
    as fast as it generates them (~25 FPS).  The client just renders incoming
    frames and sends actions fire-and-forget.  Actions are queued server-side
    and consumed on the next frame — RTT becomes a fixed input-lag, not a
    per-frame tax.
    """
    await ws.accept()
    session: WorldSession = STATE["session"]
    action_queue: asyncio.Queue = asyncio.Queue()
    streaming = True
    fps_cap = 0  # 0 = uncapped (native speed)

    await ws.send_text(json.dumps({"type": "ready", "actions": ACTION_NAMES}))

    async def receive_actions():
        """Drain incoming WS messages into the action queue (fire-and-forget)."""
        nonlocal streaming, fps_cap
        try:
            while streaming:
                msg = (await ws.receive_text()).strip()
                if msg.startswith("reset"):
                    parts = msg.split(":")
                    seed = int(parts[1]) if len(parts) > 1 else None
                    await action_queue.put(("reset", seed))
                elif msg.startswith("fps:"):
                    try:
                        fps_cap = max(0, int(msg.split(":")[1]))
                    except (ValueError, IndexError):
                        pass
                elif msg in ACTION_TO_IDX:
                    await action_queue.put(("action", msg))
        except WebSocketDisconnect:
            streaming = False
        except Exception:
            streaming = False

    async def stream_frames():
        """Continuously generate and push frames, rate-capped to fps_cap.

        fps_cap=0 means uncapped (native speed).  Otherwise the loop sleeps
        to fill the frame interval (1000/fps_cap ms), keeping the GPU idle
        between frames instead of generating frames that would be dropped.
        """
        nonlocal streaming
        loop = asyncio.get_event_loop()
        # Initial reset
        await action_queue.put(("reset", None))
        try:
            while streaming:
                # Drain the action queue: last action wins, reset takes priority
                pending_reset = False
                reset_seed = None
                pending_action = None
                while not action_queue.empty():
                    kind, val = action_queue.get_nowait()
                    if kind == "reset":
                        pending_reset = True
                        reset_seed = val
                    else:
                        pending_action = val

                t0 = time.time()
                if pending_reset:
                    jpeg = await loop.run_in_executor(
                        None, lambda: session.reset(seed=reset_seed)
                    )
                    meta = json.dumps({
                        "type": "reset",
                        "episode": session.episode_id,
                        "frame": 0,
                        "actions": ACTION_NAMES,
                    })
                    await ws.send_text(meta)
                    await ws.send_bytes(jpeg)
                    pending_action = None
                else:
                    # Generate one frame with the queued action (or NO_ACTION)
                    action = pending_action or "NO_ACTION"
                    jpeg, step_meta = await loop.run_in_executor(
                        None, lambda: session.step(action)
                    )
                    step_meta["type"] = "frame"
                    await ws.send_text(json.dumps(step_meta))
                    await ws.send_bytes(jpeg)

                # Rate-cap: sleep to fill the frame interval
                if fps_cap > 0:
                    elapsed = time.time() - t0
                    frame_interval = 1.0 / fps_cap
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)


        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS stream] error: {e}")
        finally:
            streaming = False

    # Run receiver and streamer concurrently
    await asyncio.gather(receive_actions(), stream_frames())


# ── entrypoint ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Pacman world-model realtime web UI")
    p.add_argument(
        "--config",
        default="configs/sana_config/512ms/Sana_pacman.yaml",
    )
    p.add_argument(
        "--ckpt",
        default="output/pacman_latent_v2/checkpoints/epoch_11_step_282445.pth",
        help="Checkpoint path (use epoch11 explicitly — latest.pth is a 0-byte symlink)",
    )
    p.add_argument(
        "--ft_decoder",
        default="output/vae_decoder_ft/vae_decoder_ft_best.pth",
        help="Decoder-only finetuned TAESD (encoder frozen = stock, matches diffusion model's latent space)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--steps", type=int, default=2, help="FlowEuler sampling steps")
    p.add_argument(
        "--seed_data",
        default="datasets/pacman_seed.pt",
        help="Path to seed latents file (small subset of precomputed latents for episode seeding)",
    )
    p.add_argument(
        "--use_trt",
        action="store_true",
        help="Use ONNX Runtime + TensorRT EP for diffusion forward and VAE decode (requires pre-exported ONNX models)",
    )
    p.add_argument(
        "--onnx_dir",
        default="output/onnx",
        help="Directory containing diffusion_b2.onnx and vae_decoder_b1.onnx",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.ft_decoder and osp.isfile(args.ft_decoder):
        config.vae.finetuned_decoder = args.ft_decoder
    init_state(config, args.ckpt, osp.basename(args.ckpt), args.cfg_scale, args.steps, args.seed_data, use_trt=args.use_trt, onnx_dir=args.onnx_dir)
    import uvicorn

    print(f"\n Pacman World Model UI: http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)
