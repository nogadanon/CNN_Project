"""FastAPI backend: furniture classification (CNN) + similar products (CLIP).

Run:  run_server.bat   (or, from backend\\: python -m uvicorn main:app --port 8000)
"""
import io
import os
import re
import sqlite3
import threading
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")  # TensorFlow is not installed; Keras 3 runs the model on torch

import keras
import numpy as np
import open_clip
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent.parent / "furniture_cnn_model_3.keras"
DATASET = Path(r"C:\Users\ליאור\Downloads\Furniture DB\training\CNN_furniture_dataset")
FURNITURE_DB = DATASET.parent.parent  # ...\Furniture DB (furniture.db, images\<store>\<id>.jpg)
# Cache lives outside the workspace so Live Server doesn't reload the page when it's written
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "furniture_web"
INDEX_FILE = CACHE_DIR / "clip_index.npz"

SPLITS = ("train", "test")  # another_images is not part of the catalogue
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-16", "laion2b_s34b_b88k"
TOP_K = 12
MAX_UPLOAD = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}  # the model handles WEBP poorly

# Same order as image_dataset_from_directory (sorted folder names) - verified on the test split
CLASSES = sorted(p.name for p in (DATASET / "train").iterdir() if p.is_dir())

app = FastAPI(title="Furniture classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cnn = keras.saving.load_model(MODEL_PATH)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
clip_model.eval()


def read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "התמונה גדולה מדי (מקסימום 10MB)")
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "הקובץ אינו תמונה תקינה")
    if img.format not in ALLOWED_FORMATS:  # checked by content, not by file extension
        raise HTTPException(415, "ניתן להעלות רק קבצי JPG או PNG")
    return ImageOps.exif_transpose(img).convert("RGB")


@torch.no_grad()
def clip_embed(images: list[Image.Image]) -> np.ndarray:
    x = torch.stack([clip_preprocess(im) for im in images])
    f = clip_model.encode_image(x)
    return torch.nn.functional.normalize(f, dim=-1).numpy().astype("float32")


# ---------- product info (furniture.db) ----------

def product_id(stem: str) -> str:
    """'dream-123_2' (gallery photo) -> 'dream-123'."""
    return re.sub(r"_\d+$", "", stem)


def load_products() -> dict:
    db = FURNITURE_DB / "furniture.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    rows = con.execute('SELECT "מזהה","שם מוצר","חברה","מחיר החל מ","מחיר עד","קישור למוצר","סגנון עיצוב" FROM products')
    info = {r[0]: dict(name=r[1], company=r[2], price_from=r[3], price_to=r[4], url=r[5], style=r[6]) for r in rows}
    con.close()
    return info


PRODUCTS = load_products()


def main_photo(pid: str) -> Path | None:
    """Display rule: one photo per product - the main photo images\\<store>\\<id>.jpg."""
    store, _, site_id = pid.partition("-")
    if site_id:
        p = FURNITURE_DB / "images" / store / f"{site_id}.jpg"
        if p.exists():
            return p
    return None


# ---------- similarity index ----------

class Index:
    def __init__(self):
        self.ready = False
        self.done = 0
        self.total = 0
        self.error = None
        self.paths: list[Path] = []
        self.labels = np.array([])
        self.pids = np.array([])
        self.vecs = np.zeros((0, 512), "float32")

    def build(self):
        try:
            files = sorted(
                (p, c) for s in SPLITS for c in CLASSES if (DATASET / s / c).is_dir()
                for p in (DATASET / s / c).iterdir() if p.suffix.lower() in IMG_EXT
            )
            paths = [p for p, _ in files]
            mtimes = np.array([p.stat().st_mtime for p in paths])
            self.total = len(paths)
            cached = {}
            if INDEX_FILE.exists():  # re-embed only new/changed images
                z = np.load(INDEX_FILE, allow_pickle=False)
                cached = {(k, m): v for k, m, v in zip(z["keys"], z["mtimes"], z["vecs"])}
            vecs = np.zeros((len(paths), 512), "float32")
            missing = []
            for i, p in enumerate(paths):
                v = cached.get((str(p), mtimes[i]))
                if v is None:
                    missing.append(i)
                else:
                    vecs[i] = v
            self.done = self.total - len(missing)
            for b in range(0, len(missing), 32):
                ids = missing[b:b + 32]
                vecs[ids] = clip_embed([Image.open(paths[i]).convert("RGB") for i in ids])
                self.done += len(ids)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(INDEX_FILE, keys=np.array([str(p) for p in paths]), mtimes=mtimes, vecs=vecs)

            self.paths = paths
            self.labels = np.array([c for _, c in files])
            self.pids = np.array([product_id(p.stem) for p in paths])
            self.vecs = vecs
            self.done = self.total
            self.ready = True
        except Exception as e:  # surfaced through /api/status
            self.error = repr(e)

    def search(self, query: np.ndarray, label: str, k: int):
        mask = self.labels == label
        if not mask.any():
            return []
        idx = np.flatnonzero(mask)
        sims = self.vecs[idx] @ query
        best: dict[str, tuple[float, int]] = {}  # dedupe gallery photos: best score per product
        for i, s in zip(idx, sims):
            pid = self.pids[i]
            if pid not in best or s > best[pid][0]:
                best[pid] = (float(s), int(i))
        top = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        return [dict(id=pid, score=s, image=f"/api/image/{i}", **(PRODUCTS.get(pid) or {})) for pid, (s, i) in top]

    def photo(self, i: int) -> Path:
        return main_photo(self.pids[i]) or self.paths[i]


index = Index()
threading.Thread(target=index.build, daemon=True).start()


# ---------- API ----------

@app.get("/api/classes")
def classes():
    return CLASSES


@app.get("/api/status")
def status():
    return dict(ready=index.ready, done=index.done, total=index.total, error=index.error)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    img = read_image(await file.read()).resize((224, 224))  # rescaling is inside the model
    x = np.asarray(img, dtype="float32")[None]
    probs = np.asarray(cnn.predict(x, verbose=0))[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda t: -t[1])
    return dict(predicted=ranked[0][0], probabilities=[dict(name=c, prob=p) for c, p in ranked])


@app.post("/api/similar")
async def similar(file: UploadFile = File(...), label: str = Form(...), k: int = Form(TOP_K)):
    if label not in CLASSES:
        raise HTTPException(400, "מחלקה לא מוכרת")
    if not index.ready:
        raise HTTPException(503, "מאגר הדמיון עדיין נבנה, נסו שוב בעוד רגע")
    q = clip_embed([read_image(await file.read())])[0]
    return index.search(q, label, max(1, min(k, 48)))


@app.get("/api/image/{i}")
def image(i: int):
    if not index.ready or not 0 <= i < len(index.paths):
        raise HTTPException(404)
    return FileResponse(index.photo(i))


# The site itself (http://127.0.0.1:8000/) - lets one port, e.g. an ngrok tunnel, serve everything
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="site")
