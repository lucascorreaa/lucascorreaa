"""Chuva Matrix que revela o rosto: onde a gota passa, o retrato aparece no rastro
e se apaga devagar atrás dela.

Uso: python scripts/matrix_face.py face.npz assets/matrix-face.gif
(face.npz vem de scripts/depth_map.py)
Depois: gifsicle -O3 --lossy=60 assets/matrix-face.gif -o assets/matrix-face.gif
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

src, out = sys.argv[1:3]
rng = np.random.default_rng(3)

# ---- parâmetros -----------------------------------------------------------
W, H = 1260, 720              # tamanho do GIF (exibido reduzido no README)
CW, CH = 5, 8                 # tamanho da célula (px)
FRAMES, MS = 90, 70           # quadros e duração de cada um (loop de 6,3 s)
CROP = (70, 40, 1070, 1330)   # recorte (x0, y0, x1, y1) da foto original
DROPS = 2                     # gotas por coluna
TRAIL = (8, 22)               # comprimento do rastro no fundo (células)
FACE_TRAIL = 3.6              # no rosto o rastro dura mais (multiplicador)
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

C, R = W // CW, H // CH

# ---- glifos ---------------------------------------------------------------
chars = [chr(c) for c in range(0xFF66, 0xFF9E)] + list("0123456789Z:=*+-<>|")
font = ImageFont.truetype(FONT, 9)
tiles = []
for ch in chars:
    im = Image.new("L", (CW, CH))
    ImageDraw.Draw(im).text((CW / 2, CH / 2), ch, font=font, fill=255, anchor="mm")
    tiles.append(np.asarray(im.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32) / 255)
tiles = np.stack(tiles)
NG = len(chars)
cov = tiles.reshape(NG, -1).mean(1)
by_cov = np.argsort(cov)

# ---- retrato na grade -----------------------------------------------------
d = np.load(src)
x0, y0, x1, y1 = CROP
luma, person = d["luma"][y0:y1, x0:x1], d["person"][y0:y1, x0:x1]
vy, vx = np.mgrid[y0:y1, x0:x1]
ell = ((vx - 565) / 470) ** 2 + ((vy - 650) / 640) ** 2
person = person * np.clip((1.15 - ell) / 0.35, 0, 1)  # corpo some, fica a cabeça

fw = R * (x1 - x0) / (y1 - y0) * (CH / CW)
sy, sx = (y1 - y0) / R, (x1 - x0) / fw
cx0 = (C - fw) / 2
rr, cc = np.mgrid[0:R, 0:C].astype(np.float32)
coords = [(rr + 0.5) * sy, (cc + 0.5 - cx0) * sx]


def grid(a, s):
    return map_coordinates(gaussian_filter(a, s), coords, order=1, mode="constant")


P = np.clip(grid(person, 3) * 1.2, 0, 1)
L = grid(luma, 2.5)
lo, hi = np.percentile(L[P > 0.6], [3, 98])
detail = np.clip((grid(luma, 1.0) - grid(luma, 8)) / (hi - lo) * 1.8, -0.5, 0.5)
tone = np.clip(np.clip((L - lo) / (hi - lo), 0, 1) ** 0.85 + detail, 0, 1) * P

# no rosto, o glifo é escolhido pela densidade de tinta que o tom pede
pos = np.searchsorted(cov[by_cov], tone / tone.max() * cov.max())
face_glyph = by_cov[np.clip(pos + rng.integers(-3, 4, pos.shape), 0, NG - 1)]
on_face = (P > 0.05) & (tone > 0.04)

# ---- chuva (loop perfeito: cada gota dá voltas inteiras no período) --------
period = int(R * 1.5)
drops = [(c, rng.uniform(0, period), 1 + (rng.random() < 0.2), rng.integers(*TRAIL))
         for c in range(C) for _ in range(DROPS)]
rain_glyph = rng.integers(0, NG, (R, C))
flick = np.choose(rng.integers(0, 3, (R, C)), [5, 9, 15])  # divisores de FRAMES
phase = rng.integers(0, FRAMES, (R, C))
rows = np.arange(R)

frames = []
for f in range(FRAMES):
    rain = np.zeros((R, C))
    reveal = np.zeros((R, C))
    head = np.zeros((R, C))
    for c, oy, laps, ln in drops:
        hy = (oy + laps * period * f / FRAMES) % period
        tr = (hy - rows) % period  # rastro contínuo através da volta
        m = (tr >= 0) & (tr < ln)
        rain[m, c] = np.maximum(rain[m, c], (1 - tr[m] / ln) ** 1.5)
        lf = min(ln * FACE_TRAIL, period - 12)
        m = tr < lf
        reveal[m, c] = np.maximum(reveal[m, c], np.clip((1 - tr[m] / lf) / 0.4, 0, 1))
        hr = int(np.floor(hy))
        if 0 <= hr < R:
            head[hr, c] = 1

    face = tone * reveal
    gi = np.where(on_face & (reveal > 0), face_glyph,
                  (rain_glyph + (f + phase) // flick) % NG)
    inten = np.where(on_face, (0.45 + 0.75 * tone) * reveal,
                     rain * 0.5)
    inten = np.clip(inten + head * 0.85, 0, 1)
    white = np.clip(head + np.clip(face - 0.8, 0, 1) * 2, 0, 1)

    t = tiles[gi] * inten[:, :, None, None]
    t = t.transpose(0, 2, 1, 3).reshape(R * CH, C * CW)
    t = np.maximum(t, np.repeat(np.repeat(face * 0.16, CH, 0), CW, 1))
    w = tiles[gi] * (inten * white)[:, :, None, None]
    w = w.transpose(0, 2, 1, 3).reshape(R * CH, C * CW)
    g = np.clip(t * 1.15 + gaussian_filter(t, 1.2) * 0.25, 0, 1)
    rgb = np.clip(np.stack([g * 0.1 + w * 0.8, g, g * 0.28 + w * 0.65], -1), 0, 1)
    img = np.zeros((H, W, 3))
    img[: R * CH, : C * CW] = rgb
    frames.append(Image.fromarray((img * 255).astype(np.uint8)))

pal = frames[0].quantize(colors=20, method=Image.Quantize.MEDIANCUT)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
q[0].save(out, save_all=True, append_images=q[1:], duration=MS, loop=0, optimize=True, disposal=1)

# folha de contato para conferência
sheet = Image.new("RGB", (W, H * 2))
for i, fr in enumerate([0, FRAMES // 2]):
    sheet.paste(frames[fr], (0, i * H))
sheet.resize((W * 2 // 3, H * 4 // 3), Image.LANCZOS).save(out.replace(".gif", "_sheet.png"))
print("ok")
