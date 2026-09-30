"""Renderiza o rosto em relevo dentro da chuva Matrix como GIF em loop.

Uso: python scripts/matrix_face.py face.npz assets/matrix-face.gif
(face.npz vem de scripts/depth_map.py)
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

src, out = sys.argv[1:3]
rng = np.random.default_rng(7)

# ---- parâmetros -----------------------------------------------------------
W, H = 840, 480              # tamanho do GIF
CW, CH = 5, 8                # tamanho da célula (px)
FRAMES, MS = 40, 80          # quadros e duração de cada um
CROP = (70, 40, 1070, 1330)  # recorte (x0, y0, x1, y1) da foto original
YAW, PITCH = 0.26, 0.06      # amplitude do movimento da cabeça (rad)
RELIEF = 16.0                # deslocamento lateral (células) pela profundidade
LIFT = 8.0                   # quanto os glifos "saltam" para cima (px) no relevo
LIGHT = np.array([-0.65, -0.45, 0.62])  # luz vindo da esquerda/cima
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

C, R = W // CW, H // CH
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# ---- glifos em 3 tamanhos (mais perto = maior) ----------------------------
chars = [chr(c) for c in range(0xFF66, 0xFF9E)] + list("0123456789:=*+-<>¦|Z")
NG = len(chars)
atlases, bolds = [], []
for size in (9, 10, 12):
    font = ImageFont.truetype(FONT, size)
    tw, th = size, size + 3
    tiles = []
    for ch in chars:
        g = Image.new("L", (tw, th))
        ImageDraw.Draw(g).text((tw / 2, th / 2), ch, font=font, fill=255, anchor="mm")
        tiles.append(np.asarray(g.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32) / 255)
    atlases.append(np.stack(tiles))
    bold = []
    for ch in chars:
        g = Image.new("L", (tw, th))
        ImageDraw.Draw(g).text((tw / 2, th / 2), ch, font=font, fill=255, anchor="mm", stroke_width=1, stroke_fill=90)
        bold.append(np.asarray(g.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32) / 255)
    bolds.append(np.stack(bold))

# ---- rosto ----------------------------------------------------------------
d = np.load(src)
x0, y0, x1, y1 = CROP
depth, luma, person = (d[k][y0:y1, x0:x1] for k in ("depth", "luma", "person"))

# vinheta elíptica: o corpo se dissolve na chuva, só a cabeça fica sólida
vy, vx = np.mgrid[y0:y1, x0:x1]
ell = ((vx - 565) / 460) ** 2 + ((vy - 660) / 620) ** 2
person = person * np.clip((1.2 - ell) / 0.4, 0, 1) ** 1.5

fh = R
fw = fh * (x1 - x0) / (y1 - y0) * (CH / CW)
sy, sx = (y1 - y0) / fh, (x1 - x0) / fw
cx0 = (C - fw) / 2
OS = 3  # superamostragem da grade do rosto


def small(a, s):
    a = gaussian_filter(a, s)
    ys = (np.arange(R * OS) + 0.5) / OS * sy
    xs = (np.arange(int(np.ceil(fw * OS))) + 0.5) / OS * sx
    return map_coordinates(a, np.meshgrid(ys, xs, indexing="ij"), order=1)


D = small(depth, 2)
P = np.clip(small(person, 2) * 1.3, 0, 1)
L = small(luma, 1.2)
# albedo normalizado dentro do rosto: olhos, bigode e cabelo ficam escuros
lo, hi = np.percentile(L[P > 0.6], [4, 97])
alb = np.clip((L - lo) / (hi - lo), 0, 1) ** 0.8
# textura (cachos, óculos, bigode) via passa-alta
tex = np.clip((L - gaussian_filter(L, 4)) * 4 + 0.5, 0, 1)
gy, gx = np.gradient(gaussian_filter(D, 1.0) * 90)
nrm = np.stack([-gx, -gy, np.ones_like(D)])
nrm /= np.linalg.norm(nrm, axis=0)


def sample(a, yy, xx):
    return map_coordinates(a, [yy * OS, xx * OS], order=1, mode="constant")


gy_, gx_ = np.mgrid[0:R, 0:C].astype(np.float32) + 0.5

# ---- chuva ----------------------------------------------------------------
TRAIL = 22
period = R + TRAIL + 6
drops = [(c, rng.uniform(0, period), rng.integers(1, 3) * period / FRAMES, rng.integers(10, TRAIL))
         for c in range(C) for _ in range(2)]
glyph0 = rng.integers(0, NG, (R, C))
flick = np.choose(rng.integers(0, 4, (R, C)), [2, 4, 5, 8])  # divisores de FRAMES => loop perfeito
phase = rng.integers(0, FRAMES, (R, C))

FACE_NORM = 0.62  # brilho que vira 100% no rosto
PAD = 16
frames = []
for f in range(FRAMES):
    t = f / FRAMES
    yaw = YAW * np.sin(2 * np.pi * t)
    pitch = PITCH * np.sin(4 * np.pi * t + 0.6)

    # warp inverso: cada célula da tela mostra o ponto do rosto deslocado pela profundidade
    fx, fy = gx_ - cx0, gy_.copy()
    for _ in range(5):
        dd = sample(D, fy, fx)
        fx = gx_ - cx0 - np.sin(yaw) * RELIEF * dd
        fy = gy_ - np.sin(pitch) * RELIEF * 0.5 * dd
    dd = np.clip(sample(D, fy, fx), 0, 1)
    pm = sample(P, fy, fx)
    n = np.stack([sample(nrm[i], fy, fx) for i in range(3)])
    nx = n[0] * np.cos(yaw) + n[2] * np.sin(yaw)
    nz = -n[0] * np.sin(yaw) + n[2] * np.cos(yaw)
    ny = n[1] * np.cos(pitch) + nz * np.sin(pitch)
    nz = -n[1] * np.sin(pitch) + nz * np.cos(pitch)
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    fill = np.clip(-nx * LIGHT[0] * 0.8 + nz * 0.6, 0, 1)
    spec = np.clip(nz, 0, 1) ** 12 * 0.35
    a = sample(alb, fy, fx)
    tx = sample(tex, fy, fx)
    shade = 0.22 + 0.95 * lam ** 1.4 + 0.3 * fill
    face = ((0.1 + 0.9 * a) * (0.5 + 1.1 * (tx - 0.5)) * shade + spec) * (0.35 + 0.75 * dd ** 0.7)
    face = np.clip(face / FACE_NORM, 0, 1) ** 0.85 * pm
    face = np.where(face < 0.07, 0, face)  # vazios: sombras viram preto

    if f == 0 and "--debug" in sys.argv:
        for nm, arr in (("face", face), ("lam", lam * pm), ("alb", a * pm), ("tex", tx * pm), ("dd", dd * pm)):
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((C * 3, R * 4), Image.NEAREST).save(out.replace(".gif", f"_{nm}.png"))
    rain = np.zeros((R, C))
    head = np.zeros((R, C))
    rows = np.arange(R)
    for c, oy, v, ln in drops:
        hy = (oy + v * f) % period - 6
        tr = hy - rows
        m = (tr >= 0) & (tr < ln)
        rain[m, c] = np.maximum(rain[m, c], (1 - tr[m] / ln) ** 1.6)
        hr = int(np.floor(hy))
        if 0 <= hr < R:
            head[hr, c] = 1

    bg = rain * 0.5 * (1 - pm)
    fg = face * (0.8 + 0.35 * rain)
    inten = np.maximum(bg, fg) + head * (1 - pm) * 0.8
    white = np.clip(head * (1 - pm) + np.clip(face - 0.85, 0, 1) * 2 + face * rain * head * 2, 0, 1)
    scale = np.where(pm > 0.3, np.digitize(dd, [0.5, 0.8]), 0)
    lift = np.where(pm > 0.3, dd * LIFT, 0)
    gi = (glyph0 + (f + phase) // flick) % NG

    gcan = np.zeros((H + 2 * PAD, W + 2 * PAD))
    wcan = np.zeros_like(gcan)
    order = np.argsort(dd.ravel())  # desenha do fundo para a frente
    for idx in order:
        r, c = divmod(idx, C)
        v = inten[r, c]
        if v < 0.03:
            continue
        tile = (bolds if pm[r, c] > 0.3 else atlases)[scale[r, c]][gi[r, c]]
        th, tw = tile.shape
        py = int(PAD + r * CH + CH / 2 - lift[r, c] - th / 2)
        px = int(PAD + c * CW + CW / 2 - tw / 2)
        sl = (slice(py, py + th), slice(px, px + tw))
        gcan[sl] = np.maximum(gcan[sl], tile * v)
        wcan[sl] = np.maximum(wcan[sl], tile * v * white[r, c])
    g = gcan[PAD:-PAD, PAD:-PAD]
    w = wcan[PAD:-PAD, PAD:-PAD]
    g = np.clip(g * 1.2 + gaussian_filter(g, 1.6) * 0.45, 0, 1)
    rgb = np.clip(np.stack([g * 0.12 + w * 0.75, g, g * 0.3 + w * 0.6], -1), 0, 1)
    frames.append(Image.fromarray((rgb * 255).astype(np.uint8)))

pal = frames[FRAMES // 4].quantize(colors=48, method=Image.Quantize.MEDIANCUT)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
q[0].save(out, save_all=True, append_images=q[1:], duration=MS, loop=0, optimize=True, disposal=1)

# folha de contato para conferência
sheet = Image.new("RGB", (W * 2, H * 2))
for i, k in enumerate([0, FRAMES // 4, FRAMES // 2, 3 * FRAMES // 4]):
    sheet.paste(frames[k], ((i % 2) * W, (i // 2) * H))
sheet.save(out.replace(".gif", "_sheet.png"))
print("ok")
