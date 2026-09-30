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
W, H = 840, 480          # tamanho do GIF
CW, CH = 6, 9            # tamanho da célula (px)
FRAMES, MS = 40, 80      # quadros e duração de cada um
CROP = (90, 110, 1050, 1330)  # recorte (x0, y0, x1, y1) da foto original
YAW, PITCH = 0.22, 0.07  # amplitude do movimento da cabeça (rad)
RELIEF = 26.0            # deslocamento máximo em células pelo relevo
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

C, R = W // CW, H // CH

# ---- glifos ---------------------------------------------------------------
chars = [chr(c) for c in range(0xFF66, 0xFF9E)] + list("0123456789:=*+-<>¦|Z")
font = ImageFont.truetype(FONT, 11)
atlas = []
for ch in chars:
    g = Image.new("L", (CW, CH))
    ImageDraw.Draw(g).text((CW / 2, CH / 2), ch, font=font, fill=255, anchor="mm")
    atlas.append(np.array(g.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32) / 255)
atlas = np.stack(atlas)
NG = len(chars)

# ---- rosto ----------------------------------------------------------------
d = np.load(src)
x0, y0, x1, y1 = CROP
depth, luma, person = (d[k][y0:y1, x0:x1] for k in ("depth", "luma", "person"))
# vinheta elíptica: o corpo se dissolve na chuva, só a cabeça fica sólida
vy, vx = np.mgrid[y0:y1, x0:x1]
ell = ((vx - 565) / 470) ** 2 + ((vy - 700) / 600) ** 2
person = person * np.clip((1.25 - ell) / 0.45, 0, 1) ** 1.5

# grade de amostragem: rosto ocupa a altura toda, centralizado
fh = R
fw = fh * (x1 - x0) / (y1 - y0) * (CH / CW)
sy = (y1 - y0) / fh
sx = (x1 - x0) / fw
cx0 = (C - fw) / 2

def small(a, s):
    a = gaussian_filter(a, s)
    ys = (np.arange(R * 2) + 0.5) / 2 * sy
    xs = (np.arange(int(np.ceil(fw * 2))) + 0.5) / 2 * sx
    return map_coordinates(a, np.meshgrid(ys, xs, indexing="ij"), order=1)

D = small(depth, 3)
P = np.clip(small(person, 2) * 1.2, 0, 1)
L = small(luma, 1.5)
# detalhe local (olhos, bigode, óculos) via passa-alta da luminância
detail = np.clip((L - gaussian_filter(L, 5)) * 4.5 + 0.5, 0, 1)
tone = np.clip((L - L[P > 0.5].mean()) * 1.2 + 0.5, 0, 1)
gy, gx = np.gradient(gaussian_filter(D, 1.2) * 38)
nrm = np.stack([-gx, -gy, np.ones_like(D)])
nrm /= np.linalg.norm(nrm, axis=0)

def sample(a, yy, xx):
    return map_coordinates(a, [yy * 2, xx * 2], order=1, mode="constant")

gy_, gx_ = np.mgrid[0:R, 0:C].astype(np.float32) + 0.5

# ---- chuva ----------------------------------------------------------------
TRAIL = 22
period = R + TRAIL + 6
drops = []
for c in range(C):
    for k in range(2):
        drops.append((c, rng.uniform(0, period), rng.integers(1, 3) * period / FRAMES,
                      rng.integers(10, TRAIL)))
glyph0 = rng.integers(0, NG, (R, C))
flick = np.choose(rng.integers(0, 4, (R, C)), [2, 4, 5, 8])  # divisores de FRAMES => loop perfeito
phase = rng.integers(0, 40, (R, C))

frames = []
for f in range(FRAMES):
    t = f / FRAMES
    yaw = YAW * np.sin(2 * np.pi * t)
    pitch = PITCH * np.sin(4 * np.pi * t + 0.6)

    # warp inverso: a célula de tela mostra o ponto do rosto deslocado pela profundidade
    fx = gx_ - cx0
    fy = gy_.copy()
    for _ in range(4):
        dd = sample(D, fy, fx)
        fx = gx_ - cx0 - np.sin(yaw) * RELIEF * dd
        fy = gy_ - np.sin(pitch) * RELIEF * 0.5 * dd
    dd = sample(D, fy, fx)
    pm = sample(P, fy, fx)
    n = np.stack([sample(nrm[i], fy, fx) for i in range(3)])
    # gira a normal junto com a cabeça
    nx = n[0] * np.cos(yaw) + n[2] * np.sin(yaw)
    nz = -n[0] * np.sin(yaw) + n[2] * np.cos(yaw)
    ny = n[1] * np.cos(pitch) + nz * np.sin(pitch)
    nz = -n[1] * np.sin(pitch) + nz * np.cos(pitch)
    light = np.array([-0.45, -0.55, 0.7]); light /= np.linalg.norm(light)
    lam = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
    rim = np.clip(1 - nz, 0, 1) ** 2
    face = (0.5 * lam ** 1.3 + 0.6 * (sample(detail, fy, fx) - 0.5) + 0.25
            + 0.2 * sample(tone, fy, fx) + 0.3 * rim) * (0.35 + 0.75 * dd)
    face = np.clip(face * 1.35, 0, 1) ** 0.8 * pm

    rain = np.zeros((R, C))
    head = np.zeros((R, C))
    for c, y0_, v, ln in drops:
        hy = (y0_ + v * f) % period - 6
        rows = np.arange(R)
        tr = hy - rows
        m = (tr >= 0) & (tr < ln)
        rain[m, c] = np.maximum(rain[m, c], (1 - tr[m] / ln) ** 1.6)
        hr = int(np.floor(hy))
        if 0 <= hr < R:
            head[hr, c] = 1

    inten = np.maximum(rain * 0.75 * (1 - pm * 0.7), face * (0.8 + 0.5 * rain))
    inten = np.clip(inten + head * (1 - pm * 0.5) * 0.9, 0, 1)
    whiten = np.clip(head + face * rain * 0.4 + np.clip(face - 0.75, 0, 1) * 1.5, 0, 1)

    gi = (glyph0 + (f + phase) // flick) % NG
    tiles = atlas[gi]  # R,C,CH,CW
    img = tiles * inten[:, :, None, None]
    img = img.transpose(0, 2, 1, 3).reshape(R * CH, C * CW)
    wh = np.repeat(np.repeat(whiten, CH, 0), CW, 1) * img
    glow = gaussian_filter(img, 3) * 0.6
    under = Image.fromarray((face * 255).astype(np.uint8)).resize((C * CW, R * CH), Image.BICUBIC)
    under = gaussian_filter(np.asarray(under, dtype=np.float32) / 255, 1.5) * 0.24
    g = np.clip(img * 1.25 + glow + under, 0, 1)
    rgb = np.stack([g * 0.25 + wh * 0.7, g, g * 0.35 + wh * 0.6], -1)
    rgb = np.clip(rgb, 0, 1)
    canvas = np.zeros((H, W, 3))
    canvas[: R * CH, : C * CW] = rgb
    frames.append(Image.fromarray((canvas * 255).astype(np.uint8)))

pal = frames[FRAMES // 4].quantize(colors=32, method=Image.Quantize.MEDIANCUT)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
q[0].save(out, save_all=True, append_images=q[1:], duration=MS, loop=0, optimize=True, disposal=1)
sheet = Image.new("RGB", (W * 2, H * 2))
for i, k in enumerate([0, FRAMES // 4, FRAMES // 2, 3 * FRAMES // 4]):
    sheet.paste(frames[k], ((i % 2) * W, (i // 2) * H))
sheet.save(out.replace(".gif", "_sheet.png"))
print("ok")
