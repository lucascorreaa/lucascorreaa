"""Pin art na chuva Matrix: os caracteres da chuva são empurrados para a frente
pelo relevo do rosto, como pinos, e a cena gira em perspectiva 3D.

Uso: python scripts/matrix_face.py face.npz assets/matrix-face.gif
(face.npz vem de scripts/depth_map.py)
Depois: gifsicle -O3 --lossy=40 assets/matrix-face.gif -o assets/matrix-face.gif
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

src, out = sys.argv[1:3]
rng = np.random.default_rng(7)

# ---- parâmetros -----------------------------------------------------------
W, H = 840, 480               # tamanho do GIF
CW, CH = 5, 8                 # espaçamento dos pinos (px)
FRAMES, MS = 64, 75           # quadros e duração de cada um (loop de 4,8 s)
CROP = (70, 40, 1070, 1330)   # recorte (x0, y0, x1, y1) da foto original
ZMAX = 110.0                  # quanto o rosto empurra os pinos (px)
FOCAL = 900.0                 # distância focal da câmera
YAW, PITCH = 0.32, 0.08       # amplitude do giro da cena (rad)
LIGHT = np.array([-0.6, -0.5, 0.62])
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

# a parede de pinos é maior que o quadro para as bordas não aparecerem ao girar
GW, GH = int(W * 1.5), int(H * 1.6)
C, R = GW // CW, GH // CH
FR = H // CH  # linhas visíveis (o rosto ocupa essa altura)
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# ---- glifos em vários tamanhos (perspectiva) ------------------------------
chars = [chr(c) for c in range(0xFF66, 0xFF9E)] + list("0123456789:=*+-<>¦|Z")
NG = len(chars)
SIZES = (8, 9, 10, 11, 12, 13)
atlases, bolds = [], []
for size in SIZES:
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
        ImageDraw.Draw(g).text((tw / 2, th / 2), ch, font=font, fill=255, anchor="mm",
                               stroke_width=1, stroke_fill=80)
        bold.append(np.asarray(g.transpose(Image.FLIP_LEFT_RIGHT), dtype=np.float32) / 255)
    bolds.append(np.stack(bold))

# ---- relevo do rosto na grade de pinos ------------------------------------
d = np.load(src)
x0, y0, x1, y1 = CROP
depth, luma, person = (d[k][y0:y1, x0:x1] for k in ("depth", "luma", "person"))

# vinheta: o corpo volta suavemente para o plano, só a cabeça sai da parede
vy, vx = np.mgrid[y0:y1, x0:x1]
ell = ((vx - 565) / 460) ** 2 + ((vy - 660) / 620) ** 2
person = person * np.clip((1.2 - ell) / 0.4, 0, 1) ** 1.5

fh = FR * 0.98
fw = fh * (x1 - x0) / (y1 - y0) * (CH / CW)
sy, sx = (y1 - y0) / fh, (x1 - x0) / fw
cx0, cy0 = (C - fw) / 2, (R - fh) / 2
rr, cc = np.mgrid[0:R, 0:C].astype(np.float32)
src_y = (rr + 0.5 - cy0) * sy
src_x = (cc + 0.5 - cx0) * sx


def to_grid(a, s):
    return map_coordinates(gaussian_filter(a, s), [src_y, src_x], order=1, mode="constant")


P = np.clip(to_grid(person, 2) * 1.3, 0, 1)
Z = np.clip(to_grid(depth, 3), 0, 1) * P
L = to_grid(luma, 2)
lo, hi = np.percentile(L[P > 0.6], [4, 97])
alb = np.clip((L - lo) / (hi - lo), 0, 1) ** 0.8
tex = np.clip((L - gaussian_filter(L, 3)) * 3 + 0.5, 0, 1)
refl = np.clip((0.12 + 0.88 * alb) * (0.6 + 0.8 * (tex - 0.5)), 0, 1)  # refletância do pino
refl = np.clip(refl / np.percentile(refl[P > 0.5], 92), 0, 1.2)

# normais da superfície dos pinos (em px de mundo)
gz_y, gz_x = np.gradient(gaussian_filter(Z, 0.8) * ZMAX)
nrm = np.stack([-gz_x / CW, -gz_y / CH, np.ones_like(Z)])
nrm /= np.linalg.norm(nrm, axis=0)

# posição 3D dos pinos (origem no centro da tela)
X = (cc + 0.5) * CW - GW / 2
Y = (rr + 0.5) * CH - GH / 2
Zw = Z * ZMAX

# ---- chuva (loop perfeito: cada gota percorre períodos inteiros) ----------
TRAIL = 26
period = R + TRAIL + 4
drops = [(c, rng.uniform(0, period), rng.integers(10, TRAIL))
         for c in range(C) for _ in range(3)]
v = period / FRAMES           # uma volta por loop => chuva lenta
glyph0 = rng.integers(0, NG, (R, C))
flick = np.choose(rng.integers(0, 3, (R, C)), [4, 8, 16])  # divisores de FRAMES
phase = rng.integers(0, FRAMES, (R, C))

PAD = 24
frames = []
for f in range(FRAMES):
    t = f / FRAMES
    yaw = YAW * np.sin(2 * np.pi * t)
    pitch = PITCH * np.sin(2 * np.pi * t + np.pi / 2)
    cyw, syw, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)

    # gira a cena (em torno do plano da parede) e projeta em perspectiva
    Xr = X * cyw + Zw * syw
    Zr = -X * syw + Zw * cyw
    Yr = Y * cp - Zr * sp
    Zr = Y * sp + Zr * cp
    k = FOCAL / (FOCAL - Zr)
    px = W / 2 + Xr * k
    py = H / 2 + Yr * k

    # iluminação: luz fixa na câmera, normais giram com a cena
    nx = nrm[0] * cyw + nrm[2] * syw
    nz = -nrm[0] * syw + nrm[2] * cyw
    ny = nrm[1] * cp - nz * sp
    nz = nrm[1] * sp + nz * cp
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    lit = np.clip((0.3 + 1.1 * lam ** 1.3) * refl * (0.45 + 0.75 * Z ** 0.7) / 0.55, 0, 1) ** 0.75

    # chuva: rastro caindo em cada coluna
    rain = np.zeros((R, C))
    head = np.zeros((R, C))
    rows = np.arange(R)
    for c, oy, ln in drops:
        hy = (oy + v * f) % period - 4
        tr = hy - rows
        m = (tr >= 0) & (tr < ln)
        rain[m, c] = np.maximum(rain[m, c], (1 - tr[m] / ln) ** 1.5)
        hr = int(np.floor(hy))
        if 0 <= hr < R:
            head[hr, c] = 1

    # pino no plano: só a chuva; pino empurrado: sempre visível, a chuva o acende
    on_face = P > 0.25
    inten = np.where(on_face, lit * (0.55 + 0.75 * rain) + head * 0.5 * lit,
                     rain * 0.6 + head * 0.9)
    white = np.clip(head + np.where(on_face, rain * lit * 0.9, 0), 0, 1)
    size_i = np.clip(np.round((k - 1) * 30 + 1), 0, len(SIZES) - 1).astype(int)
    size_i = np.where(on_face, np.maximum(size_i, 1), 0)
    gi = (glyph0 + (f + phase) // flick) % NG

    gcan = np.zeros((H + 2 * PAD, W + 2 * PAD))
    wcan = np.zeros_like(gcan)
    for idx in np.argsort(Zr.ravel()):  # do fundo para a frente
        r, c = divmod(idx, C)
        val = inten[r, c]
        if val < 0.035:
            continue
        tile = (bolds if on_face[r, c] else atlases)[size_i[r, c]][gi[r, c]]
        th, tw = tile.shape
        yy = int(PAD + py[r, c] - th / 2)
        xx = int(PAD + px[r, c] - tw / 2)
        if not (0 <= yy < H + 2 * PAD - th and 0 <= xx < W + 2 * PAD - tw):
            continue
        sl = (slice(yy, yy + th), slice(xx, xx + tw))
        if on_face[r, c]:  # pino da frente cobre o de trás
            gcan[sl] = np.where(tile > 0.05, tile * val, gcan[sl] * 0.35)
            wcan[sl] = np.where(tile > 0.05, tile * val * white[r, c], wcan[sl] * 0.35)
        else:
            gcan[sl] = np.maximum(gcan[sl], tile * val)
            wcan[sl] = np.maximum(wcan[sl], tile * val * white[r, c])
    g = gcan[PAD:-PAD, PAD:-PAD]
    w = wcan[PAD:-PAD, PAD:-PAD]
    g = np.clip(g * 1.25 + gaussian_filter(g, 1.5) * 0.4, 0, 1)
    rgb = np.clip(np.stack([g * 0.12 + w * 0.7, g, g * 0.3 + w * 0.55], -1), 0, 1)
    frames.append(Image.fromarray((rgb * 255).astype(np.uint8)))

pal = frames[FRAMES // 4].quantize(colors=24, method=Image.Quantize.MEDIANCUT)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
q[0].save(out, save_all=True, append_images=q[1:], duration=MS, loop=0, optimize=True, disposal=1)

# folha de contato para conferência
sheet = Image.new("RGB", (W * 2, H * 2))
for i, fr in enumerate([0, FRAMES // 4, FRAMES // 2, 3 * FRAMES // 4]):
    sheet.paste(frames[fr], ((i % 2) * W, (i // 2) * H))
sheet.save(out.replace(".gif", "_sheet.png"))
print("ok")
