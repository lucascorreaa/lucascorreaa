"""Gera o mapa de profundidade + luminância do rosto a partir de uma foto.

Dependências: pip install mediapipe pillow numpy scipy
Uso: python scripts/depth_map.py foto.jpg face_landmarker.task selfie_multiclass.tflite saida.npz
Modelos MediaPipe:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
"""
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageFilter
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter

photo, face_model, seg_model, out = sys.argv[1:5]
img = Image.open(photo).convert("RGB")
W, H = img.size
mp_img = mp.Image.create_from_file(photo)

# Segmentação: 0 fundo, 1 cabelo, 2 pele corpo, 3 pele rosto, 4 roupa, 5 outros
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=seg_model), output_category_mask=True))
cat = seg.segment(mp_img).category_mask.numpy_view().squeeze()
person = (cat > 0).astype(np.float32)
hair = (cat == 1).astype(np.float32)

# Landmarks 3D do rosto
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model), num_faces=1))
pts = lm.detect(mp_img).face_landmarks[0]
xy = np.array([[p.x * W, p.y * H] for p in pts])
z = -np.array([p.z for p in pts]) * W  # positivo = perto da câmera

# Corpo/cabelo: volume arredondado a partir da distância até a borda
dist = distance_transform_edt(person)
body = np.sqrt(np.clip(dist / 180.0, 0, 1))
depth = body * 60.0

# Rosto: interpola os landmarks e soma ao volume
yy, xx = np.mgrid[0:H:4, 0:W:4]
face_z = griddata(xy, z, (xx, yy), method="linear")
face_mask = ~np.isnan(face_z)
face_z = np.nan_to_num(face_z, nan=0.0)
fz = np.array(Image.fromarray(face_z.astype(np.float32)).resize((W, H), Image.BILINEAR))
fm = np.array(Image.fromarray((face_mask * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
fm = gaussian_filter(fm, 25)
fz = fz - fz[fm > 0.5].min()
depth = depth + fm * (fz + 40) + hair * gaussian_filter(hair, 40) * 30
depth = gaussian_filter(depth, 6) * person
depth /= depth.max()

luma = np.array(img.convert("L"), dtype=np.float32) / 255.0
np.savez_compressed(out, depth=depth.astype(np.float32), luma=luma.astype(np.float32),
                    person=person, face=fm.astype(np.float32))
Image.fromarray((depth * 255).astype(np.uint8)).save(out.replace(".npz", "_depth.png"))
print("ok", W, H)
