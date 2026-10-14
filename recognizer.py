"""
recognizer.py — Face Recognition tuned for ESP32-CAM (OV2640) frames
====================================================================
Pipeline (applied identically to enrolment crops and live query crops):

  saved face crop (MediaPipe box + 20% margin, ESP32 VGA JPEG)
    → ESP32 clean-up: JPEG-block / sensor-noise denoise, gamma + CLAHE
      (OV2640 auto-exposure is weak; doorways are often back-lit)
    → upscale small crops (VGA faces are only ~90–140 px wide)
    → YuNet 5-point landmark detection → similarity-transform alignment (112×112)
    → SFace deep embedding (128-d), averaged with its mirror image
    → cosine similarity against every enrolled embedding of every person
    → accept only if best ≥ threshold AND clearly ahead of the next person

If the ONNX models cannot be downloaded, it falls back to LBPH with the same
ESP32 pre-processing, fixed-size tight crops and augmentation.

Public interface is unchanged:
  face_recognizer.load_and_train()
  face_recognizer.recognize(face_bgr) -> (person_id, name, distance) | None   (lower = better)
  face_recognizer.add_new_face(face_path, person_id, name)

Run `python recognizer.py --calibrate` once a few visits are labelled to see
genuine/impostor scores on YOUR camera and a suggested threshold.
"""

import cv2
import numpy as np
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import database

log = logging.getLogger(__name__)

# ─── Models (OpenCV Zoo, auto-downloaded on first start like the BlazeFace model) ──
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"      # ~230 KB
YUNET_MODEL_URL  = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = "face_recognition_sface_2021dec.onnx"    # ~37 MB
SFACE_MODEL_URL  = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_recognition_sface/face_recognition_sface_2021dec.onnx")

# ─── Matching thresholds (cosine similarity, SFace) ──────────────────────────
# OpenCV's reference threshold is 0.363 on clean LFW images. ESP32 JPEGs lower
# genuine scores a little, so 0.36 keeps recall while the margin test below
# guards against confusing two enrolled people. Tune with --calibrate.
SFACE_MATCH_THRESHOLD = 0.36
SFACE_MIN_MARGIN      = 0.05   # best person must beat 2nd-best person by this much

# ─── LBPH fallback (chi-square distance on 100×100 CLAHE crops) ─────────────
LBPH_FACE_SIZE        = 100
LBPH_MATCH_THRESHOLD  = 75.0

# ─── ESP32-CAM pre-processing ─────────────────────────────────────────────────
MIN_CROP_SIDE      = 224       # upscale small VGA crops so YuNet landmarks are stable
YUNET_SCORE_THRESH = 0.55      # crop is already verified by MediaPipe; be lenient
CROP_MARGIN_RATIO  = 0.20      # margin state_machine adds around the MediaPipe box


def _ensure_model(path: str, url: str) -> bool:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return True
    try:
        log.info(f"Downloading {p.name} ...")
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(p)
        log.info(f"Downloaded {p.name}.")
        return True
    except Exception as e:
        log.warning(f"Could not download {p.name}: {e}")
        return False


def _adjust_gamma(img: np.ndarray) -> np.ndarray:
    """Pull under/over-exposed OV2640 frames towards mid-grey before CLAHE."""
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    if 0.35 <= mean <= 0.65:
        return img
    gamma = np.log(0.5) / np.log(min(max(mean, 0.05), 0.95))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def preprocess_esp32(face_bgr: np.ndarray) -> np.ndarray:
    """Clean an ESP32-CAM face crop: denoise JPEG artefacts, fix exposure, local contrast, upscale."""
    img = face_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Upscale first so denoising works on smooth blocks rather than raw 8×8 JPEG cells
    h, w = img.shape[:2]
    if min(h, w) < MIN_CROP_SIDE:
        s = MIN_CROP_SIDE / float(min(h, w))
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_CUBIC)

    # OV2640 at jpeg_quality 12: blocking + chroma noise, especially in low light.
    # Edge-preserving bilateral filter (~2 ms; NL-means is ~500 ms and would stall the camera thread)
    img = cv2.bilateralFilter(img, 5, 30, 5)

    img = _adjust_gamma(img)

    # CLAHE on luminance only (keeps colour for SFace, evens out back-lighting)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _tight_center_crop(img: np.ndarray) -> np.ndarray:
    """Strip the 20% margin state_machine adds, leaving just the face box."""
    h, w = img.shape[:2]
    inner = 1.0 / (1.0 + 2 * CROP_MARGIN_RATIO)
    cw, ch = int(w * inner), int(h * inner)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


class FaceRecognizerManager:
    def __init__(self, confidence_threshold: Optional[float] = None):
        self._lock = threading.RLock()
        self.person_names: Dict[int, str] = {}       # person_id -> name
        self.is_trained = False
        self.backend = None                           # "sface" | "lbph" | None

        # SFace gallery: list of (person_id, embedding)
        self._gallery: List[Tuple[int, np.ndarray]] = []
        self._embed_cache: Dict[Tuple[str, float], Optional[np.ndarray]] = {}

        self._detector = None
        self._sface = None
        self._lbph = None

        self.match_threshold = SFACE_MATCH_THRESHOLD
        self.lbph_threshold = confidence_threshold if confidence_threshold is not None else LBPH_MATCH_THRESHOLD

    # ── backend setup ─────────────────────────────────────────────────────────
    def _init_backend(self):
        if self.backend is not None:
            return
        if (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
                and _ensure_model(YUNET_MODEL_PATH, YUNET_MODEL_URL)
                and _ensure_model(SFACE_MODEL_PATH, SFACE_MODEL_URL)):
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    YUNET_MODEL_PATH, "", (320, 320), YUNET_SCORE_THRESH, 0.3, 5000)
                self._sface = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
                self.backend = "sface"
                log.info("Face recognizer backend: YuNet alignment + SFace embeddings.")
                return
            except cv2.error as e:
                log.warning(f"Failed to load SFace/YuNet models: {e}")
        if hasattr(cv2, "face"):
            self.backend = "lbph"
            log.warning("Face recognizer backend: LBPH fallback (install SFace models for better accuracy).")
        else:
            log.error("No face recognition backend available (need OpenCV >= 4.8 or opencv-contrib-python).")

    # ── SFace path ────────────────────────────────────────────────────────────
    def _align(self, img: np.ndarray) -> np.ndarray:
        """Return a 112×112 aligned face using YuNet landmarks, or a tight resize if none found."""
        pad = int(0.15 * max(img.shape[:2]))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = padded.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(padded)
        if faces is not None and len(faces):
            face = max(faces, key=lambda f: f[2] * f[3])
            return self._sface.alignCrop(padded, face)
        log.debug("YuNet found no landmarks in crop; using unaligned tight crop.")
        return cv2.resize(_tight_center_crop(img), (112, 112), interpolation=cv2.INTER_AREA)

    def _embed(self, face_bgr: np.ndarray) -> np.ndarray:
        aligned = self._align(preprocess_esp32(face_bgr))
        f1 = self._sface.feature(aligned).flatten()
        f2 = self._sface.feature(cv2.flip(aligned, 1)).flatten()
        v = f1 / (np.linalg.norm(f1) + 1e-9) + f2 / (np.linalg.norm(f2) + 1e-9)
        return v / (np.linalg.norm(v) + 1e-9)

    def _embed_file(self, path: Path) -> Optional[np.ndarray]:
        key = (str(path), path.stat().st_mtime)
        if key not in self._embed_cache:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self._embed_cache[key] = self._embed(img) if img is not None else None
        return self._embed_cache[key]

    def _person_scores(self, emb: np.ndarray, gallery=None) -> List[Tuple[int, float]]:
        """Best cosine similarity per person, sorted descending."""
        best: Dict[int, float] = {}
        for pid, g in (gallery if gallery is not None else self._gallery):
            s = float(np.dot(emb, g))
            if s > best.get(pid, -1.0):
                best[pid] = s
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    # ── LBPH fallback path ────────────────────────────────────────────────────
    @staticmethod
    def _lbph_prep(face_bgr: np.ndarray) -> np.ndarray:
        img = _tight_center_crop(preprocess_esp32(face_bgr))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (LBPH_FACE_SIZE, LBPH_FACE_SIZE), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _lbph_augment(gray: np.ndarray) -> List[np.ndarray]:
        """Single-shot enrolment → add mirror, exposure and small-tilt variants."""
        out = [gray, cv2.flip(gray, 1)]
        out += [cv2.convertScaleAbs(gray, alpha=a, beta=b) for a, b in ((0.85, -10), (1.15, 10))]
        c = (LBPH_FACE_SIZE / 2, LBPH_FACE_SIZE / 2)
        for ang in (-8, 8):
            M = cv2.getRotationMatrix2D(c, ang, 1.0)
            out.append(cv2.warpAffine(gray, M, gray.shape[::-1], borderMode=cv2.BORDER_REPLICATE))
        return out

    # ── public API ────────────────────────────────────────────────────────────
    def load_and_train(self):
        """Load all labeled face crops from the database and (re)build the recognizer."""
        with self._lock:
            self._init_backend()
            if self.backend is None:
                return

            labeled_faces = database.get_all_labeled_faces()
            if not labeled_faces:
                log.info("No labeled faces found for training.")
                return

            names: Dict[int, str] = {}
            gallery: List[Tuple[int, np.ndarray]] = []
            lbph_faces, lbph_labels = [], []

            for record in labeled_faces:
                img_path = Path(record["image_path"])
                face_path = img_path.with_name(f"{img_path.stem}_face{img_path.suffix}")
                if not face_path.exists():
                    continue
                pid = record["person_id"]
                try:
                    if self.backend == "sface":
                        emb = self._embed_file(face_path)
                        if emb is None:
                            continue
                        gallery.append((pid, emb))
                    else:
                        img = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        for g in self._lbph_augment(self._lbph_prep(img)):
                            lbph_faces.append(g)
                            lbph_labels.append(pid)
                except cv2.error as e:
                    log.warning(f"Skipping {face_path.name}: {e}")
                    continue
                names[pid] = record["name"]

            self.person_names = names
            if self.backend == "sface":
                self._gallery = gallery
                self.is_trained = bool(gallery)
                count = len(gallery)
            else:
                if lbph_faces:
                    rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
                    rec.train(lbph_faces, np.array(lbph_labels))
                    self._lbph = rec
                self.is_trained = bool(lbph_faces)
                count = len(lbph_faces)

            if self.is_trained:
                log.info(f"Trained {self.backend} recognizer on {count} samples across {len(names)} people.")
            else:
                log.info("No valid face images found for training.")

    def recognize(self, face_image_bgr: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """
        Recognize a face. Returns (person_id, name, distance) or None.
        Lower distance means a better match (SFace: (1 - cosine) * 100).
        """
        with self._lock:
            if not self.is_trained or face_image_bgr is None or face_image_bgr.size == 0:
                return None
            try:
                if self.backend == "sface":
                    scores = self._person_scores(self._embed(face_image_bgr))
                    if not scores:
                        return None
                    pid, best = scores[0]
                    second = scores[1][1] if len(scores) > 1 else -1.0
                    log.info(f"SFace best={self.person_names.get(pid)} sim={best:.3f} "
                             f"2nd={second:.3f} (thr={self.match_threshold:.2f})")
                    if best >= self.match_threshold and (best - second) >= SFACE_MIN_MARGIN:
                        return (pid, self.person_names.get(pid, "Unknown"), (1.0 - best) * 100.0)
                    return None

                gray = self._lbph_prep(face_image_bgr)
                label, dist = self._lbph.predict(gray)
                label_f, dist_f = self._lbph.predict(cv2.flip(gray, 1))
                if dist_f < dist:
                    label, dist = label_f, dist_f
                log.info(f"LBPH predicted label={label}, distance={dist:.1f} (thr={self.lbph_threshold:.1f})")
                if dist <= self.lbph_threshold:
                    return (label, self.person_names.get(label, "Unknown"), dist)
                return None
            except cv2.error as e:
                log.warning(f"Face recognition failed: {e}")
                return None

    def add_new_face(self, face_image_path: str, person_id: int, name: str):
        """Enrol a newly labeled face. Rebuilds from the DB so relabels/renames are honoured."""
        if not Path(face_image_path).exists():
            return
        self.load_and_train()
        log.info(f"Updated FaceRecognizer with new face for {name} (ID: {person_id})")

    # ── calibration helper ───────────────────────────────────────────────────
    def calibrate(self):
        """Leave-one-out genuine/impostor similarity on the labelled DB faces."""
        self.load_and_train()
        if self.backend != "sface" or len(self._gallery) < 2:
            print("Need the SFace backend and at least 2 labelled faces to calibrate.")
            return
        genuine, impostor = [], []
        for i, (pid, emb) in enumerate(self._gallery):
            rest = self._gallery[:i] + self._gallery[i + 1:]
            for opid, s in self._person_scores(emb, rest):
                (genuine if opid == pid else impostor).append(s)
        print(f"Samples: {len(self._gallery)} faces, {len(self.person_names)} people")
        if genuine:
            print(f"Genuine  sim: min={min(genuine):.3f} median={np.median(genuine):.3f}")
        if impostor:
            print(f"Impostor sim: max={max(impostor):.3f} median={np.median(impostor):.3f}")
        if genuine and impostor:
            suggested = float(np.clip((np.percentile(genuine, 10) + max(impostor)) / 2, 0.30, 0.55))
            print(f"Suggested SFACE_MATCH_THRESHOLD ≈ {suggested:.2f} (current {self.match_threshold:.2f})")


# Global singleton instance
face_recognizer = FaceRecognizerManager()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if "--calibrate" in sys.argv:
        face_recognizer.calibrate()
    else:
        print(__doc__)
