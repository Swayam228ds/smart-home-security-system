import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

import database

log = logging.getLogger(__name__)

class FaceRecognizerManager:
    def __init__(self, confidence_threshold: float = 120.0):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.confidence_threshold = confidence_threshold
        self.is_trained = False
        self.person_names = {}  # Map person_id -> name

    def load_and_train(self):
        """Load all labeled face crops from the database and train the recognizer."""
        labeled_faces = database.get_all_labeled_faces()
        if not labeled_faces:
            log.info("No labeled faces found for training.")
            return

        faces = []
        labels = []

        for record in labeled_faces:
            img_path = Path(record["image_path"])
            face_path = img_path.with_name(f"{img_path.stem}_face{img_path.suffix}")
            
            if not face_path.exists():
                continue
                
            img = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            person_id = record["person_id"]
            self.person_names[person_id] = record["name"]
            
            faces.append(img)
            labels.append(person_id)

        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.is_trained = True
            log.info(f"Trained LBPHFaceRecognizer on {len(faces)} faces across {len(set(labels))} people.")
        else:
            log.info("No valid face images found for training.")

    def recognize(self, face_image_bgr: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """
        Recognize a face. Returns (person_id, name, confidence) or None.
        Lower confidence distance means a better match.
        """
        if not self.is_trained:
            return None

        gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        label, confidence = self.recognizer.predict(gray)

        log.info(f"LBPH predicted label={label}, distance={confidence:.1f}")

        if confidence <= self.confidence_threshold:
            name = self.person_names.get(label, "Unknown")
            return (label, name, confidence)
        return None

    def add_new_face(self, face_image_path: str, person_id: int, name: str):
        """Update the recognizer with a single new face crop."""
        path = Path(face_image_path)
        if not path.exists():
            return
            
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.person_names[person_id] = name
            if self.is_trained:
                self.recognizer.update([img], np.array([person_id]))
            else:
                self.recognizer.train([img], np.array([person_id]))
                self.is_trained = True
            log.info(f"Updated FaceRecognizer with new face for {name} (ID: {person_id})")

# Global singleton instance
face_recognizer = FaceRecognizerManager()
