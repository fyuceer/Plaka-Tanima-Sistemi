from __future__ import annotations
import argparse
import csv
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import easyocr
import numpy as np
from ultralytics import YOLO

# =========================================================
# GENEL AYARLAR
# =========================================================

logger = logging.getLogger("plate_recognition")

PLATE_PATTERN = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}\d{2,4}$"
)

OCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CSV_HEADER = [
    "timestamp",
    "plate",
    "confidence"
]

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# =========================================================
# KOMUT SATIRI ARGÜMANLARI
# =========================================================

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Gerçek zamanlı Türk plaka tanıma sistemi."
    )

    parser.add_argument(
        "--model",
        default="best.pt",
        help="YOLO model dosyasının yolu."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Kamera numarası veya video dosyasının yolu."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="YOLO plaka tespit güven eşiği."
    )

    parser.add_argument(
        "--output",
        default="plates.csv",
        help="Plaka kayıtlarının tutulacağı CSV dosyası."
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=30.0,
        help=(
            "Aynı plakanın tekrar kaydedilmesi için "
            "beklenecek süre (saniye)."
        )
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="EasyOCR'ı CPU üzerinde çalıştır."
    )

    return parser.parse_args()

# =========================================================
# OCR METNİNİ TEMİZLEME
# =========================================================

def normalize_plate_text(text: str) -> str:
    
    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )

# =========================================================
# GÖRÜNTÜ ÖN İŞLEME
# =========================================================

def preprocess_plate(
    plate: np.ndarray
) -> np.ndarray:
    
    gray = cv.cvtColor(
        plate,
        cv.COLOR_BGR2GRAY
    )

    _, threshold = cv.threshold(
        gray,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    return threshold

# =========================================================
# PLAKA KAYIT SINIFI
# =========================================================

class PlateLogger:
    
    def __init__(
        self,
        path: Path,
        cooldown: float
    ) -> None:

        self.path = path
        self.cooldown = cooldown

        self.last_seen: dict[str, float] = {}

        if (
            not self.path.exists()
            or self.path.stat().st_size == 0
        ):

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with self.path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    CSV_HEADER
                )

    def save(
        self,
        plate: str,
        confidence: float
    ) -> bool:
       
        current_time = time.monotonic()

        last_time = self.last_seen.get(
            plate
        )

        if (
            last_time is not None
            and current_time - last_time < self.cooldown
        ):

            return False

        self.last_seen[plate] = current_time

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with self.path.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    timestamp,
                    plate,
                    f"{confidence:.2f}"
                ]
            )

        logger.info(
            "Plaka kaydedildi: %s | OCR güven: %.2f",
            plate,
            confidence
        )

        return True

# =========================================================
# PLAKA TANIMA SINIFI
# =========================================================

class PlateRecognizer:
   
    def __init__(
        self,
        model_path: str,
        confidence: float,
        use_gpu: bool
    ) -> None:

        self.confidence = confidence

        self.model = YOLO(
            model_path
        )

        logger.info(
            "YOLO modeli yüklendi: %s",
            model_path
        )

        logger.info(
            "EasyOCR başlatılıyor..."
        )

        self.reader = easyocr.Reader(
            ["en", "tr"],
            gpu=use_gpu
        )

        logger.info(
            "EasyOCR hazır."
        )

    # -----------------------------------------------------
    # PLAKA OKUMA
    # -----------------------------------------------------

    def read_plate(
        self,
        plate_image: np.ndarray
    ) -> tuple[str, float] | None:
        
        processed = preprocess_plate(
            plate_image
        )

        results = self.reader.readtext(
            processed,
            allowlist=OCR_ALLOWLIST
        )

        if not results:
            return None

        results.sort(
            key=lambda result: result[0][0][0]
        )

        raw_text = "".join(
            result[1]
            for result in results
        )

        plate_text = normalize_plate_text(
            raw_text
        )

        ocr_confidence = float(
            np.mean(
                [
                    result[2]
                    for result in results
                ]
            )
        )

        if not PLATE_PATTERN.match(
            plate_text
        ):

            logger.debug(
                "Geçersiz plaka formatı: %s",
                plate_text
            )

            return None

        return (
            plate_text,
            ocr_confidence
        )

    # -----------------------------------------------------
    # FRAME İŞLEME
    # -----------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray
    ):
        
        height, width = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                detection_confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(
                    x1,
                    0
                )

                y1 = max(
                    y1,
                    0
                )

                x2 = min(
                    x2,
                    width
                )

                y2 = min(
                    y2,
                    height
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):

                    continue

                plate_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                plate = self.read_plate(
                    plate_crop
                )

                yield (
                    (x1, y1, x2, y2),
                    detection_confidence,
                    plate
                )

# =========================================================
# GÖRÜNTÜ ÜZERİNE BİLGİ ÇİZME
# =========================================================

def draw_detection(
    frame: np.ndarray,
    bbox,
    detection_confidence: float,
    plate
) -> None:

    x1, y1, x2, y2 = bbox

    # Plaka kutusu.
    cv.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        GREEN,
        2
    )

    cv.putText(
        frame,
        f"Conf: {detection_confidence:.2f}",
        (
            x1,
            max(y1 - 10, 15)
        ),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        RED,
        2,
        cv.LINE_AA
    )

    if plate:

        plate_text, ocr_confidence = plate

        cv.putText(
            frame,
            plate_text,
            (
                x1,
                y2 + 30
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            GREEN,
            3,
            cv.LINE_AA
        )

        cv.putText(
            frame,
            f"OCR: {ocr_confidence:.2f}",
            (
                x1,
                y2 + 55
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2,
            cv.LINE_AA
        )

# =========================================================
# KAMERA / VİDEO KAYNAĞINI AÇMA
# =========================================================

def open_capture(
    source: str
) -> cv.VideoCapture:
    
    if source.isdigit():

        cap = cv.VideoCapture(
            int(source)
        )

    else:

        cap = cv.VideoCapture(
            source
        )

    if not cap.isOpened():

        raise RuntimeError(
            f"Video kaynağı açılamadı: {source}"
        )

    return cap

# =========================================================
# ANA PROGRAM
# =========================================================

def main() -> int:

    args = parse_args()

    # Logging ayarları.
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        )
    )

    # -----------------------------------------------------
    # MODEL KONTROLÜ
    # -----------------------------------------------------

    model_path = Path(
        args.model
    )

    if not model_path.is_file():

        logger.error(
            "YOLO model dosyası bulunamadı: %s",
            model_path
        )

        return 1

    # -----------------------------------------------------
    # PLAKA TANIMA SİSTEMİ
    # -----------------------------------------------------

    try:

        recognizer = PlateRecognizer(
            model_path=str(model_path),
            confidence=args.conf,
            use_gpu=not args.no_gpu
        )

    except Exception as error:

        logger.exception(
            "Plaka tanıma sistemi başlatılamadı: %s",
            error
        )

        return 1

    # -----------------------------------------------------
    # PLAKA KAYIT SİSTEMİ
    # -----------------------------------------------------

    plate_logger = PlateLogger(
        path=Path(args.output),
        cooldown=args.cooldown
    )

    # -----------------------------------------------------
    # KAMERA / VİDEO
    # -----------------------------------------------------

    try:

        cap = open_capture(
            args.source
        )

    except RuntimeError as error:

        logger.error(
            "%s",
            error
        )

        return 1

    logger.info(
        "Plaka tanıma sistemi başlatıldı."
    )

    logger.info(
        "Çıkmak için 'q' tuşuna basın."
    )

    # -----------------------------------------------------
    # ANA DÖNGÜ
    # -----------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                logger.warning(
                    "Kameradan görüntü alınamadı "
                    "veya video sona erdi."
                )

                break

            for (
                bbox,
                detection_confidence,
                plate
            ) in recognizer.process_frame(
                frame
            ):

                if plate:

                    plate_text, ocr_confidence = plate

                    # Plakayı CSV'ye kaydet.
                    plate_logger.save(
                        plate_text,
                        ocr_confidence
                    )

                draw_detection(
                    frame,
                    bbox,
                    detection_confidence,
                    plate
                )

            cv.imshow(
                "Plaka Tanima",
                frame
            )

            if (
                cv.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    except KeyboardInterrupt:

        logger.info(
            "Program kullanıcı tarafından durduruldu."
        )

    except Exception as error:

        logger.exception(
            "Program çalışırken hata oluştu: %s",
            error
        )

    finally:

        cap.release()

        cv.destroyAllWindows()

        logger.info(
            "Kaynaklar serbest bırakıldı."
        )

    return 0

# =========================================================
# PROGRAM BAŞLANGICI
# =========================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )