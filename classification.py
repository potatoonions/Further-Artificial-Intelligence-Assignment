import json
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "galaxy_cnn.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Map model short labels → display names used by the UI
_LABEL_DISPLAY = {"E": "Elliptical", "S": "Spiral", "SB": "Bar-Spiral"}

_model = None
_metadata = None


def _load():
    """Lazy-load the model and metadata on first call."""
    global _model, _metadata
    if _model is not None:
        return
    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)


def classify(image_input) -> dict:
    """
    Classify a galaxy image using the trained CNN.

    Parameters
    ----------
    image_input : str | Path | PIL.Image.Image
        A file path or an already-loaded PIL Image.
        The image is resized to the model's expected input size automatically.

    Returns
    -------
    dict:
        label         - "Elliptical" | "Spiral" | "Bar-Spiral"
        confidence    - float in [0, 1]
        probabilities - {"Elliptical": float, "Spiral": float, "Bar-Spiral": float}
        error         - None on success, or an error string on failure
    """
    try:
        _load()

        img_size = _metadata["img_size"]
        class_labels = {int(k): v for k, v in _metadata["class_labels"].items()}

        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB").resize((img_size, img_size))
        else:
            img = image_input.convert("RGB").resize((img_size, img_size))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        predictions = _model.predict(arr, verbose=0)[0]
        label_index = int(np.argmax(predictions))
        display_label = _LABEL_DISPLAY[class_labels[label_index]]

        probabilities = {
            _LABEL_DISPLAY[class_labels[i]]: float(predictions[i])
            for i in range(len(predictions))
        }

        return {
            "label": display_label,
            "confidence": float(predictions[label_index]),
            "probabilities": probabilities,
            "error": None,
        }

    except Exception as e:
        return {
            "label": None,
            "confidence": None,
            "probabilities": None,
            "error": str(e),
        }
