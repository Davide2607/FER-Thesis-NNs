import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import tensorflow as tf
import numpy as np

from modules.config import  ACCURACY_RESULTS_PATH, ALL_MODELS_PATHS, \
                            ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_YAML_PATH, \
                            OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_YAML_PATH, OCCLUDED_TEST_SET_PATH, OCCLUDED_TEST_SET_RESIZED_PATH 
from modules.data import generate_h5_from_images, load_data_generator
from modules.model import load_model

# ============== MACROS ===============

PATHS = {
    "ADELE": {
        "test_set": None,
        "test_set_resized": None,
        "test_set_h5": ADELE_TEST_SET_H5_PATH,
        "test_set_yaml": ADELE_TEST_SET_YAML_PATH,
    },
    "OCCLUDED": {
        "test_set": OCCLUDED_TEST_SET_PATH,
        "test_set_resized": OCCLUDED_TEST_SET_RESIZED_PATH,
        "test_set_h5": OCCLUDED_TEST_SET_H5_PATH,
        "test_set_yaml": OCCLUDED_TEST_SET_YAML_PATH,
    }
}

MODEL_PATHS_SUBSET = ALL_MODELS_PATHS
TEST_SET = "ADELE"  # Options: "ADELE", "OCCLUDED"
MODELS_NAMES = ["resnet_finetuning", "pattlite_finetuning", "vgg19_finetuning", "inceptionv3_finetuning", "convnext_finetuning", "efficientnet_finetuning", "yolo_last"]
# MODELS_NAMES = ["resnet_finetuning"]

REDIRECT_OUTPUT = False
LOG_FILE = os.path.join(ACCURACY_RESULTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}_accuracies_{TEST_SET.lower()}.log")

DEBUG = True
# =========== END OF MACROS ===========


# ============== Functions ===============
# will move these to modules/model.py asap

def evaluate_yolo_model(model, test_generator):
    """Evaluate an Ultralytics YOLO model manually over a Keras-style generator.

    Returns (loss, accuracy). Only accuracy is computed; loss is returned
    as None (placeholder) since you only care about accuracy.
    """
    total_samples = 0
    correct = 0

    # Ensure generator iterator state is reset
    try:
        iter(test_generator)
    except Exception:
        pass

    for batch in test_generator:
        # generator yields (X_batch, y_batch)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch = batch[0], batch[1]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")

        # Convert y_batch to integer labels
        if y_batch.ndim > 1 and y_batch.shape[-1] > 1:
            y_int = np.argmax(y_batch, axis=1)
        else:
            y_int = np.array(y_batch).astype(int).reshape(-1)

        # Prepare images for ultralytics (uint8 0-255)
        X = np.array(X_batch)
        if np.issubdtype(X.dtype, np.floating):
            X_for_model = (np.clip(X, 0, 1) * 255).astype(np.uint8)
        else:
            X_for_model = X.astype(np.uint8)

        X_as_list = [X_for_model[i] for i in range(X_for_model.shape[0])]

        # Run prediction (ultralytics handles batching)
        results = model.predict(source=X_as_list, imgsz=128, device=None, verbose=False)

        # results is iterable of per-image Result objects
        batch_preds = []
        for res in results:
            # classification model: Results.probs (per-class probabilities)
            probs = getattr(res, "probs", None)
            if probs is not None:
                pred = probs.top1
                batch_preds.append(pred)

        batch_preds = np.array(batch_preds, dtype=int)
        n = len(y_int)
        # Safety: if predictions count differs from labels, truncate
        if batch_preds.shape[0] != n:
            batch_preds = batch_preds[:n]

        total_samples += n
        correct += int((batch_preds == y_int).sum())

    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return None, accuracy

def evaluate_model(model, model_name, test_generator):
    if "yolo" in model_name:
        test_loss, test_acc = evaluate_yolo_model(model, test_generator)
    else:
        test_loss, test_acc = model.evaluate(test_generator)

    if test_loss is None:
        test_loss = -1.0 
    return test_loss, test_acc

# =========== End Of Functions ===========


# ================= Global ==================

if REDIRECT_OUTPUT:
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout

# Print the LD_LIBRARY_PATH environment variable
ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
print(f"LD_LIBRARY_PATH: {ld_library_path}")

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f" - {gpu}")
    # Set memory growth to avoid allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. The code will run on the CPU.")

# =================== End Of Global =====================


# ================= Main ==================

if __name__ == "__main__":
    # 0) Setup macros as args
    if sys.argv.__len__() == 2:
        TEST_SET = sys.argv[1]
        if TEST_SET not in PATHS.keys():
            print(f"Unknown TEST_SET: {TEST_SET}. Available options: {list(PATHS.keys())}")
            sys.exit(1)
    elif sys.argv.__len__() > 2:
        print("Usage: python evaluate_model.py [TEST_SET]")
        sys.exit(1)

    # 1) Load the test set
    # if you can't find the h5 file, generate it from the images
    if not os.path.exists(PATHS[TEST_SET]["test_set_h5"]):
        generate_h5_from_images(PATHS[TEST_SET]["test_set"], PATHS[TEST_SET]["test_set_resized"], PATHS[TEST_SET]["test_set_h5"])
    test_generator = load_data_generator(PATHS[TEST_SET]["test_set_h5"], 'test')

    print(f"Loaded {TEST_SET} test set with {len(test_generator.x_data)} samples.")

    # 2) Run the evaluations on the test set
    models_results = {name: {"test_loss": None, "test_acc": None} for name in MODELS_NAMES}

    for model_name in MODELS_NAMES:
        print("======================================")
        print(f"Evaluating model: {model_name}")

        if DEBUG:
            print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

        # a) Load the model
        model = load_model(model_name, MODEL_PATHS_SUBSET)
        if DEBUG:
            print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
        if model is None:
            print("Model loading not implemented for this model type.")
            continue
        else:
            # b) Evaluate the model
            test_loss, test_acc = evaluate_model(model, model_name, test_generator)
            models_results[model_name]["test_loss"] = test_loss
            models_results[model_name]["test_acc"] = test_acc
    print("======================================")

    # 3) Print the final results
    print(f"\n\nFinal evaluation results on {TEST_SET.lower()} test set:")
    for model_name, results in models_results.items():
        print(f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}")
    