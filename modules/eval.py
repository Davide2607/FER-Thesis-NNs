import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from modules.config import EMOTIONS


def evaluate_yolo_model_folders(model, test_folder_path, debug=False):
    # test_folder_path should be something like C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\data\datasets\bosphorus_test_finale
    categories = EMOTIONS

    classes = [folder_name for folder_name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, folder_name))]
    classes.sort()
    class_to_number = {class_name: index for index, class_name in enumerate(classes)}

    true_labels = []

    for class_name in classes:
        class_index = class_to_number[class_name]
        class_folder_path = os.path.join(test_folder_path, class_name)
        num_images = len([name for name in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, name))])
        true_labels.extend([class_index] * num_images)

    if debug:
        print("Class to Number Mapping:", class_to_number)
        print("Classes (Alphabetical Order):", classes)
        print("True Labels (Numerical):", true_labels)

    pred_labels = []

    for category in categories:
        # Eseguire la predizione per ogni cartella di immagini
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(f'{os.path.join(test_folder_path, category)}', device=device)

        # Estrarre le probabilità di classe per ogni risultato e aggiungerle alla lista
        pred_labels.extend([result.probs.top1 for result in results])
    
    # Calcolo dell'accuratezza
    accuracy = accuracy_score(true_labels, pred_labels)

    # 5) Return None for loss (not computed), and accuracy
    return None, accuracy

def evaluate_yolo_model_testgen(model, test_generator, debug=False):
    """Evaluate an Ultralytics YOLO model manually over a Keras-style generator.

    Returns (loss, accuracy). Only accuracy is computed; loss is returned
    as None (placeholder) since you only care about accuracy.
    """
    # 0) Ensure generator iterator state is reset
    try:
        iter(test_generator)
    except Exception:
        pass
    
    pred_labels = []
    true_labels = []

    for batch in test_generator:
        # 1) generator yields (X_batch, y_batch)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch = batch[0], batch[1]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")

        # 2a) Convert one-hot y_batch to integer labels
        y_int = np.argmax(y_batch, axis=1)

        # 2b) Listify X_batch
        # X_as_list = [X_for_model[i] for i in range(X_for_model.shape[0])]
        X_as_list = [X_batch[i] for i in range(X_batch.shape[0])]

        # 3a) Run prediction 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(source=X_as_list, device=device)

        # 3b) Save predictions and labels
        pred_labels.extend([result.probs.top1 for result in results])
        true_labels.extend(y_int.tolist())

    # 4) Compute accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # 5) Return None for loss (not computed), and accuracy
    return None, accuracy

def evaluate_model(model, model_name, test_generator, yolo_test_folder_path=None, debug=False):
    if yolo_test_folder_path is not None and test_generator is not None:
        raise ValueError("Provide either yolo_test_folder_path or test_generator, not both.")

    if "yolo" in model_name:
        if yolo_test_folder_path:
            # raise ValueError("I don't want this to be used anymore. Don't provide yolo_test_folder_path, only data_generators.")
            print("Evaluating YOLO model using folder structure instead of test_generator...")
            test_loss, test_acc = evaluate_yolo_model_folders(model, yolo_test_folder_path, debug=debug)
        else:
            test_loss, test_acc = evaluate_yolo_model_testgen(model, test_generator, debug=debug)
    else:
        test_loss, test_acc = model.evaluate(test_generator)

    if test_loss is None:
        test_loss = -1.0 
    return test_loss, test_acc