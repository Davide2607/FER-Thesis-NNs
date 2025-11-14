import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import argparse

from modules.config import ADELE_TEST_SET_H5_PATH, BOSPHORUS_TEST_HQ_H5_PATH
from modules.data import load_data_generator


# Argument parser setup
parser = argparse.ArgumentParser(description="Test data generator with occlusion layer.")
parser.add_argument("-o", "--occlusion_probability", type=float, required=True, help="Probability of occlusion (float).")
parser.add_argument("-p", "--parallelize_masking", type=bool, required=True, help="Whether to parallelize masking (boolean).")

# Parse arguments
args = parser.parse_args()

# Access arguments
occlusion_probability = args.occlusion_probability
parallelize_masking = args.parallelize_masking

#        1161 x 1161               128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH 
DATASET = ADELE_TEST_SET_H5_PATH     

if __name__ == "__main__":
    print(f"Testing data generator with occlusion_probability={occlusion_probability}, parallelize_masking={parallelize_masking}")

    # 1) First time run with occlusions
    start_time = time.time() 
    test_generator = load_data_generator(DATASET, 'test', occlusion_probability, parallelize_masking)

    for batch in test_generator:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch, batch_paths = batch[0], batch[1], batch[2]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
        
        print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}, Batch paths: {batch_paths}")

        for image in X_batch:
            pass
            # print(f"Image shape: {image.shape}")
            # plot_image(image)
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # # 2) Second time run without occlusions
    # start_time = time.time()
    # test_generator = load_data_generator(DATASET, 'test', 0.0)
    
    # for batch in test_generator:
    #     if isinstance(batch, (list, tuple)) and len(batch) >= 2:
    #         X_batch, y_batch, batch_paths, batch_x_hashes = batch[0], batch[1], batch[2], batch[3]
    #     else:
    #         raise ValueError("test_generator must yield (X_batch, y_batch) tuples")

    #     print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}, Batch paths: {batch_paths.shape if batch_paths is not None else 'None'}, Batch hashes: {batch_x_hashes.shape}")

    #     for image in X_batch:
    #         pass
    #         # print(f"Image shape: {image.shape}")
    #         # plot_image(image)

    # end_time = time.time()  # Record the end time
    # elapsed_time = end_time - start_time  # Calculate elapsed time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds")

