import argparse
import os
import numpy as np
import torch

from recsys.datasets.avazu import CAT_FEATURE_COUNT, AvazuIterDataPipe, TOTAL_TRAINING_SAMPLES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="path to the dir where the csv file train is downloaded and unzipped")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="path to which the train/val/test splits are saved")

    parser.add_argument("--is_split", action='store_true')
    return parser.parse_args()


def main():
    # Refactored and added additional processing from the original version to properly preprocess dense features in the Avazu dataset
    print("starting parsing args")
    args = parse_args()
    print("done parsing args")
    if args.is_split:
        if not os.path.exists(args.input_dir):
            raise ValueError(f"{args.input_dir} has existed")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        print("finished making appropriate directories")
        idx = 0

        for _t in ("sparse", "dense", 'label'):
            if (idx % 1000000 == 0):
                print ("processing row = ", idx)
            npy = np.load(os.path.join(args.input_dir, f"{_t}.npy"))
            train_split = npy[:TOTAL_TRAINING_SAMPLES]
            np.save(os.path.join(args.output_dir, f"train_{_t}.npy"), train_split)
            val_test_split = npy[TOTAL_TRAINING_SAMPLES:]
            np.save(os.path.join(args.output_dir, f"val_test_{_t}.npy"), val_test_split)
            del npy

    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        sparse_output_file_path = os.path.join(args.output_dir, "sparse.npy")
        dense_output_file_path = os.path.join(args.output_dir, "dense.npy")
        label_output_file_path = os.path.join(args.output_dir, "label.npy")
        print("beginning file splitting")
        idx = 0
        sparse, dense, labels = [], [], []
        for row_sparse, row_dense, row_label in AvazuIterDataPipe(args.input_dir):
            if (idx % 1000000 == 0):
                print ("processing row = ", idx)
            sparse.append(row_sparse)
            dense.append(row_dense)
            labels.append(row_label)
            idx += 1
        print("done processing, now converting to numpy arrays")
        sparse_np = np.array(sparse, dtype=np.int32)
        del sparse
        dense_np = np.array(dense, dtype=np.float32)
        del dense
        labels_np = np.array(labels, dtype=np.int32).reshape(-1, 1)
        del labels
        print("done converting, now writing to output files")
        for f_path, arr in [(sparse_output_file_path, sparse_np), (dense_output_file_path, dense_np), (label_output_file_path, labels_np)]:
            np.save(f_path, arr)
    print("done!")


if __name__ == "__main__":
    main()
