import argparse
import logging

import pandas as pd
import tensorflow as tf

from data_preparation.ocr_correction.split_text import text_norm

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)


def read_gs_out(path):
    with tf.io.gfile.GFile(path) as f:
        return pd.DataFrame({"pred": [line.strip() for line in f.readlines()]})


def restore_file(output_file, index_file, restored_file):
    logger.info(f"Restoring {output_file} using {index_file}")
    index = read_gs_out(index_file)
    predictions = read_gs_out(output_file)

    predictions.index = list(map(int, index.pred))
    pred_grouped = predictions.groupby(level=0).pred.apply(" ".join).apply(text_norm)

    logger.info(f"Saving to {restored_file}")
    with open(restored_file, "w") as f:
        f.write("\n".join(pred_grouped) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_file", help="Model output with corrected text")
    parser.add_argument(
        "index_file", help="Corresponding index file created by split_text"
    )
    parser.add_argument("restored_file", help="Save path for restored text")

    args = parser.parse_args()

    restore_file(**vars(args))
