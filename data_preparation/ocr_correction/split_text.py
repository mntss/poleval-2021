import argparse
import difflib
import logging
import lzma
import os

import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

tqdm.pandas()

sentencepiece_model = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../sentencepiece.model")
)
# TODO gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model
tokenizer = spm.SentencePieceProcessor(model_file=sentencepiece_model)  # type: ignore


logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)


def read_file(fname):
    output = []

    logger.info("Reading %s", fname)
    with lzma.open(fname) as fh:
        for line in fh:
            line = line.decode("utf-8")
            in_text = line.replace("\\n", "\n").replace("\\\\", "\\").strip("\n")
            output.append(in_text)
    return output


def read_split(base_dir, split):
    inputs = [
        line.split("\t")[-1]
        for line in read_file(os.path.join(base_dir, split, "in.tsv.xz"))
    ]

    data = {"inputs": inputs}

    out_file = os.path.join(base_dir, split, "expected.tsv.xz")
    if os.path.exists(out_file):
        data["targets"] = read_file(out_file)
    return pd.DataFrame(data)


def get_token_offests(text):
    return np.cumsum(list(map(len, tokenizer.EncodeAsPieces(" " + text))))


def align_example(inputs, targets, limit, match_th):
    limit = int(limit * 0.99)
    inp_off, tgt_off = get_token_offests(inputs), get_token_offests(targets)

    if len(inp_off) < limit and len(tgt_off) < limit:
        return [(inputs, targets)]

    inp_limit = len(inputs) if len(inp_off) < limit else inp_off[limit - 1]
    tgt_limit = len(targets) if len(tgt_off) < limit else tgt_off[limit - 1]

    matches = difflib.SequenceMatcher(a=inputs, b=targets).get_matching_blocks()

    try:
        m = next(
            m
            for m in reversed(matches)
            if m.a < inp_limit
            and min(inp_limit - m.a, m.size) > match_th
            and m.b < tgt_limit
            and min(tgt_limit - m.b, m.size) > match_th
            and " " in inputs[m.a : min(m.a + m.size, inp_limit)]
        )
    except StopIteration:
        return [(inputs, targets)]

    split_offset = inputs[m.a : min(m.a + m.size, inp_limit)].rfind(" ")
    aligned = inputs[: m.a + split_offset], targets[: m.b + split_offset]

    rest = align_example(
        inputs[m.a + split_offset :],
        targets[m.b + split_offset :],
        limit,
        match_th,
    )

    return [aligned] + rest


def text_norm(t):
    return " ".join(t.split())


def make_data(data, length_limit, match_threshold, sim_threshold=0.4, min_length=20):
    def split_example(row):
        return align_example(
            row["inputs"],
            row["targets"],
            length_limit,
            match_threshold,
        )

    def sim_ratio(row):
        return difflib.SequenceMatcher(a=row["inputs"], b=row["targets"]).ratio()

    logger.info("Splitting %d examples into aligned fragments", len(data))

    data_aligned = (
        data.applymap(text_norm)
        .progress_apply(split_example, axis=1)
        .explode()
        .apply(pd.Series)
        .rename(columns={0: "inputs", 1: "targets"})
    )

    length = data_aligned.applymap(tokenizer.EncodeAsIds).applymap(len)

    logger.info(
        "Created %d fragments, calculating similarity statistics", len(data_aligned)
    )

    data_meta = data_aligned.assign(
        ratio=length.eval("inputs / targets"),
        len_inputs=length["inputs"],
        len_targets=length["targets"],
    ).assign(sim_ratio=lambda df: df.progress_apply(sim_ratio, axis=1))

    data_filtered = data_meta.query(
        f"ratio < 1.2 & ratio > 0.8 & sim_ratio > {sim_threshold} "
        f"& ({min_length} < len_inputs < {length_limit}) "
        f"& ({min_length} < len_targets < {length_limit})"
    )

    removed_examples = len(data_aligned) - len(data_filtered)
    if removed_examples:
        logger.warning("Removed %d examples due to low similarity", removed_examples)

    return data_filtered[["inputs", "targets"]]


def token_split(text, limit):
    limit = int(limit * 0.99)
    offsets = get_token_offests(text)
    if len(offsets) - 1 < limit:
        return [text]

    char_limit = text[: offsets[limit]].rfind(" ")
    if len(text) - char_limit < 30:  # adjust to avoid short reminders
        char_limit = len(text) // 2

    if char_limit == -1:
        return [text]
    return [text[:char_limit]] + token_split(text[char_limit + 1 :], limit)


def main(base_dir, length_limit, match_threshold):
    train_data = read_split(base_dir, "train")
    train_examples = make_data(train_data, length_limit, match_threshold)

    train_examples.to_csv(
        f"train-examples-{length_limit}.tsv",
        index=False,
        sep="\t",
        header=False,
    )

    for split in ["dev-0", "test-A", "test-B"]:
        logger.info("Processing %s split", split)
        data = read_split(base_dir, split)

        fragments = (
            data.inputs.apply(text_norm)
            .apply(token_split, limit=length_limit)
            .explode()
            .to_frame()
        )

        fragments.to_csv(
            f"{split}-input-{length_limit}.txt",
            index=False,
            sep="\t",
            header=False,
        )

        with open(f"{split}-{length_limit}.index", "w") as f:
            f.write("\n".join(map(str, fragments.index)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", help="OCR task repository directory")
    parser.add_argument(
        "--length-limit", type=int, default=384, help="Maximum example length"
    )
    parser.add_argument(
        "--match-threshold",
        type=int,
        default=20,
        help="Minimum match length to consider a split",
    )

    args = parser.parse_args()

    main(**vars(args))
