import argparse
import logging

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from elasticsearch import Elasticsearch
from tqdm import tqdm

from data_preparation.question_answering.common import make_inputs, prepare_es_index

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

WIKI_EN_INDEX = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "paragraph": {"type": "text"},
        }
    },
}

es = Elasticsearch()


def wiki_paragraph_generator(index_name):
    ds, info = tfds.load("trivia_qa/unfiltered", split="train", with_info=True)
    ds = ds.flat_map(lambda ex: tf.data.Dataset.from_tensor_slices(ex["entity_pages"]))

    seen = set()

    for doc in tqdm(ds.as_numpy_iterator()):
        doc_hash = hash(doc["wiki_context"])
        if doc_hash in seen:
            continue

        for p in doc["wiki_context"].decode("utf-8").split("\n\n"):
            yield {
                "_index": index_name,
                "title": doc["title"].decode("utf-8"),
                "paragraph": p,
            }

        seen.add(doc_hash)


def create_train_data(length_limit):
    ds_qa, info = tfds.load("trivia_qa/unfiltered", split="train", with_info=True)
    train_data = []

    def search(q):
        return es.search(
            index=idx,
            body={"query": {"match": {"paragraph": q}}},
        )

    num_examples = info.splits["train"].num_examples
    for doc in tqdm(ds_qa.as_numpy_iterator(), total=num_examples):
        question = doc["question"].decode("utf-8")
        results = search(question)

        inputs = make_inputs(question, results, int(length_limit * 0.99))

        train_data.append(
            {
                "inputs": inputs.replace("\n", " "),
                "targets": doc["answer"]["value"].decode("utf-8"),
            }
        )

    return pd.DataFrame(train_data)


def main(index_name, length_limit):
    if not es.indices.exists(index_name):
        prepare_es_index(es, index_name, WIKI_EN_INDEX, wiki_paragraph_generator)
    else:
        logger.info("Target index already exists")

    train_data = create_train_data(length_limit)

    out_path = "trivia-qa-train.tsv"
    train_data.to_csv(
        out_path,
        index=False,
        sep="\t",
        header=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare prompt-answer pairs from TriviaQA dataset"
    )

    parser.add_argument("index_name", help="Elasticsearch index name")
    parser.add_argument(
        "--length-limit", type=int, default=510, help="Maximum prompt length"
    )

    args = parser.parse_args()

    main(**vars(args))
