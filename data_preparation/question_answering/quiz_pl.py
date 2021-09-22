import argparse
import logging
import os

import pandas as pd
import spacy
import tensorflow_datasets as tfds
from elasticsearch import Elasticsearch
from tqdm import tqdm

from data_preparation.question_answering.common import make_inputs, prepare_es_index

logging.basicConfig(format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

WIKI_PL_INDEX = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "standard",
                }
            },
        }
    },
    "mappings": {
        "properties": {
            "lemma": {"type": "text", "analyzer": "whitespace"},
            "title": {"type": "text"},
            "paragraph": {"type": "text"},
        }
    },
}

es = Elasticsearch(os.environ.get("ELASTIC_HOST"))


def extract_valid_paragraphs(doc, min_length=20):
    meta_sections = {"Linki zewnętrzne", "Przypisy", "Zobacz też"}

    paragraphs = doc["text"].decode("utf-8").split("\n")
    result = ""
    for p in paragraphs:
        if p.strip() in meta_sections:
            return

        if p.startswith("Kategoria:") or p.startswith("PATRZ"):
            continue

        if p:
            result += " " + p
        elif result:
            if len(result) > min_length:
                yield result.strip()
            result = ""

    if result:
        yield result


def spacy_encode(text_nlp):
    tokens = [
        w.lemma_.lower().strip() for w in text_nlp if not w.is_stop and not w.is_punct
    ]
    return " ".join(filter(bool, tokens))


def wiki_paragraph_generator(idx):
    ds, info = tfds.load("wikipedia/20201201.pl", split="train", with_info=True)
    nlp = spacy.load("pl_core_news_md", exclude=["ner"])

    for doc in tqdm(ds.as_numpy_iterator(), total=info.splits["train"].num_examples):
        paragraphs = extract_valid_paragraphs(doc)
        for paragraph in nlp.pipe(paragraphs):  # type: ignore
            yield {
                "_index": idx,
                "title": doc["title"].decode("utf-8"),
                "paragraph": str(paragraph),
                "lemma": spacy_encode(paragraph),
            }


def read_data(base_dir, split):
    ret = {}
    question_path = os.path.join(base_dir, split, "in.tsv")
    logger.info("Reading questions from %s", question_path)
    ret["question"] = pd.read_csv(
        question_path,
        sep="\t",
        header=None,
        squeeze=True,
    )

    ans_path = os.path.join(base_dir, split, "expected.tsv")
    if os.path.exists(ans_path):
        logger.info("Reading answers from %s", ans_path)

        ret["answer"] = pd.read_csv(
            ans_path,
            sep="\t\t",
            header=None,
            squeeze=True,
            engine="python",
        )
    return pd.DataFrame(ret)


def select_ans(answers):
    aliases = answers.split("\t")
    for ans in aliases:
        if ans.isnumeric():
            return ans

    return aliases[0]


def retrieve_context(df, index_name, length_limit):
    nlp = spacy.load("pl_core_news_md", exclude=["ner"])

    def search_lemma(q, idx=index_name):
        q = nlp(q)
        return es.search(
            index=idx,
            body={
                "query": {"match": {"lemma": {"query": spacy_encode(q)}}},
            },
        )

    ret = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out = {}

        results = search_lemma(row["question"])
        out["inputs"] = make_inputs(row["question"], results, int(length_limit * 0.99))

        if "answer" in row:
            out["targets"] = select_ans(row["answer"])
        ret.append(out)
    return pd.DataFrame(ret)


def main(base_dir, index_name, length_limit):
    if not es.indices.exists(index_name):
        prepare_es_index(es, index_name, WIKI_PL_INDEX, wiki_paragraph_generator)
    else:
        logger.info("Target index already exists")

    for split in ["dev-0", "test-A", "test-B"]:
        data = read_data(base_dir, split)

        logger.info("Searching relevant passages")
        question_with_context = retrieve_context(data, index_name, length_limit)

        out_path = f"{split}-input-{length_limit}.tsv"
        logger.info("Writing %s", out_path)
        question_with_context.to_csv(
            out_path,
            index=False,
            sep="\t",
            header=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", help="Directory of QA repository")
    parser.add_argument("index_name", help="Elasticsearch index name")
    parser.add_argument(
        "--length-limit", type=int, default=510, help="Maximum prompt length"
    )

    args = parser.parse_args()
    main(**vars(args))
