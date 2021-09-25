import logging
import os

import pandas as pd
import spacy
import tensorflow_datasets as tfds
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from data_preparation.question_answering.common import make_inputs

es = Elasticsearch(os.environ.get("ELASTIC_HOST"))

logging.basicConfig(format="%(asctime)s [%(levelname)-8s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def create_index(index_name):
    return es.indices.create(
        index=index_name,
        body={
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
        },
    )


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
    ds = tfds.load("wikipedia/20201201.pl", split="train")
    nlp = spacy.load("pl_core_news_md", exclude=["ner"])

    for doc in ds.as_numpy_iterator():
        paragraphs = extract_valid_paragraphs(doc)
        for paragraph in nlp.pipe(paragraphs):  # type: ignore
            print(type(paragraph))
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

    def search_lemma(q, idx=index_name):  # "wiki_dump_pl_05"):
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


def prepare_es_index(index_name):
    logger.info("Creating target index")
    create_index(index_name)
    logger.info("Indexing pl wikipedia")
    paragraphs = wiki_paragraph_generator(index_name)
    success, errors = helpers.bulk(es, paragraphs, stats_only=True)
    logger.info("Indexed %d wikipedia passages", success)
    if errors:
        logger.warning("Failed to index %d passages", errors)


def main(base_dir, index_name, length_limit):
    if not es.indices.exists(index_name):
        # prepare_es_index(index_name)
        pass
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

if __name__ == '__main__':
    main("/Users/mateusz.piotrowski/Repositories/2021-question-answering", "wiki_dump_pl_05", 510)

