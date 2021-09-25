import sentencepiece as spm
import numpy as np

t5_tokenizer = spm.SentencePieceProcessor(model_file="sentencepiece.model")  # type: ignore


def token_length(text):
    return len(t5_tokenizer.EncodeAsIds(text))


def get_token_offests(text):
    return np.cumsum(list(map(len, t5_tokenizer.EncodeAsPieces(" " + text))))


def make_inputs(question, search_results, limit=510):
    context = question + " context"
    title, paragraph = "", ""
    limit -= token_length(context)

    for h in search_results["hits"]["hits"]:
        source = h["_source"]

        if title != source["title"] and limit - token_length(source["title"]) > 0:
            title = source["title"]
            context += " " + title
            limit -= token_length(title)
        if paragraph == source["paragraph"]:
            continue

        paragraph = source["paragraph"]
        limit -= token_length(paragraph)
        if limit > 0:
            context += " " + paragraph
        else:
            context += " " + paragraph[: get_token_offests(paragraph)[limit]]
            return context
    return context
