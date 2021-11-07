## PolEval 2021 submissions

This repository contains winning submissions for [Task 3: Post-correction of OCR results](http://poleval.pl/tasks/task3)
and [Task 4: Question answering challenge](http://poleval.pl/tasks/task4). Both submission rely on fine-tuning the [mT5 
model](https://github.com/google-research/multilingual-t5) on respective tasks.

Solution details are described in the [workshop proceedings](http://poleval.pl/files/poleval2021.pdf)  

---

### Task 3 results

| model    | dev-0  | test-A | test-B |
|:--------:|:------:|:------:|:------:|
| original | 16.550 | 16.527 | 16.543 |
| base     | 4.678  | 4.792  | 4.796  |
| large    | 4.418  | 4.515  | 4.559  |
| XXL      | 3.604  | 3.725  | 3.744  |

---

### Task 4 results

| model | test-B |
|:-----:|:------:|
| base  | 52.12  |
| large | 59.20  |
| XXL   | 71.68  |

---

## Data preparation

### Setup

Common steps for both tasks

1. Install pip requirements

```shell
pip install -r requirements.txt
```

3. Download mT5 vocabulary to repository root

```shell
gsutil cp gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model .
```

3. Prepare GCS bucket for storing training datasets: https://cloud.google.com/storage/docs/creating-buckets
4. Update `gs_base_path` in `config/config.yaml`


### OCR correction

The provided data contains pages of text which are in many instances longer then maximum 
sequence length allowed by the model architecture. To alleviate that the training 
examples are created by aligning and splitting longer input/output pairs. 

1. Pull task repository
```shell
git clone -b secret https://github.com/poleval/2021-ocr-correction.git
```

2. Split examples into chunks to match maximum sequence length
```shell
python3 -m data_preparation.ocr_correction.split_text \
  2021-ocr-correction \
  --length-limit 384
```

3. Upload files to created bucket, update or match paths from `config/task/ocr_correction.yaml`. 
Keep `.index` files to restore full text from predictions

### Question answering

For question answering the model input prompt consists of question and context passages 
retrieved from Wikipedia. This section shows how to reproduce the data used in submission.

The prepared data is available [here](https://drive.google.com/drive/folders/1g_EV1WBhH8kxZdMLXLVWXQ5HnOeeRZkn?usp=sharing).
Skip to step 5 if using this dataset.


1. Pull task repository
```shell
git clone -b secret https://github.com/poleval/2021-question-answering.git
```

2. Start local Elasticsearch instance using docker (skip if using existing cluster)
```shell
docker volume create poleval-es # recommended for persistence
docker run \
  -p 9200:9200 \
  -p 9300:9300 \
  -v poleval-es:/usr/share/elasticsearch/data \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:7.13.4
```

3. Download spaCy model
```shell
python -m spacy download pl_core_news_md
```

4. Index and retrieve context passages for Polish QA dataset
```shell
python3 -m data_preparation.question_answering.quiz_pl \
  2021-question-answering \
  wiki_passages_pl
```

4. Index and retrieve context passages for TriviaQA dataset
```shell
python3 -m data_preparation.question_answering.trivia_qa wiki_passages_en
```

5. Select questions only for prediction
```shell
cat test-B-input-510.tsv | cut -f1 > test-B-questions-510.tsv
```

6. Upload files to created bucket, update or match paths from `config/task/question_answering.yaml`


## Training and evaluation

The models were trained using TPUv3 device. Model configuration is defined in `config/` folder.
After completing the training inference will be run using prompts from files specified under 
`config/task/<task.yaml> -> predict_files`

1. Start TPUv3 and cloud instance eg. using [ctpu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu)

```
ctpu up --name poleval --tpu-size=v3-8 -tf-version 2.5.0 
```

2. SSH to TPU instance, download this repository and install the requirements
3. Start the training (or resume from the latest checkpoint) specifying task and model configuration
```shell
python3 main.py model=xxl task=question_answering +tpu_name=poleval 
```

4. (OCR only) Concatenate the corrected fragments to produce source text

```shell
python3 -m data_preparation.ocr_correction.restore \
  gs://my-bucket/data/ocr/dev-0-input-384.txt-1100000 \
  dev-0-384.index \
  dev-0-restored.txt
```

5. Evaluate results using [geval](https://gitlab.com/filipg/geval) tool

```shell
cd 2021-question-answering # or 2021-ocr-correction
gsutil cp gs://my-bucket/data/polish_qa/test-B-questions-510.tsv-1010000 test-B/out.tsv
./geval --test-name test-B 
```



## Acknowledgments

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
