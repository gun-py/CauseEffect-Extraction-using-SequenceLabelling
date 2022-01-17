import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import pickle


import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# reference https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb#scrollTo=uBzDW1FO63pK

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_labels(path: str) -> List[str]:
    print("get_labels:", ["_", "B-C", "I-C", "B-E", "I-E"])
    return ["_", "B-C", "I-C", "B-E", "I-E"]

def get_pos_labels(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
    if "<pad>" not in labels:
        labels = ["<pad>"] + labels
    print('pos_labels:', labels)
    print('Number of pos_labels:', len(labels))
    return labels

class Split(Enum):
    train = "train"
    dev = "val"
    test = "test"

@dataclass
class InputExample:
    #### ADD POS ####
    guid: str
    words: List[str]
    pos_tags: List[str]
    labels: Optional[List[str]]

@dataclass
class InputFeatures:
    #### ADD POS ####

    input_ids: List[int]
    attention_mask: List[int]
    pos_tag_ids: [List[int]]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset
    from transformers import torch_distributed_zero_first

    class FinDataset(Dataset):
        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            local_rank=-1,
        ):
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            with torch_distributed_zero_first(local_rank):
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(data_dir, mode)
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=bool(model_type in ["roberta"]),
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    if local_rank in [-1, 0]:
                        logger.info(f"Saving features into cached file {cached_features_file}")
                        torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    #### ADD POS ####
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        pos_tags = []
        for line in f:
            if line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos_tags=pos_tags))
                    guid_index += 1
                    words = []
                    labels = []
                    pos_tags = []
            else:
                splits = line.split()
                words.append(splits[0])
                pos_tags.append(splits[1])
                if len(splits) > 2:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("_")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos_tags=pos_tags))
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:

    #### ADD POS ####
    pos_labels = get_pos_labels('data/pos_tags.txt')
    pos_map = {label: i for i, label in enumerate(pos_labels)}
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        #### ADD POS ####
        tokens = []
        label_ids = []
        pos_tag_ids = [] #pad_pos_tag = 0 (<pad>)
        pad_token_pos_id = 0

        for word, label, pos in zip(example.words, example.labels, example.pos_tags):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                pos_tag_ids.extend([pos_map[pos]] + [pad_token_pos_id] * (len(word_tokens) - 1))
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            pos_tag_ids = pos_tag_ids[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        pos_tag_ids += [pad_token_pos_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            pos_tag_ids += [pad_token_pos_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            pos_tag_ids += [pad_token_pos_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            pos_tag_ids = [pad_token_pos_id] + pos_tag_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            pos_tag_ids = ([pad_token_pos_id] * padding_length) + pos_tag_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            pos_tag_ids += [pad_token_pos_id] * padding_length

        if len(input_ids) != max_seq_length:
            print('error:', input_ids)
            print(padding_length)
            print(max_seq_length)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("pos_tag_ids: %s", " ".join([str(x) for x in pos_tag_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids, pos_tag_ids=pos_tag_ids
            )
        )
    return features

model_args = ModelArguments(
    model_name_or_path = "bert-base-cased",
)

data_args = DataTrainingArguments(
    data_dir = "data", 
    labels = "data/labels.txt", 
    max_seq_length  = 350
)

training_args = TrainingArguments(
    output_dir="fincausal",
    overwrite_output_dir=True,
    do_train=True,
    #do_eval=True,
    per_gpu_train_batch_size = 4,
    #per_gpu_eval_batch_size=4,
    num_train_epochs = 2,
    #logging_steps=500,
    #logging_first_step=True,
    save_steps = 3000,
    #evaluate_during_training=True,
    do_predict=True
    #add_pos=True,
)

# Set seed
set_seed(training_args.seed)

labels = get_labels(data_args.labels)
label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)

pos_labels = get_pos_labels('data/pos_tags.txt')
pos_label_map: Dict[int, str] = {i: label for i, label in enumerate(pos_labels)}
num_pos_labels = len(pos_labels)

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    id2label=label_map,
    label2id={label: i for i, label in enumerate(labels)},
    id2poslabel=pos_label_map,
    num_pos_labels=num_pos_labels,
    add_pos=training_args.add_pos,
    cache_dir=model_args.cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast,
)

model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

# Get datasets
train_dataset = (
    FinDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.train,
        local_rank=training_args.local_rank,
    )
    if training_args.do_train
    else None
)

eval_dataset = (
    FinDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
        local_rank=training_args.local_rank,
    )
    if training_args.do_eval
    else None
)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Training
if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

# Evaluation
results = {}
if training_args.do_eval and training_args.local_rank in [-1, 0]:
    logger.info("*** Evaluate ***")

    result = trainer.evaluate()

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

        results.update(result)
# Predict
if training_args.do_predict and training_args.local_rank in [-1, 0]:
    test_dataset = FinDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.test,
        local_rank=training_args.local_rank,
    )

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds_list, _ = align_predictions(predictions, label_ids)

    output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key, value in metrics.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

    # Save predictions
    output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
    print(len(preds_list))
    with open(output_test_predictions_file, "w", encoding='utf-8') as writer:
        with open(os.path.join(data_args.data_dir, "test.txt"), "r", encoding='utf-8') as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not preds_list[example_id]:
                        example_id += 1
                elif preds_list[example_id]:
                    output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    pass
 