# Standard imports
import csv
import json
import os

# 3rd Party imports
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


# Local imports
from args import get_train_test_args
from dataset import get_dataset
from trainer import Trainer
import util

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)

    # Determine device
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:

        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")

        trainer = Trainer(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
