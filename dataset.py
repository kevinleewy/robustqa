# Standard imports
import json
import os
from pathlib import Path
from collections import defaultdict as ddict

# 3rd Party imports
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Local imports
from categories import CATEGORIES, getCategoryIds
import util

class QADataset(Dataset):
    def __init__(self, encodings, train=True):
        self.encodings = encodings
        self.keys = ['input_ids', 'attention_mask', 'dataset_ids']
        if train:
            self.keys += ['start_positions', 'end_positions']
        assert(all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key : torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings['input_ids'])

def get_dataset(args, datasets, data_dir, tokenizer, split_name, category='all'):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for idx, dataset in enumerate(datasets):
        dataset_name += f'_{dataset}'
        dataset_dict_curr = read_squad(f'{data_dir}/{dataset}', category)
        dataset_dict_curr['dataset_id'] = [idx] * len(dataset_dict_curr['question'])
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

    if category != 'all':
        for id, c in enumerate(CATEGORIES):
            if c['name'] == category:
                dataset_name += f'_cat{id}'

    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return QADataset(data_encodings, train=(split_name=='train')), dataset_dict


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
    tokenized_examples["dataset_ids"] = []
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

        tokenized_examples["dataset_ids"].append(dataset_dict['dataset_id'][sample_index])

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
    tokenized_examples['dataset_ids'] = []
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

        tokenized_examples["dataset_ids"].append(dataset_dict['dataset_id'][sample_index])

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):

    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        print(f'Loading encodings from {cache_path}')
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        print(f'Caching encodings to {cache_path}')
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


def read_squad(path, category='all'):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']

                # Filter by category
                if category != 'all':
                    if category not in [CATEGORIES[id]['name'] for id in getCategoryIds(question)]:
                        continue 

                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    
        # DEBUG: To keep load times short
        # if len(data_dict_collapsed['id']) >= 100:
        #     break

    return data_dict_collapsed

