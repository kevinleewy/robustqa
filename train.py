# Standard imports
import csv
import json
import os

# 3rd Party imports
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


# Local imports
from args import get_train_test_args
from dataset import get_dataset
from model import RobustQA
from trainer import Trainer
import util

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:

        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")

        # Determine device (This line must come after `json.dumps(vars(args)`)
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        # Determine device
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
