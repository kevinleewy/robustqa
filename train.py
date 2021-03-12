# Standard imports
import csv
import json
import os
import re

# 3rd Party imports
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


# Local imports
from args import get_train_test_args
from categories import CATEGORIES
from dataset import get_dataset
from model import Ensemble
from trainer import Trainer
import util

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:

        if args.load_dir is None:
            model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        else:
            checkpoint_path = os.path.join(args.load_dir, 'checkpoint')
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)

        if args.log_file is None:
            args.log_file = 'log_train'

        log = util.get_logger(args.save_dir, args.log_file)
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")

        # Determine device (This line must come after `json.dumps(vars(args)`)
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        trainer = Trainer(args, log)
        train_dataset, _, train_dataset_sizes = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', args.category)
        
        log.info("Preparing Validation Data...")
        val_dataset, val_dict, _ = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val', args.category)
        
        if val_dict is None:
            log.info("No data to be found...")
        else:
            train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    sampler=RandomSampler(train_dataset))
            val_loader = DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(val_dataset))
            best_scores = trainer.train(model, train_loader, val_loader, train_dataset_sizes, val_dict)

    if args.do_eval:
        # Determine device
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        split_name = 'test' if 'test' in args.eval_dir else 'validation'

        if args.do_ensemble:

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            args.save_dir = util.get_save_dir(args.save_dir, args.run_name)

            if args.log_file is None:
                args.log_file = f'log_{split_name}'

            log = util.get_logger(args.save_dir, args.log_file)
            trainer = Trainer(args, log)

            # Read list of models to construct ensemble
            checkpoints = []
            with open(args.ensemble_cfg) as f:
                for line in f:
                    if line.startswith('#') or line.startswith('//'):
                        continue
                    checkpoints.append(line.strip())

            model = Ensemble(device=args.device)

            # Load each model and their respective weights
            for cp in checkpoints:
                log.info(f'Loading from {cp}')
                checkpoint_path = os.path.join(cp, 'checkpoint')
                val_log_path = os.path.join(cp, 'log_indomain_validation.txt')
                weights = []
                with open(val_log_path) as f:
                    for line in f:
                        entries = re.findall(r'Eval category=(.*) F1: ([0-9\.]+),', line)
                        if len(entries) == 1 and len(entries[0]) == 2:
                            category, weight = entries[0][0], entries[0][1]
                            if category != 'all':
                                weights.append(float(weight))

                assert len(weights) == len(CATEGORIES)

                model.add_pretrained_model(checkpoint_path, weights)

        else:

            if args.log_file is None:
                args.log_file = f'log_{split_name}'

            log = util.get_logger(args.save_dir, args.log_file)
            trainer = Trainer(args, log)

            if args.load_dir is None:
                args.load_dir = args.save_dir
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
            model.to(args.device)

        if args.category == 'all':

            if split_name == 'test':
                categories = ['all']
            else:
                categories = [c['name'] for c in CATEGORIES] + ['all']

        else:
            categories = [args.category]

        for category in categories:

            eval_dataset, eval_dict, _ = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name, category)
            
            if eval_dict is None:
                log.info("No data to be found... Skipping this category")
                continue
            
            eval_loader = DataLoader(eval_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(eval_dataset))
            eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                    eval_dict, return_preds=True,
                                                    split=split_name)
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
            log.info(f'Eval category={category} {results_str}')

            if category == 'all':
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
