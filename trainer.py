# Standard imports
from collections import OrderedDict
import os

# 3rd Party imports
from tensorboardX import SummaryWriter
from transformers import AdamW
import torch
import torch.nn as nn
from tqdm import tqdm

# Local imports
from model import Discriminator
import util

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions

        # Adversarial training
        self.adversarial = args.adversarial
        self.num_classes = len(args.train_datasets.split(','))
        if args.do_finetune:
            self.num_classes += len(args.finetune_datasets.split(','))
        self.dis_lambda = args.dis_lambda
        self.anneal = args.anneal
        self.concat = args.concat
        self.sep_id = 102
        if args.adversarial:
            if args.concat:
                input_size = 2 * args.hidden_size
            else:
                input_size = args.hidden_size
            self.discriminator = Discriminator(self.num_classes, input_size, args.hidden_size, args.num_layers, args.dropout)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()

        if self.adversarial:
            self.discriminator.eval()

        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)

                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, train_dataset_sizes, val_dict):
        
        device = self.device
        model.to(device)
        qa_optim = AdamW(model.parameters(), lr=self.lr)
        if self.adversarial:
            self.discriminator.to(device)
            dis_optim = AdamW(self.discriminator.parameters(), lr=self.lr)
            dis_lambda = self.dis_lambda
            dataset_weights = util.compute_imbalanced_class_weights(train_dataset_sizes, as_tensor=True)
            dataset_weights.to(device)
            assert dataset_weights.size(0) == self.num_classes
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    
                    loss_dict = {}

                    # Zero out gradients
                    qa_optim.zero_grad()
                    if self.adversarial:
                        dis_optim.zero_grad()

                    # Set models to train mode
                    model.train()
                    if self.adversarial:
                        self.discriminator.train()

                    # Unpack data
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    dataset_ids = batch['dataset_ids'].to(device)

                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, output_hidden_states=True)
                    
                    # ((batch_size, sequence_length, hidden_size))
                    # [torch.Size([16, 384, 768]), torch.Size([16, 384, 768]), torch.Size([16, 384, 768]), torch.Size([16, 384, 768]), torch.Size([16, 384, 768]), torch.Size([16, 384, 768]), torch.Size([16, 384, 768])]
                    # outputs.hidden_states

                    # Compute QA Loss
                    qa_loss = outputs.loss

                    loss_dict['qa_NLL'] = qa_loss.item()

                    if self.adversarial:

                        # (batch_size, hidden_size=768)
                        cls_embedding = outputs.hidden_states[-1][:, 0, :]

                        if self.concat:
                            batch_size = input_ids.size(0)
                            sep_idx = (input_ids == self.sep_id).sum(1)
                            sep_embedding = outputs.hidden_states[torch.arange(batch_size), sep_idx]
                            dis_input = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
                        else:
                            dis_input = cls_embedding
                        log_prob = self.discriminator(dis_input.detach())
                        
                        # KL Divergence Loss
                        targets = torch.ones_like(log_prob) * (1 / self.discriminator.num_classes)
                        kl_criterion = nn.KLDivLoss(reduction="batchmean")
                        if self.anneal:
                            dis_lambda *= util.kl_coef(global_idx)
                        adv_loss = dis_lambda * kl_criterion(log_prob, targets)
                        qa_loss += adv_loss

                        loss_dict['adv_loss'] = adv_loss.item()
                    loss_dict['qa_loss'] = qa_loss.item()

                    # Backprop
                    qa_loss.backward()
                    qa_optim.step()
                    qa_optim.zero_grad()

                    if self.adversarial:

                        with torch.no_grad():
                            # Forward pass
                            outputs = model(input_ids, attention_mask=attention_mask,
                                            start_positions=start_positions,
                                            end_positions=end_positions, output_hidden_states=True)

                            # (batch_size, hidden_size=768)
                            cls_embedding = outputs.hidden_states[-1][:, 0, :]

                            if self.concat:
                                batch_size = input_ids.size(0)
                                sep_idx = (input_ids == self.sep_id).sum(1)
                                sep_embedding = outputs.hidden_states[torch.arange(batch_size), sep_idx]
                                dis_input = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
                            else:
                                dis_input = cls_embedding
                        log_prob = self.discriminator(dis_input.detach())

                        # Compute discriminator loss
                        criterion = nn.NLLLoss(weight=dataset_weights).to(device)
                        dis_loss = criterion(log_prob, dataset_ids)
                        
                        loss_dict['D_loss'] = dis_loss.item()

                        # Backprop discriminator
                        dis_loss.backward()
                        dis_optim.step()
                        dis_optim.zero_grad()

                    # Progress bar update
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, **loss_dict)
                    
                    # TensorboardX update
                    for k, v in loss_dict.items():
                        tbx.add_scalar(f'train/{k}', v, global_idx)

                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.log.info(f'Achieved F1 score of {curr_score} at step {global_idx}. Saving model ...')
                            self.save(model)
                            
                    global_idx += 1
        return best_scores