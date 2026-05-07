import argparse
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test')
parser.add_argument('--epoch', default=150, type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('--batch-size', '--batch_size', default=1000, type=int,
                    dest='batch_size', metavar='N',
                    help='mini-batch size (default: 1000), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--encoder_type', type=str, default='HYBRID', help='Encoder type')
parser.add_argument('--seed', default=2023, type=int, help='seed for initializing training.')

parser.add_argument('--alpha', default=0.1, type=float, help='weight for the image-text contrastive loss (default: 0.1)')
parser.add_argument('--result_path', type=str, default='./results/')  
parser.add_argument('--eeg_data_path', type=str, default='./Things_EEG2/Preprocessed_data_250Hz/') 
parser.add_argument('--img_train_path', type=str, default='./EEG2image/features/clip-rn50_features_train.pt')
parser.add_argument('--img_test_path', type=str, default='./EEG2image/features/clip-rn50_features_test.pt')
parser.add_argument('--text_train_path', type=str, default='./EEG2image/features/Qwen_feature_maps_training_clip_cn.npy')
parser.add_argument('--text_test_path', type=str, default='./EEG2image/features/Qwen_feature_maps_test_clip_cn.npy')

# parameters for pretraining weight transfer strategies
parser.add_argument('--no_pretrain', default=False, action='store_true',
                    help='Do not load MAE pretrained weights (train from scratch)')
parser.add_argument('--load_pretrain_groups', type=str, default='ALL',
                    help='Parameter groups to load from pretrain')

parser.add_argument('--init_groups', type=str, default='ALL', #ENC,GAT,POS,CATTN,CNN,PROJ
                    help='Parameter groups to initialize')
parser.add_argument('--early_stopping', action='store_true', default=True,
                    help='Enable early stopping with 20 epochs patience on validation loss')
parser.add_argument('--exp_id', type=str, default=None,
                    help='Experiment identifier for output filename (default: auto-generate from config)')

import os
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from open_clip.loss import ClipLoss 
import datetime

from eeg_encoders import NICE, ATMS, MCRL, HYBRID, Config, ResidualAdd
from modules import weights_init_tensor, ParameterGroupManager

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
model_idx = 'test_eeg1'

ParameterGroupManager._COMPILED = ParameterGroupManager.compile_patterns()
    
class Proj_img(nn.Sequential):
    # def __init__(self, embedding_dim=768, proj_dim=1024, drop_proj=0.3):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        
        self.start_epoch = 0
        self.eeg_data_path = args.eeg_data_path
        self.train_feature_file_path = args.img_train_path
        self.test_feature_file_path = args.img_test_path
        self.early_stopping = args.early_stopping

        self.pretrain = False

        self.log_write = open(args.result_path + "log_subject%d.txt" % self.nSub, "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        # self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cls = ClipLoss().cuda()
        self.alpha = args.alpha

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.load_precomputed_features()

        self.eeg_model = globals()[args.encoder_type]().cuda()
        self.Proj_img = Proj_img().cuda()

        self.module_name = args.encoder_type
        
        self.load_pretrain_groups = ParameterGroupManager.parse_groups(args.load_pretrain_groups)
        self.init_groups = ParameterGroupManager.parse_groups(args.init_groups)
        ParameterGroupManager.set_init_safe(self.init_groups)

        # Load MAE pretrained weights if not disabled
        if not args.no_pretrain:
            # print("Attempting to load pretrained weights...")
            # print("Module name for checkpoint search:", self.module_name)

            ckpt_dir = os.path.join(args.result_path, 'mae_eeg_pretrain', 'checkpoints')
            pattern = os.path.join(ckpt_dir, f"mae_pretrain_{self.module_name}_sub{self.nSub:02d}_*.pth")
            # pattern = os.path.join(ckpt_dir, f"mae_pretrain_*_mr0.3_embed256_depth2_sub{self.nSub:02d}_*.pth")
            candidates = glob.glob(pattern)
            candidates = [p for p in candidates if os.path.isfile(p)]

            if candidates:
                ckpt_path = max(candidates, key=os.path.getmtime)
                print(f"Found pretrain checkpoint for subject {self.nSub}: {ckpt_path}. Attempting to load into eeg_model.")

                loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                state_dict = loaded.get('model_state', loaded) if isinstance(loaded, dict) else loaded

                if isinstance(state_dict, dict):
                    filtered_keys = ParameterGroupManager.build_load_dict(
                        self.eeg_model, state_dict, self.load_pretrain_groups
                    )
                    
                    print(f"  parameters: {len(state_dict)}")
                    print(f"  filtered parameters: {len(filtered_keys)}")
                    print(f"  load groups: {self.load_pretrain_groups}")
                    
                    try:
                        missing, unexpected = self.eeg_model.load_state_dict(filtered_keys, strict=False)
                        print(f"    Loaded: {len(filtered_keys)} parameters")
                        # print(f"    Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    except Exception as e:
                        print(f"   Warning: failed to load encoder parameters: {e}")
                else:
                    print("Checkpoint state_dict is not a dict; skipping loading.")
            else:
                print(f"No MAE checkpoint found for subject {self.nSub} at {ckpt_dir}")
                print("Training from scratch...")

        else:
            print("--no_pretrain flag set. Training from scratch without loading MAE weights.")
            self.eeg_model = nn.DataParallel(self.eeg_model, device_ids=[i for i in range(len(gpus))])
            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.centers = {}
        
        print('initial define done.')


    def load_precomputed_features(self):

        train_data = torch.load(self.train_feature_file_path)
        self.train_img_features = train_data["img_features"]    # shape: (16540, 1024)
        test_data = torch.load(self.test_feature_file_path)
        self.test_img_features = test_data["img_features"]    # shape: (200, 1024)
        

    def evaluate_model(self, eeg_features, img_features_all, labels, k):
        """
        Evaluate the model for k-way retrieval using batch image features.
        """
        total = 0
        correct = 0
        all_labels = set(range(img_features_all.size(0)))  # All possible classes in batch

        for idx, label in enumerate(labels):
            # Select k-1 random classes excluding the correct class
            possible_classes = list(all_labels - {label.item()})
            selected_classes = random.sample(possible_classes, k - 1) + [label.item()]

            selected_img_features = img_features_all[selected_classes]
            similarity = (100.0 * eeg_features[idx] @ selected_img_features.T).softmax(dim=-1)
            predicted_label = selected_classes[torch.argmax(similarity).item()]
            if predicted_label == label.item():
                correct += 1
            total += 1

        accuracy = correct / total
        return accuracy


    def evaluate_topk_epoch(self, text_feats_all):
        """Run a lightweight test pass to report retrieval/classification top1 and top5."""
        self.eeg_model.eval()
        self.Proj_img.eval()
        total = 0
        top1 = 0
        top5 = 0
        top1_cls = 0
        top5_cls = 0

        with torch.no_grad():
            for teeg, timg, ttext, tlabel in self.test_dataloader:

                teeg = teeg.cuda().type(self.Tensor)
                timg_features = timg.cuda().type(self.Tensor)
                tlabel = tlabel.type(self.LongTensor).cuda()

                subject_ids = torch.full((teeg.size(0),), self.nSub - 1, dtype=torch.long).cuda()
                teeg = teeg.squeeze(1)
                tfea = self.eeg_model(teeg, subject_ids)
                tfea = tfea / (tfea.norm(dim=1, keepdim=True) + 1e-12)
                
                # Apply Proj_img to image features
                timg_features = self.Proj_img(timg_features)
                timg_features = timg_features / (timg_features.norm(dim=1, keepdim=True) + 1e-12)
                
                sim = (100.0 * tfea @ timg_features.t()).softmax(dim=-1)
                _, indices = sim.topk(5)
                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top5 += (tt_label == indices[:, :5]).sum().item()

                sim_class = (100.0 * tfea @ text_feats_all.t()).softmax(dim=-1)
                _, indices_class = sim_class.topk(5)
                top1_cls += (tt_label == indices_class[:, :1]).sum().item()
                top5_cls += (tt_label == indices_class[:, :5]).sum().item()

        if total == 0:
            return 0.0, 0.0, 0.0, 0.0
        return top1 / total, top5 / total, top1_cls / total, top5_cls / total
    

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label
    

    def get_image_data(self):
        train_img_feature = self.train_img_features
        test_img_feature = self.test_img_features
        return train_img_feature, test_img_feature

    
    def get_text_data(self):
        train_txt_feature = np.load("./EEG2image/features/Qwen_feature_maps_training_clip_cn.npy", allow_pickle=True)
        test_txt_feature = np.load("./EEG2image/features/Qwen_feature_maps_test_clip_cn.npy", allow_pickle=True)
        train_txt_feature = np.squeeze(train_txt_feature)
        test_txt_feature = np.squeeze(test_txt_feature)
        return train_txt_feature, test_txt_feature


    def train(self):

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, test_img_feature = self.get_image_data()
        train_txt_feature, test_txt_feature = self.get_text_data()

        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]
        train_txt_feature = train_txt_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = train_img_feature[:740]
        val_image = torch.from_numpy(val_image) if isinstance(val_image, np.ndarray) else val_image
        val_text = torch.from_numpy(train_txt_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = train_img_feature[740:]
        train_image = torch.from_numpy(train_image) if isinstance(train_image, np.ndarray) else train_image
        train_text = torch.from_numpy(train_txt_feature[740:])

        test_eeg = torch.from_numpy(test_eeg)
        test_label = torch.from_numpy(test_label)
        test_txt_feature = torch.from_numpy(test_txt_feature)
        test_img_feature = torch.from_numpy(test_img_feature) if isinstance(test_img_feature, np.ndarray) else test_img_feature

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image, train_text)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image, val_text) #torch.Size([740, 1, 63, 250]) torch.Size([16540, 768]) torch.Size([740, 768])
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_img_feature, test_txt_feature, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        ParameterGroupManager.selective_init(self.eeg_model, self.init_groups)

        # Optimizers (using AdamW with weight decay)
        self.optimizer_eeg = torch.optim.AdamW(itertools.chain(self.eeg_model.parameters()), lr=self.lr, betas=(self.b1, self.b2), weight_decay=0.01)
        self.optimizer_img = torch.optim.AdamW(itertools.chain(self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2), weight_decay=0.01)
        
        # Track top-3 best models based on validation loss
        top3_models = []  # List of tuples: (val_loss, epoch, state_dict, train_loss)
        
        # Early stopping based on validation loss with patience
        patience = 10  
        patience_counter = 0

        for e in range(self.n_epochs):
            epoch_num = e + 1  # use 1-based epoch for logging
            in_epoch = time.time()
            self.eeg_model.train()
            self.Proj_img.train()
            
            # Track training loss
            train_loss_sum = 0.0
            train_batches = 0

            for i, (eeg, img, txt) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                txt_features = Variable(txt.cuda().type(self.Tensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                img_features = self.Proj_img(img_features)
                # Normalize features
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)

                batch_size = eeg.size(0)
                subject_ids = torch.full((batch_size,), self.nSub - 1, dtype=torch.long).cuda()
                eeg = eeg.squeeze(1)
                eeg_features = self.eeg_model(eeg, subject_ids)
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)

                # contrative loss - EEG and image
                logit_scale = self.logit_scale.exp()

                loss_cos_ie = self.criterion_cls(eeg_features, img_features, logit_scale)
                loss_cos_it = self.criterion_cls(img_features, txt_features, logit_scale)
                
                alpha = self.alpha
                loss = (1 - alpha) * loss_cos_ie + alpha * loss_cos_it
                # Track training loss
                train_loss_sum += loss.item()
                train_batches += 1
                
                self.optimizer_eeg.zero_grad()
                self.optimizer_img.zero_grad()
                loss.backward()
                self.optimizer_eeg.step()
                self.optimizer_img.step()

            # Compute average training loss
            train_loss_avg = train_loss_sum / train_batches if train_batches > 0 else 0.0

            if epoch_num % 1 == 0:
                self.eeg_model.eval()
                with torch.no_grad():
                    # Validation - accumulate loss across all batches
                    vloss_sum = 0.0
                    val_batches = 0
                    for i, (veeg, vimg, vtxt) in enumerate(self.val_dataloader):
                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        subject_id = self.nSub - 1
                        subject_ids = torch.full((veeg.size(0),), subject_id, dtype=torch.long).cuda()
                        veeg = veeg.squeeze(1)
                        veeg_features = self.eeg_model(veeg, subject_ids)
                        vimg_features = self.Proj_img(vimg_features)
                        vtxt_features = vtxt.cuda().type(self.Tensor)
                        vtxt_features = vtxt_features / vtxt_features.norm(dim=1, keepdim=True)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()

                        vloss1 = self.criterion_cls(veeg_features, vimg_features, logit_scale)
                        vloss2 = self.criterion_cls(vimg_features, vtxt_features, logit_scale)
                        
                        alpha = self.alpha
                        vloss = (1 - alpha) * vloss1 + alpha * vloss2
                        
                        vloss_sum += vloss.item()
                        val_batches += 1
                    
                    # Compute average validation loss across all batches
                    vloss_val = vloss_sum / val_batches if val_batches > 0 else 0.0
                    
                    # Save state_dict only once per epoch (outside the batch loop)
                    eeg_sd = getattr(self.eeg_model, 'module', self.eeg_model).state_dict()
                    eeg_sd_cpu = {k: v.cpu().clone() for k, v in eeg_sd.items()}
                    proj_img_sd = getattr(self.Proj_img, 'module', self.Proj_img).state_dict()
                    proj_img_sd_cpu = {k: v.cpu().clone() for k, v in proj_img_sd.items()}
                        
                print('Epoch:', e, '  Validation Loss: %.4f' % (vloss_val,))


                # Update top-3 models list
                # Track if this epoch improved the top-3 models (for early stopping)
                improved = False
                
                # Find position to insert this model
                inserted = False
                for idx, (prev_val_loss, _, _, _, _) in enumerate(top3_models):
                    if vloss_val < prev_val_loss:
                        top3_models.insert(idx, (vloss_val, epoch_num, eeg_sd_cpu, proj_img_sd_cpu, train_loss_avg))
                        if len(top3_models) > 3:
                            top3_models.pop()  # Keep only top 3
                        inserted = True
                        improved = True  # Mark as improved since we inserted into top3
                        break
                
                # If not inserted but top3 not full, it's still an improvement (added to top3)
                if not inserted and len(top3_models) < 3:
                    top3_models.append((vloss_val, epoch_num, eeg_sd_cpu, proj_img_sd_cpu, train_loss_avg))
                    improved = True  # Mark as improved since we added to top3

                # Update early stopping counter
                # Reset counter if validation loss improved (entered top-3), otherwise increment
                if improved:
                    if self.early_stopping:
                        patience_counter = 0  # Reset when new model enters top-3
                else:
                    if self.early_stopping:
                        patience_counter += 1
                        print(f'  [Patience: {patience_counter}/{patience}]')
                        if patience_counter >= patience:
                            print(f'\n*** Early stopping triggered at epoch {epoch_num} after {patience} epochs without improvement ***')
                            if top3_models:
                                print(f'Best val loss: {top3_models[0][0]:.4f} at epoch {top3_models[0][1]}')
                            break

        # Load best models for final evaluation
        if top3_models:
            os.makedirs('./model/', exist_ok=True)
            best_eeg_sds = []
            best_proj_img_sds = []
            for idx, (val_loss, epoch_num, eeg_sd, proj_img_sd, _) in enumerate(top3_models):
                torch.save(eeg_sd, f'./model/{model_idx}_eeg_model_top{idx+1}_epoch{epoch_num}.pth')
                torch.save(proj_img_sd, f'./model/{model_idx}_proj_img_model_top{idx+1}_epoch{epoch_num}.pth')
                best_eeg_sds.append(eeg_sd)
                best_proj_img_sds.append(proj_img_sd)
        else:
            print("\nNo best models saved (training ended before collecting top-3 models)")
            best_eeg_sds = []
            best_proj_img_sds = []

        # Final test evaluation with ensemble of top-3 models
        all_results = {k: [] for k in [2, 4, 10, 20, 50, 100]}
        all_topacc = {k: [] for k in range(10)}
        all_results_class = {k: [] for k in [2, 4, 10, 20, 50, 100]}
        all_topacc_class = {k: [] for k in range(10)}

        for model_idx_ensemble, (best_eeg_sd, best_proj_img_sd) in enumerate(zip(best_eeg_sds, best_proj_img_sds)):
            self.eeg_model.load_state_dict(best_eeg_sd, strict=False)
            self.Proj_img.load_state_dict(best_proj_img_sd, strict=False)
            self.eeg_model.eval()
            self.Proj_img.eval()

            # compute full results
            resultsacc = {i: 0 for i in range(10)}            # retrieval top1-10
            resultsacc_class = {i: 0 for i in range(10)}       # classification top1-10
            total = 0
            results = {}          # retrieval k-way
            results_class = {}    # classification k-way
            with torch.no_grad():
                for teeg, timg, ttext, tlabel in self.test_dataloader:
                    teeg = teeg.cuda().type(self.Tensor)
                    timg_features = timg.cuda().type(self.Tensor)
                    ttext_features = ttext.cuda().type(self.Tensor)
                    tlabel = tlabel.type(self.LongTensor).cuda()

                    subject_ids = torch.full((teeg.size(0),), self.nSub - 1, dtype=torch.long).cuda()
                    teeg = teeg.squeeze(1)
                    tfea = self.eeg_model(teeg, subject_ids)
                    tfea = tfea / (tfea.norm(dim=1, keepdim=True) + 1e-12)

                    # Apply Proj_img to image features during testing
                    timg_features = self.Proj_img(timg_features)
                    timg_features = timg_features / (timg_features.norm(dim=1, keepdim=True) + 1e-12)
                    ttext_features = ttext_features / (ttext_features.norm(dim=1, keepdim=True) + 1e-12)

                    for k in [2, 4, 10, 20, 50, 100]:
                        # evaluate_model uses batch image features (timg_features) for k-way, like bestmodel1112.py
                        results[k] = results.get(k, 0.0) + self.evaluate_model(tfea, timg_features, tlabel, k=k) * tlabel.size(0)
                        results_class[k] = results_class.get(k, 0.0) + self.evaluate_model(tfea, ttext_features, tlabel, k=k) * tlabel.size(0)
                
                    sim = (100.0 * tfea @ timg_features.t()).softmax(dim=-1)
                    _, indices = sim.topk(10)
                    tt_label = tlabel.view(-1, 1)
                    total += tlabel.size(0)
                    for i in range(10):
                        resultsacc[i] += (tt_label == indices[:, :i+1]).sum().item()

                    # classification top1-10 using text features (use self.test_text_features instead of ttext_features from dataloader)
                    sim_class = (100.0 * tfea @ ttext_features.t()).softmax(dim=-1)
                    _, indices_class = sim_class.topk(10)
                    for i in range(10):
                        resultsacc_class[i] += (tlabel.view(-1, 1) == indices_class[:, :i+1]).sum().item()

            # finalize results aggregation for this model
            topacc = {i: float(resultsacc[i]) / float(total) for i in range(10)}
            topacc_class = {i: float(resultsacc_class[i]) / float(total) for i in range(10)}
            # average k-way results
            for k in [2, 4, 10, 20, 50, 100]:
                results[k] = results.get(k, 0.0) / float(total) if total > 0 else 0.0
                results_class[k] = results_class.get(k, 0.0) / float(total) if total > 0 else 0.0

            # Store results for ensemble averaging
            for k in range(10):
                all_topacc[k].append(topacc[k])
                all_topacc_class[k].append(topacc_class[k])
            for k in [2, 4, 10, 20, 50, 100]:
                all_results[k].append(results[k])
                all_results_class[k].append(results_class[k])

        # Compute ensemble average results
        print('\n' + '='*80)
        print('ENSEMBLE AVERAGE RESULTS (Top-3 Models)')
        print('='*80)
        
        avg_topacc = {k: np.mean(all_topacc[k]) if all_topacc[k] else 0.0 for k in range(10)}
        avg_topacc_class = {k: np.mean(all_topacc_class[k]) if all_topacc_class[k] else 0.0 for k in range(10)}
        avg_results = {k: np.mean(all_results[k]) if all_results[k] else 0.0 for k in [2, 4, 10, 20, 50, 100]}
        avg_results_class = {k: np.mean(all_results_class[k]) if all_results_class[k] else 0.0 for k in [2, 4, 10, 20, 50, 100]}
        
        # print and save ensemble average results
        print('top1-%.6f, top2-%.6f, top3-%.6f, top4-%.6f, top5-%.6f, top6-%.6f, top7-%.6f, top8-%.6f, top9-%.6f, top10-%.6f' %
              (avg_topacc[0], avg_topacc[1], avg_topacc[2], avg_topacc[3], avg_topacc[4], avg_topacc[5], avg_topacc[6], avg_topacc[7], avg_topacc[8], avg_topacc[9]))
        self.log_write.write('ENSEMBLE AVERAGE - top1-%.6f, top2-%.6f, top3-%.6f, top4-%.6f, top5-%.6f, top6-%.6f, top7-%.6f, top8-%.6f, top9-%.6f, top10-%.6f\n' %
              (avg_topacc[0], avg_topacc[1], avg_topacc[2], avg_topacc[3], avg_topacc[4], avg_topacc[5], avg_topacc[6], avg_topacc[7], avg_topacc[8], avg_topacc[9]))
        print('way-2-%.6f, way-4-%.6f, way-10-%.6f, way-20-%.6f, way-50-%.6f, way-100-%.6f' %
              (avg_results[2], avg_results[4], avg_results[10], avg_results[20], avg_results[50], avg_results[100]))
        self.log_write.write('ENSEMBLE AVERAGE - way-2-%.6f, way-4-%.6f, way-10-%.6f, way-20-%.6f, way-50-%.6f, way-100-%.6f\n' %
              (avg_results[2], avg_results[4], avg_results[10], avg_results[20], avg_results[50], avg_results[100]))

        # print and save classification accuracy
        print('class-top1-%.6f, class-top2-%.6f, class-top3-%.6f, class-top4-%.6f, class-top5-%.6f, class-top6-%.6f, class-top7-%.6f, class-top8-%.6f, class-top9-%.6f, class-top10-%.6f' %
            (avg_topacc_class[0], avg_topacc_class[1], avg_topacc_class[2], avg_topacc_class[3], avg_topacc_class[4], avg_topacc_class[5], avg_topacc_class[6], avg_topacc_class[7], avg_topacc_class[8], avg_topacc_class[9]))
        self.log_write.write('ENSEMBLE AVERAGE - class-top1-%.6f, class-top2-%.6f, class-top3-%.6f, class-top4-%.6f, class-top5-%.6f, class-top6-%.6f, class-top7-%.6f, class-top8-%.6f, class-top9-%.6f, class-top10-%.6f\n' %
            (avg_topacc_class[0], avg_topacc_class[1], avg_topacc_class[2], avg_topacc_class[3], avg_topacc_class[4], avg_topacc_class[5], avg_topacc_class[6], avg_topacc_class[7], avg_topacc_class[8], avg_topacc_class[9]))
        print('class-way-2-%.6f, class-way-4-%.6f, class-way-10-%.6f, class-way-20-%.6f, class-way-50-%.6f, class-way-100-%.6f' %
            (avg_results_class[2], avg_results_class[4], avg_results_class[10], avg_results_class[20], avg_results_class[50], avg_results_class[100]))
        self.log_write.write('ENSEMBLE AVERAGE - class-way-2-%.6f, class-way-4-%.6f, class-way-10-%.6f, class-way-20-%.6f, class-way-50-%.6f, class-way-100-%.6f\n' %
            (avg_results_class[2], avg_results_class[4], avg_results_class[10], avg_results_class[20], avg_results_class[50], avg_results_class[100]))
        
        return avg_results, avg_topacc, avg_results_class, avg_topacc_class
        

def main():
    args = parser.parse_args()
    num_sub = args.num_sub   
    cal_num = 0
    
    # produce short exp_id based on config if not provided
    if args.exp_id is None:
        import hashlib
        config_str = f"{args.encoder_type}_{args.load_pretrain_groups}_{args.init_groups}_{args.no_pretrain}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6].upper()
        args.exp_id = config_hash
        print(f"Auto-generated exp_id: {args.exp_id}")

    top_k_accs = {k: [] for k in range(10)}  # retrieval top1-top10
    way_k_accs = {k: [] for k in [2, 4, 10, 20, 50, 100]}  # retrieval k-way
    top_k_accs_class = {k: [] for k in range(10)}  # classification top1-top10
    way_k_accs_class = {k: [] for k in [2, 4, 10, 20, 50, 100]}  # classification k-way
   
    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = args.seed

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        results, topacc, results_class, topacc_class = ie.train() 
        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        # --- top-k accuracy ---
        for k in range(10):
            val = topacc.get(k, 0.0) 
            top_k_accs[k].append(val)
            val_cls = topacc_class.get(k, 0.0)
            top_k_accs_class[k].append(val_cls)

        # --- k-way accuracy ---
        for k in [2, 4, 10, 20, 50, 100]:
            val = results.get(k, 0.0)
            way_k_accs[k].append(val)
            val_cls = results_class.get(k, 0.0)
            way_k_accs_class[k].append(val_cls)


    # Create retrieval and classification results matrices (save separately)
    if cal_num > 0:
        # Retrieval results
        retrieval_data = []
        for sub_idx in range(cal_num):
            row = []
            for k in range(10):
                row.append(top_k_accs[k][sub_idx])
            for k in [2, 4, 10, 20, 50, 100]:
                row.append(way_k_accs[k][sub_idx])
            retrieval_data.append(row)

        retrieval_avg_row = [np.mean(top_k_accs[k]) for k in range(10)] + \
                            [np.mean(way_k_accs[k]) for k in [2, 4, 10, 20, 50, 100]]
        retrieval_data.append(retrieval_avg_row)

        retrieval_array = np.array(retrieval_data)
        index = [f'subject_{i+1}' for i in range(cal_num)] + ['average']
        retrieval_columns = [f'top{k+1}' for k in range(10)] + [f'{k}-way' for k in [2, 4, 10, 20, 50, 100]]

        pd_retrieval = pd.DataFrame(data=np.round(retrieval_array, 4), index=index, columns=retrieval_columns)

        # Classification results
        classification_data = []
        for sub_idx in range(cal_num):
            row = []
            for k in range(10):
                row.append(top_k_accs_class[k][sub_idx])
            for k in [2, 4, 10, 20, 50, 100]:
                row.append(way_k_accs_class[k][sub_idx])
            classification_data.append(row)

        classification_avg_row = [np.mean(top_k_accs_class[k]) for k in range(10)] + \
                                 [np.mean(way_k_accs_class[k]) for k in [2, 4, 10, 20, 50, 100]]
        classification_data.append(classification_avg_row)

        classification_array = np.array(classification_data)
        classification_columns = [f'top{k+1}' for k in range(10)] + [f'{k}-way' for k in [2, 4, 10, 20, 50, 100]]
        pd_classification = pd.DataFrame(data=np.round(classification_array, 4), index=index, columns=classification_columns)

        # Save to separate CSV files with short exp_id
        current_date = datetime.datetime.now().strftime("%m%d")
        
        # Use different prefix based on no_pretrain flag
        prefix = "retrieval" if args.no_pretrain else "mae_retrieval"
        prefix_cls = "classification" if args.no_pretrain else "mae_classification"
        
        # Use short exp_id in filename
        retrieval_filename = os.path.join(args.result_path, f"{prefix}_{args.encoder_type}_{args.exp_id}_{current_date}_1.csv")
        classification_filename = os.path.join(args.result_path, f"{prefix_cls}_{args.encoder_type}_{args.exp_id}_{current_date}_1.csv")
        
        pd_retrieval.to_csv(retrieval_filename, float_format='%.4f')
        pd_classification.to_csv(classification_filename, float_format='%.4f')
        
        print(f"Retrieval results saved to: {retrieval_filename}")
        print(f"Classification results saved to: {classification_filename}")
    else:
        print("No results to save - no subjects processed")


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))