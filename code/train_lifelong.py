from copy import deepcopy
import os
import logging
from transformers import ConvNextFeatureExtractor, ConvNextModel
import torch
import argparse
from DataHelper import get_dataset, get_online_dataset
from tqdm.auto import tqdm
import torch
from torch import nn
import wandb
import json
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from matplotlib import pyplot as plt

def reservoir(num_seen_examples: int, buffer_size: int) -> int:

    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size: #total batch size
        return rand
    else:
        return -1

class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.attributes = ['images','feats','tgts']
    
    def init_tensors(self, images=torch.randn(1, 3, 224, 224), feats=torch.randn(1, 2), tgts=torch.randn(1, 2)):

        for attr_str in self.attributes:

            attr = eval(attr_str)

            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32

                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ))

    
    def get_data(self, size: int):
        if size > min(self.num_seen_examples, self.images.shape[0]):
            size = min(self.num_seen_examples, self.images.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.images.shape[0]),
                                  size=size, replace=False)

        ret_tuple = (torch.stack([ee.cpu()
                            for ee in self.images[choice]]),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def add_data(self, images, feats, tgts):
        
        if not hasattr(self, 'images'):
            self.init_tensors()

        # it is random and not consider task
        for i in range(images.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                if images is not None:
                    self.images[index] = images[i]
                if feats is not None:
                    self.feats[index] = feats[i]
                if tgts is not None:
                    self.tgts[index] = tgts[i]


    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
        self.pre_layer = nn.Linear(768 + 2, 64)
        self.gelu = nn.GELU()
        self.final_layer = nn.Linear(64, 1)
    
    def forward(self, x, feat):
        feature = self.feature_extractor(x)['pooler_output']
        out = self.pre_layer(torch.cat([feature, feat], dim=1))
        return self.final_layer(self.gelu(out))
    
    def intermediate_forward(self, x, feat):
        feature = self.feature_extractor(x)['pooler_output']
        out = self.pre_layer(torch.cat([feature, feat], dim=1))
        return out
    

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--final_lr', default=5e-5, type=float)
parser.add_argument('--seed', default=2023, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save_pth', default=None, type=str)
parser.add_argument('--base_dir', default='./', type=str)
parser.add_argument('--training', action='store_true')
parser.add_argument('--lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', type=float, default=0.15)
parser.add_argument('--lambda3', type=float, default=0.6)
parser.add_argument('--lambda4', type=float, default=0.05)
parser.add_argument('--lambda5', type=float, default=0.1)
parser.add_argument('--extra_info', type=str, default='')
parser.add_argument('--split_num', type=int, default=4)
# ablation
parser.add_argument('--distill', action='store_true')
parser.add_argument('--buffer_size', type=int, default=None)


args = parser.parse_args()

setup_seed(args.seed)

if args.wandb:
    wandb.init(
        project='甲状腺oneline_0722',
        group=f'epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}',
        name=f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}',
    )

args.data_path = os.path.join(args.base_dir, 'data', 'data_all.xlsx')
args.image_path = os.path.join(args.base_dir, 'images')

if args.save_pth is None:
    args.save_pth = os.path.join(args.base_dir, 'model_online', f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}')

logging.basicConfig(
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename=os.path.join(args.base_dir, './logs', f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}'),
    level=logging.INFO,
    filemode='w'
)

def train(dataloader, testloader, model, epoch, optimizer, scheduler=None, model_ori=None, buffer=None):
    for epo in tqdm(range(epoch)):
        model.train()
        tot_loss = 0.0
        LOSS_EACH = [0.0] * 5
        tot_zero_loss = 0.0
        tot_distill_loss = 0.0
        tot_buffer_loss = 0.0
        for image, feat, tgt in dataloader:

            image = image[0].cuda()
            feat = feat.cuda()

            outputs = model(image, feat).squeeze()
            tgt = tgt.cuda()
            LABEL_MASK = [(tgt[:, -1] == i).cuda() for i in [-2, -1, 0, 1, 2]]

            LOSS = [
                args.lambda1 * torch.sum(torch.clamp(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[0]), min=0) ** 2),
                args.lambda2 * torch.sum(torch.clamp(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[1]), min=0) ** 2),
                args.lambda3 * torch.sum(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[2]) ** 2),
                args.lambda4 * torch.sum(torch.clamp(((tgt[:, 0] - outputs.squeeze()) * LABEL_MASK[3]), min=0) ** 2),
                args.lambda5 * torch.sum(torch.clamp(((tgt[:, 0] - outputs.squeeze()) * LABEL_MASK[4]), min=0) ** 2),
            ]
            num = [
                LABEL_MASK[i].sum() for i in range(5)
            ]
            LOSS = [LOSS[i] / num[i] if num[i] else LOSS[i] for i in range(5)]
            
            loss = sum(LOSS)
            
            zero_loss = F.relu(-outputs).sum()
            tot_zero_loss += zero_loss.item()

            loss += zero_loss

            if model_ori is not None:
                model_ori.eval()

                intermediate_stu = model(image, feat)
                with torch.no_grad():
                    intermediate_tea = model_ori(image, feat)
                distill_loss = nn.MSELoss()(intermediate_stu, intermediate_tea)
                loss += distill_loss

                tot_distill_loss += distill_loss.item()
            
            if buffer is not None and hasattr(buffer, 'images'):
                buffer_image, buffer_feats, buffer_tgts = buffer.get_data(args.batch_size)
                buffer_outputs = model(buffer_image.cuda(), buffer_feats.cuda()).squeeze()
                buffer_tgts = buffer_tgts.cuda()
                buffer_LABEL_MASK = [(buffer_tgts[:, -1] == i).cuda() for i in [-2, -1, 0, 1, 2]]

                buffer_LOSS = [
                    args.lambda1 * torch.sum(torch.clamp(((buffer_outputs.squeeze() - buffer_tgts[:, 0]) * buffer_LABEL_MASK[0]), min=0) ** 2),
                    args.lambda2 * torch.sum(torch.clamp(((buffer_outputs.squeeze() - buffer_tgts[:, 0]) * buffer_LABEL_MASK[1]), min=0) ** 2),
                    args.lambda3 * torch.sum(((buffer_outputs.squeeze() - buffer_tgts[:, 0]) * buffer_LABEL_MASK[2]) ** 2),
                    args.lambda4 * torch.sum(torch.clamp(((buffer_tgts[:, 0] - buffer_outputs.squeeze()) * buffer_LABEL_MASK[3]), min=0) ** 2),
                    args.lambda5 * torch.sum(torch.clamp(((buffer_tgts[:, 0] - buffer_outputs.squeeze()) * buffer_LABEL_MASK[4]), min=0) ** 2),
                ]
                buffer_num = [
                    buffer_LABEL_MASK[i].sum() for i in range(5)
                ]
                buffer_LOSS = [buffer_LOSS[i] / buffer_num[i] if buffer_num[i] else buffer_LOSS[i] for i in range(5)]
                
                buffer_loss = sum(buffer_LOSS)
                loss += buffer_loss

                tot_buffer_loss += buffer_loss.item()

            LOSS_EACH = [LOSS_EACH[i] + LOSS[i].item() for i in range(5)]
            tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()
        
        logging.info(json.dumps({f'train_loss tot': tot_loss}))
        logging.info(json.dumps({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)}))

        print(json.dumps({f'train_loss tot': tot_loss}))
        print(json.dumps({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)}))
        
        logging.info(json.dumps({f'zero_loss tot': tot_zero_loss}))
        logging.info(json.dumps({f'distill tot': tot_distill_loss}))

        if args.wandb:
            wandb.log({f'train_loss tot': tot_loss})
            wandb.log({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)})
            wandb.log({f'zero_loss tot': tot_zero_loss})
            wandb.log({f'distill tot': tot_distill_loss})
            wandb.log({f'buffer tot': tot_buffer_loss})

        if scheduler is not None:
            scheduler.step()
        
        evaluate(testloader, model, do_print=True)

def evaluate(dataloader, model, do_print=False):
    model.eval()
    tot_loss = 0.0
    tot_zero_loss = 0.0
    LOSS_EACH = [0.0] * 5
    P, T, L = [], [], []
    with torch.no_grad():
        for image, feat, label in dataloader:
            image = image[0].cuda()
            feat = feat.cuda()
            outputs = model(image, feat).squeeze()

            tgt = label.cuda()
            LABEL_MASK = [(tgt[:, -1] == i).cuda() for i in [-2, -1, 0, 1, 2]]
            LOSS = [
                args.lambda1 * torch.sum(torch.clamp(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[0]), min=0) ** 2),
                args.lambda2 * torch.sum(torch.clamp(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[1]), min=0) ** 2),
                args.lambda3 * torch.sum(((outputs.squeeze() - tgt[:, 0]) * LABEL_MASK[2]) ** 2),
                args.lambda4 * torch.sum(torch.clamp(((tgt[:, 0] - outputs.squeeze()) * LABEL_MASK[3]), min=0) ** 2),
                args.lambda5 * torch.sum(torch.clamp(((tgt[:, 0] - outputs.squeeze()) * LABEL_MASK[4]), min=0) ** 2),
            ]
            num = [
                LABEL_MASK[i].sum() for i in range(5)
            ]
            LOSS = [LOSS[i] / num[i] if num[i] else LOSS[i] for i in range(5)]
            
            loss = sum(LOSS)

            zero_loss = F.relu(-outputs).sum()
            tot_zero_loss += zero_loss.item()

            loss += zero_loss
            
            LOSS_EACH = [LOSS_EACH[i] + LOSS[i].item() for i in range(5)]
            tot_loss += loss.item()
            P += outputs.squeeze().cpu().numpy().tolist()
            T += tgt[:, -2].cpu().numpy().tolist()
            L += tgt[:, -1].cpu().numpy().tolist()
    
    r, x, y, weighted_precision = test(P, T, L)
    if do_print:
        logging.info("r: {}, x: {}, y: {}, wp: {}".format(r,x,y, weighted_precision))
        print("r: {}, x: {}, y: {}, wp: {}".format(r,x,y, weighted_precision))
    if args.wandb:
        wandb.log({f'correlation for 0 class': r, 'acc for postive class': x, 'acc for negative class': y, 'weighted precision': weighted_precision})
        wandb.log({f'eval_loss tot': tot_loss})
        wandb.log({f'eval_loss {i-2}': LOSS_EACH[i] for i in range(5)})
        wandb.log({f'eval_zero_loss tot': tot_zero_loss})

def test(P,T,L):
    zero, positive, negative = [], [], []
    zero_t, positive_t, negative_t = [], [], []
    for i, item in enumerate(L):
        if item == 0.0:
            zero.append(P[i])
            zero_t.append(T[i])
        elif item > 0:
            positive.append(P[i])
            positive_t.append(T[i])
        else:
            negative.append(P[i])
            negative_t.append(T[i])

    r = np.corrcoef(zero, zero_t)[0][1]
    x = sum([positive_t[i] < positive[i] for i in range(len(positive))]) / len(positive)
    y = sum([negative[i] < negative_t[i] for i in range(len(negative_t))]) / len(negative_t)
    weighted_precision = (x * len(positive) + y * len(negative_t)) / (len(positive) + len(negative_t))
    return r, x, y, weighted_precision

if __name__ == '__main__':
    train_dataset, test_dataset = get_online_dataset(args)
    train_loaders = []
    for j in range(args.split_num):
        train_loaders.append(DataLoader(train_dataset[j], batch_size=args.batch_size, shuffle=True, num_workers=16))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = Net()

    model = model.cuda()
    if args.wandb:
        wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model_ori = None
    buffer = None

    if args.buffer_size is not None:
        buffer = Buffer(args.buffer_size)
    
    if args.training:
        for j in range(args.split_num):
            if j > 0 and args.distill:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.final_lr * (0.95 ** (j-1)))
                model_ori = deepcopy(model)
                for params in model_ori.parameters():
                    params.requires_grad = False
                scheduler = None
            train(train_loaders[j], test_loader, model, epochs, optimizer, scheduler, model_ori, buffer)

            if buffer is not None:
                # save buffer...
                image, feats, tgt = next(iter(train_loaders[j]))
                buffer.add_data(image[0], feats, tgt)

        if args.save_pth is not None:
            torch.save(model.state_dict(), args.save_pth)
    else:
        model.load_state_dict(torch.load(args.save_pth, map_location='cpu'))
        evaluate(test_loader, model, do_print=True)
