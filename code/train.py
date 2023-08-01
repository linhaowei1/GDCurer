import os
import logging
from transformers import ConvNextFeatureExtractor, ConvNextModel
import torch
import argparse
from DataHelper import get_dataset
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
    

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
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
parser.add_argument('--zero_loss', action='store_true')
parser.add_argument('--onlymse', action='store_true')
parser.add_argument('--extra_info', type=str, default='')
# ablation on input feature
parser.add_argument('--woimage', action='store_true')
parser.add_argument('--wohalflife', action='store_true')
parser.add_argument('--wouptake', action='store_true')
parser.add_argument('--nomse', action='store_true')
parser.add_argument('--noloss', action='store_true')
args = parser.parse_args()

setup_seed(args.seed)

if args.wandb:
    wandb.init(
        project='GDCurer',
        group=f'epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}',
        name=f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}',
    )

args.data_path = os.path.join(args.base_dir, 'data', 'data_all.xlsx')
args.image_path = os.path.join(args.base_dir, 'images')

if args.save_pth is None:
    args.save_pth = os.path.join(args.base_dir, 'model', f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}')

logging.basicConfig(
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename=os.path.join(args.base_dir, './logs', f'seed{args.seed}_bs{args.batch_size}_epoch{args.epochs}_{args.lambda1}_{args.lambda2}_{args.lambda3}_{args.lambda4}_{args.lambda5}_{args.extra_info}'),
    level=logging.INFO,
    filemode='w'
)

def train(dataloader, testloader, model, epoch, optimizer, scheduler):
    for epo in tqdm(range(epoch)):
        model.train()
        tot_loss = 0.0
        LOSS_EACH = [0.0] * 5
        tot_zero_loss = 0.0

        for image, feat, tgt in dataloader:

            image = image[0].cuda()
            feat = feat.cuda()

            if args.woimage:
                image = torch.zeros_like(image).cuda()
            if args.wohalflife:
                feat[:, 1] = 0.0
            if args.wouptake:
                feat[:, 0] = 0.0

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
            
            if args.onlymse:
                loss = LOSS[2]
            elif args.nomse:
                loss = LOSS[0] + LOSS[1] + LOSS[3] + LOSS[4]
            elif args.noloss:
                loss = 0
            else:
                loss = sum(LOSS)
            
            zero_loss = F.relu(-outputs).sum()
            tot_zero_loss += zero_loss.item()

            if args.zero_loss:
                loss += zero_loss


            LOSS_EACH = [LOSS_EACH[i] + LOSS[i].item() for i in range(5)]
            tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()
        
        logging.info(json.dumps({f'train_loss tot': tot_loss, "lr": scheduler.get_last_lr()[0]}))
        logging.info(json.dumps({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)}))

        print(json.dumps({f'train_loss tot': tot_loss, "lr": scheduler.get_last_lr()[0]}))
        print(json.dumps({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)}))
        
        logging.info(json.dumps({f'zero_loss tot': tot_zero_loss}))

        if args.wandb:
            wandb.log({f'train_loss tot': tot_loss, "lr": scheduler.get_last_lr()[0]})
            wandb.log({f'train_loss {i-2}': LOSS_EACH[i] for i in range(5)})
            wandb.log({f'zero_loss tot': tot_zero_loss})

        scheduler.step()
        evaluate(testloader, model)

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

            if args.woimage:
                image = torch.zeros_like(image).cuda()
            if args.wohalflife:
                feat[:, 1] = 0.0
            if args.wouptake:
                feat[:, 0] = 0.0

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
            
            if args.onlymse:
                loss = LOSS[2]
            elif args.nomse:
                loss = LOSS[0] + LOSS[1] + LOSS[3] + LOSS[4]
            elif args.noloss:
                loss = 0
            else:
                loss = sum(LOSS)

            zero_loss = F.relu(-outputs).sum()
            tot_zero_loss += zero_loss.item()

            if args.zero_loss:
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
    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    model = Net()

    model = model.cuda()
    if args.wandb:
        wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.training:
        train(train_loader, test_loader, model, epochs, optimizer, scheduler)
        if args.save_pth is not None:
            torch.save(model.state_dict(), args.save_pth)
        evaluate(test_loader, model, do_print=True)
    else:
        model.load_state_dict(torch.load(args.save_pth, map_location='cpu'))
        evaluate(test_loader, model, do_print=True)