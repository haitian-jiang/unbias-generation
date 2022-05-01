import os
import math
import json
import random
import torch
import argparse
from model import MLPDiscriminator
from torch.nn import BCELoss
from utils import now_time, sentence_format

class DataLoader:
    def __init__(self, sentence_dir, anchor_path, index_dir, dictidx_table, anchor_type='aspect_cluster'):
        # (validation, test) from generation
        assert anchor_type == 'aspect_cluster' or anchor_type == 'sentiment'
        self.word_dict = dictidx_table['word']
        self.__unk = self.word_dict.word2idx['<unk>']
        anchor_gt = self.load_anchor(anchor_path, index_dir, anchor_type, dictidx_table[anchor_type])
        sentence_gt, sentence_gen = self.load_sentence(sentence_dir)
        self.train, self.valid, self.test = self.split_data(anchor_gt, sentence_gt, sentence_gen)

    def load_anchor(self, anchor_path, index_dir, anchor_type, anchor_table):
        reviews = []
        with open(anchor_path, "r") as f:
            for line in f:
                reviews.append(json.loads(line))
        
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        
        index = valid_index + test_index
        anchor_gt = []
        for i in index:
            anchor_gt.append(anchor_table.entity2idx[reviews[i][anchor_type]])
        return anchor_gt
        
    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    def load_sentence(self, sentence_dir):
        sentence_gt, sentence_gen = [], []
        with open(os.path.join(sentence_dir, 'generated-validation.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)//4):
                sentence_gt.append(self.seq2ids(lines[i*4].strip()))
                sentence_gen.append(self.seq2ids(lines[i*4+2].strip()))
        with open(os.path.join(sentence_dir, 'generated-test.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)//4):
                sentence_gt.append(self.seq2ids(lines[i*4].strip()))
                sentence_gen.append(self.seq2ids(lines[i*4+2].strip()))
        return sentence_gt, sentence_gen

    def split_data(self, anchor_gt, sentence_gt, sentence_gen):
        assert len(anchor_gt) == len(sentence_gt) and len(sentence_gt) == len(sentence_gen)
        length = len(anchor_gt)
        train, valid, test = {}, {}, {}
        train['anchor_gt'] = anchor_gt[:length//10*8]
        train['sentence_gt'] = sentence_gt[:length//10*8]
        train['sentence_gen'] = sentence_gen[:length//10*8]
        valid['anchor_gt'] = anchor_gt[length//10*8: length//10*9]
        valid['sentence_gt'] = sentence_gt[length//10*8: length//10*9]
        valid['sentence_gen'] = sentence_gen[length//10*8: length//10*9]
        test['anchor_gt'] = anchor_gt[length//10*9:]
        test['sentence_gt'] = sentence_gt[length//10*9:]
        test['sentence_gen'] = sentence_gen[length//10*9:]
        return train, valid, test

def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)

class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False) -> None:
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        anchor_gt = data['anchor_gt']
        sentence_gt = [sentence_format(s, seq_len, pad, bos, eos) for s in data['sentence_gt']]
        sentence_gen = [sentence_format(s, seq_len, pad, bos, eos) for s in data['sentence_gen']]
        self.anchor_gt = torch.tensor(anchor_gt, dtype=torch.int64).contiguous()
        self.sentence_gt = torch.tensor(sentence_gt, dtype=torch.int64).contiguous()
        self.sentence_gen = torch.tensor(sentence_gen, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(anchor_gt)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
    
    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)
        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        anchor_gt = self.anchor_gt[index]
        sentence_gt = self.sentence_gt[index]
        sentence_gen = self.sentence_gen[index]
        label = torch.tensor([1.]*len(sentence_gt)+[0.]*len(sentence_gen))
        return torch.cat([anchor_gt,anchor_gt]), torch.cat([sentence_gt, sentence_gen]), label
        

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_dir', type=str, help='./save/xx')
    parser.add_argument('--gt_path', type=str, help='xx/ABSA/')
    parser.add_argument('--index_dir', type=str, help='xx/stage1/')
    parser.add_argument('--anchor_type', type=str, default='aspect_cluster')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayer', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--endure_times', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--margin', type=int, default=1, help='ranking loss margin')
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()



args = parse_arg()
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device('cuda')

generator = torch.load(os.path.join(args.sentence_dir, 'model.pt'))
dictidx_table = torch.load(os.path.join(args.sentence_dir, 'dataloader.pt'))
print(now_time() + 'Loading data')
dataset = DataLoader(args.sentence_dir, args.gt_path, args.index_dir, dictidx_table, args.anchor_type)
word2idx = dictidx_table['word'].word2idx
train_data = Batchify(dataset.train, word2idx, shuffle=True, batch_size=args.batch_size)
val_data = Batchify(dataset.valid, word2idx, batch_size=args.batch_size)
test_data = Batchify(dataset.test, word2idx, batch_size=args.batch_size)

pad_idx = word2idx['<pad>']
word_embeddings = generator.word_embeddings
anchor_embeddings = generator.senti_embeddings if args.anchor_type == 'sentiment' else generator.aspect_embeddings
nanchor = len(dictidx_table[args.anchor_type])
model = MLPDiscriminator(pad_idx, word_embeddings, nanchor, anchor_embeddings, nhead=args.nhead, nlayers=args.nlayer).to(device)
criterion = BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

def train(data):
    model.train()
    total_loss, total_sample = 0.0, 0
    while True:
        anchor, sentence, label = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = anchor.size(0)
        anchor = anchor.to(device)  # (batch_size,)
        sentence = sentence.t().to(device)  # (seq_len, batch_size)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(sentence, anchor)  # (batch_size, emsize)
        loss = criterion(pred, label)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_loss = total_loss / total_sample
            print(now_time() + 'loss: {:4.4f} | {:5d}/{:5d} batches'.format(
                cur_loss, data.step, data.total_step))
            total_loss = 0
            total_sample = 0
        if data.step == data.total_step:
            break

@torch.no_grad()
def evaluate(data):
    model.eval()
    total_correct, total_loss, total_sample = 0, 0, 0
    while True:
        anchor, sentence, label = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = anchor.size(0)
        anchor = anchor.to(device)  # (batch_size,)
        sentence = sentence.t().to(device)  # (seq_len, batch_size)
        label = label.to(device)
        optimizer.zero_grad()
        pred = model(sentence, anchor)  # (batch_size, emsize)
        loss = criterion(pred, label)

        total_loss += batch_size * loss.item()
        total_sample += batch_size

        correct = ((pred > 0.5).int() == label).sum()
        total_correct += correct
        if data.step == data.total_step:
            break
    return total_correct / total_sample, total_loss / total_sample




# Loop over epochs.
best_val_acc = float('inf')
endure_count = 0
model_path = os.path.join(args.sentence_dir, f'discriminatorMLPAdam-{args.anchor_type}.pt')
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_acc, val_loss = evaluate(val_data)
    print(now_time() + f'Accuracy {val_acc:4.4f} | loss {val_loss:4.4f} on validation')
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_acc:
        best_val_acc = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        # scheduler.step()
        # print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
test_acc, test_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + f'Accuracy {test_acc:4.4f} | loss {test_loss:4.4f} on test, end of training.')