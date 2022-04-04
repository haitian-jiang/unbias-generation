import os
import math
import torch
import argparse
import torch.nn as nn
from module import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='Generator')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--checkpoint', type=str, default='./save/',
                    help='directory to save the final model')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--load_json', action='store_true',
                    help='True to use json; Otherwise use pickle')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--dataset', type=str, default='test',
                    help='train, validation or test')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, f"generated-{args.dataset}.txt")

print(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, load_json=args.load_json)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
# feature_set = corpus.feature_set
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)

def generate(data):
    model.eval()
    idss_predict = []
    context_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, aspect, senti = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            senti = senti.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
            aspect = aspect.t().to(device)  # (1, batch_size)
            text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for idx in range(args.words):
                if idx == 0:
                    log_word_prob, log_context_dis, _ = model(user, item, senti, aspect, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    context = predict(log_context_dis, topk=args.words)  # (batch_size, words)
                    context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _ = model(user, item, senti, aspect, text, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)
            if data.step == data.total_step: break
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    return text_out




print(now_time() + 'Generating text')
if args.dataset == 'test':
    text_o = generate(test_data)
elif args.dataset == 'train':
    text_o = generate(train_data)
elif args.dataset == 'validation':
    text_o = generate(val_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))