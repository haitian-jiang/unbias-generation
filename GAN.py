from json import load
import os
import math
import torch
import argparse
import torch.nn as nn
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent
from model import Generator, DiscriminatorEMBD, GAN


def parse_arg(): 
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--data_path', type=str, default=None, help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, help='xx/stage1/ ramdom split for training data')
    parser.add_argument('--save_dir', type=str, default='./save/', help='directory to save the model and generated text')
    parser.add_argument('--outfile_surfix', type=str, default='GAN-gen.txt', help='output file for generated text')
    parser.add_argument('--use_pretrained', action='store_true', help='Use the pretrained ')
    parser.add_argument('--pretrained_path', type=str, default='./save/Clothing', help='directory to the pretrained model')
    # hyper parameter
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--endure_times', type=int, default=5, help='the max endure times of loss increase on validation')
    parser.add_argument('--emsize', type=int, default=512, help='size of embeddings')
    parser.add_argument('--vocab_size', type=int, default=20000, help='keep the most frequent words in the dict')
    # network structure
    parser.add_argument('--words', type=int, default=15, help='number of words to generate for each sample')
    parser.add_argument('--context_reg', type=float, default=1.0, help='regularization on context prediction task')
    parser.add_argument('--text_reg', type=float, default=1.0, help='regularization on text generation task')
    parser.add_argument('--sentiment_reg', type=float, default=1.0, help='regularization on sentiment discriminator loss')
    parser.add_argument('--aspect_reg', type=float, default=1.0, help='regularization on aspect discriminator loss')
    # transformer structure
    parser.add_argument('--nheadG', type=int, default=2, help='the number of heads in generator')
    parser.add_argument('--nhidG', type=int, default=2048, help='number of hidden units per layer in generator')
    parser.add_argument('--nlayerG', type=int, default=2, help='number of layers in generator')
    parser.add_argument('--dropoutG', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--nheadD', type=int, default=8, help='the number of heads in discriminator')
    parser.add_argument('--nhidD', type=int, default=2048, help='number of hidden units per layer in discriminator')
    parser.add_argument('--nlayerD', type=int, default=6, help='number of layers in discriminator')
    parser.add_argument('--dropoutD', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    # training parameters
    parser.add_argument('--lrG', type=float, default=1.0, help='initial learning rate for discriminator')
    parser.add_argument('--lrD', type=float, default=0.1, help='initial learning rate for discriminator')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--log_interval', type=int, default=200, help='switching training G & D')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    return parser.parse_args()


def load_pretrained(args):
    dataloader = torch.load(os.path.join(args.pretrained_path), 'dataloader.pt')
    generator = torch.load(os.path.join(args.pretrained_path), 'model.pt')
    aspect_discriminator = torch.load(os.path.join(args.pretrained_path), 'discriminator-aspect_cluster.pt')
    sentiment_discriminator = torch.load(os.path.join(args.pretrained_path), 'discriminator-sentiment.pt')
    return dataloader, generator, aspect_discriminator, sentiment_discriminator



if __name__ == '__main__':
    args = parse_arg()
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, 'GAN.pt')

    # pretrained models, None if no such models
    preLD, preG, preDA, preDS = [None] * 4
    trained_embds = [None] * 5
    if args.use_pretrained:
        preLD, preG, preDA, preDS = load_pretrained(args)
        trained_embds = [preG.user_embeddings, preG.item_embeddings, 
                         preG.aspect_embeddings, preG.senti_embeddings, preG.word_embeddings]
    trained_model = preG, preDS, preDA

    # load data
    print(now_time() + 'Loading data')
    corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, True, preLD)
    word2idx = corpus.word_dict.word2idx
    idx2word = corpus.word_dict.idx2word

    train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
    val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
    test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)

    # build model
    tgt_len = args.words + 1  # added <bos> or <eos>
    pad_idx = word2idx['<pad>']
    ntoken = len(corpus.word_dict); nuser = len(corpus.user_dict)
    nitem = len(corpus.item_dict); nsenti = len(corpus.senti_dict)
    naspect = len(corpus.aspect_dict); n_embds = (nuser, nitem, nsenti, naspect, ntoken)
    gan = GAN(args.emsize, n_embds, tgt_len, pad_idx, args, trained_embds, trained_model, device)

    # Loop over epochs.
    best_val_loss = float('inf')
    endure_count = 0
    for epoch in range(1, args.epochs + 1):
        print(now_time() + f'epoch {epoch}')
        gan.train(train_data)
        Gloss = gan.evaluate(val_data)
        # Save the model if the validation loss is the best we've seen so far.
        if Gloss < best_val_loss:
            best_val_loss = Gloss
            with open(model_path, 'wb') as f:
                torch.save(gan, f)
        else:
            endure_count += 1
            print(now_time() + 'Endured {} time(s)'.format(endure_count))
            if endure_count == args.endure_times:
                print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            gan.schedulerG.step()
            print(now_time() + 'Learning rate of G set to {:2.8f}'.format(gan.schedulerG.get_last_lr()[0]))

    with open(model_path, 'rb') as f:
        best_model = torch.load(f).to(device)

    # Run on test data.
    best_model.evaluate(test_data, 'test')

    print(now_time() + 'Generating text')
    text_o = gan.generate(test_data, word2idx, idx2word)
    prediction_path = os.path.join(args.save_dir, args.outfile_surfix)  # TODO
    with open(prediction_path, 'w', encoding='utf-8') as f:
        f.write(text_o)
    print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
