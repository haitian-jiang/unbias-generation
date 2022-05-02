from module import *
from utils import now_time, ids2tokens, bleu_score, unique_sentence_percent, rouge_score

class MLPReLU(nn.Module):
    def __init__(self, emsize):
        super().__init__()
        self.linear1 = nn.Linear(emsize, 2*emsize)
        self.linear2 = nn.Linear(2*emsize, emsize)
        self.linear3 = nn.Linear(emsize, 1)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        out = self.relu(self.linear1(hidden))  # (batch_size, emsize)
        out = self.relu(self.linear2(out))
        out = torch.squeeze(self.linear3(out))  # (batch_size,)
        return out

class MLPDiscriminator(nn.Module):
    def __init__(self, pad_idx, word_embeddings, nanchor, anchor_embeddings=None, dropout=0.2, nhead=8, nhid=2048, nlayers=6) -> None:
        super().__init__()
        self.word_embeddings = word_embeddings
        self.emsize = word_embeddings.embedding_dim

        if anchor_embeddings is not None:
            self.anchor_embeddings = anchor_embeddings
        else:
            self.anchor_embeddings = nn.Embedding(nanchor, self.emsize)

        self.pos_encoder = PositionalEncoding(self.emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.score = MLPReLU(2*self.emsize)
        self.sigmoid = nn.Sigmoid()
        self.pad_idx = pad_idx
        
    def forward(self, text, anchor=None):
        pad_mask = text.t() == self.pad_idx
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        a_src = self.anchor_embeddings(anchor) if anchor is not None else None
        src = w_src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src, None, pad_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        kitten = torch.cat([hidden[0], a_src], 1)  # (batch_size, emsize) -> (batch_size, 2*emsize)
        out = self.score(kitten)
        return self.sigmoid(out)


class DiscriminatorEMBD(nn.Module):
    def __init__(self, emsize, dropout=0.2, nhead=8, nhid=2048, nlayers=6) -> None:
        super().__init__()
        self.emsize = emsize

        self.pos_encoder = PositionalEncoding(self.emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.score = MLPReLU(2*self.emsize)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, w_src, a_src, pad_mask):
        src = w_src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src, None, pad_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        kitten = torch.cat([hidden[0], a_src], 1)  # (batch_size, emsize) -> (batch_size, 2*emsize)
        out = self.score(kitten)
        return self.sigmoid(out)

class Generator(nn.Module):
    def __init__(self, tgt_len, pad_idx, emsize, ntoken, nhead, nhid, nlayers, dropout, embds) -> None:
        super().__init__()
        self.src_len = 4
        self.pad_idx = pad_idx
        self.emsize = emsize
        self.user_embeddings = embds[0]
        self.item_embeddings = embds[1]
        self.senti_embeddings = embds[2]
        self.aspect_embeddings = embds[3]
        self.word_embeddings = embds[4]
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.attn_mask = generate_peter_mask(self.src_len, tgt_len)
        initrange = 0.1
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, senti, aspect, text, seq_prediction=True, context_prediction=True):
        device = user.device
        batch_size = user.size(0)
        total_len = self.src_len + text.size(0)
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.src_len).bool().to(device)  # (batch_size, src_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - src_len)
        pad_mask_G = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        s_src = self.senti_embeddings(senti.unsqueeze(0))
        a_src = self.aspect_embeddings(aspect.unsqueeze(0))
        w_src = self.word_embeddings(text)  # (total_len - src_len, batch_size, emsize)
        src = torch.cat([u_src, i_src, a_src, s_src, w_src])  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src, attn_mask, pad_mask_G)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, right, w_src



class GAN(object):
    """ Architecture
                real word embd ┌─────────────┐
                  ┌───────────▶│Discriminator├──▶loss(aspect,sentiment)
                  │            └─────────────┘
                  │                   ▲ 
                  │    fake word embd,│sentiment & aspect embd
            ┌─────┴─────┐      ┌──────┴──────┐
    input──▶│ embedding ├─────▶│  Generator  ├──▶loss(text,context)
            └───────────┘      └─────────────┘
    """
    def __init__(self, emsize, n_embds, tgt_len, pad_idx, args, trained_embds, trained_model, device):
        super().__init__()
        # unpacking parameters
        self.nuser, self.nitem, self.nsenti, self.naspect, self.ntoken = n_embds
        user_embd, item_embd, asp_embd, sent_embd, word_embd = trained_embds
        trainedG, trainedDS, trainedDA = trained_model
        self.device = device
        self.cpu = torch.device('cpu')
        self.args = args
        self.tgt_len = tgt_len
        
        # embeddings for generator and discriminator
        self.user_embeddings = user_embd if user_embd is not None else nn.Embedding(self.nuser, emsize)
        self.item_embeddings = item_embd if item_embd is not None else nn.Embedding(self.nitem, emsize)
        self.senti_embeddings = sent_embd if sent_embd is not None else nn.Embedding(self.nsenti, emsize)
        self.aspect_embeddings = asp_embd if asp_embd is not None else nn.Embedding(self.naspect, emsize)
        self.word_embeddings = word_embd if word_embd is not None else nn.Embedding(self.ntoken, emsize)

        # generator
        embds = (self.user_embeddings, self.item_embeddings, self.senti_embeddings, self.aspect_embeddings, self.word_embeddings)
        self.netG = Generator(tgt_len, pad_idx, args.emsize, self.ntoken, args.nheadG, args.nhidG, args.nlayerG, args.dropoutG, embds).to(device)
        if trainedG is not None:
            self.netG.pos_encoder = trainedG.pos_encoder
            self.netG.transformer_encoder = trainedG.transformer_encoder
            self.netG.hidden2token = trainedG.hidden2token

        # discriminators
        self.netDS = DiscriminatorEMBD(args.emsize, args.dropoutD, args.nheadD, args.nhidD, args.nlayerD).to(device)
        self.netDA = DiscriminatorEMBD(args.emsize, args.dropoutD, args.nheadD, args.nhidD, args.nlayerD).to(device)
        if trainedDS is not None:
            self.netDS.pos_encoder = trainedDS.pos_encoder
            self.netDS.transformer_encoder = trainedDS.transformer_encoder
            self.netDS.score = trainedDS.score
        if trainedDA is not None:
            self.netDA.pos_encoder = trainedDA.pos_encoder
            self.netDA.transformer_encoder = trainedDA.transformer_encoder
            self.netDA.score = trainedDA.score

        self.optimG = torch.optim.SGD(self.netG.parameters(), lr=args.lrG)
        self.optimDS = torch.optim.SGD(self.netDS.parameters(), lr=args.lrD)
        self.optimDA = torch.optim.SGD(self.netDA.parameters(), lr=args.lrD)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimG, 1, gamma=0.25)

        self.criterionG = nn.NLLLoss(ignore_index=pad_idx)
        self.criterionD = nn.BCELoss()
        
    def trainG(self, batch, label_real):
        self.netG.train()
        args = self.args
        """prepare data"""
        user, item, _, seq, aspect, senti = batch  # (batch_size, seq_len), data.step += 1
        user = user.to(self.device)  # (batch_size,)
        item = item.to(self.device)
        senti = senti.to(self.device)  # (batch_size,)
        seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
        aspect = aspect.to(self.device)  # (1, batch_size)
        text = seq[:-1]  # (tgt_len, batch_size)
            
        """loss from G"""
        self.optimG.zero_grad()
         # (tgt_len, batch_size, ntoken), (batch_size, ntoken), ... , (tgt_len, batch_size, emsize)
        log_word_prob, log_context_dis, txt_pad_mask, real_w_src = self.netG(user, item, senti, aspect, text) 
        context_dis = log_context_dis.unsqueeze(0).repeat((self.tgt_len-1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        """loss from D"""
        word_prob = log_word_prob.exp()
        w_src = word_prob.matmul(self.word_embeddings.weight)
        s_src = self.senti_embeddings(senti)  # (batch_size, emsize)
        a_src = self.aspect_embeddings(aspect)
        pred_s = self.netDS(w_src, s_src, txt_pad_mask)
        pred_a = self.netDA(w_src, a_src, txt_pad_mask)

        c_loss = self.criterionG(context_dis.view(-1, self.ntoken), seq[1:-1].reshape((-1,)))
        t_loss = self.criterionG(log_word_prob.view(-1, self.ntoken), seq[1:].reshape((-1,)))
        s_loss = self.criterionD(pred_s, label_real)
        a_loss = self.criterionD(pred_a, label_real)
        g_loss = args.text_reg*t_loss + args.context_reg*c_loss + args.sentiment_reg*s_loss + args.aspect_reg*a_loss
        # losses = [l.item() for l in (c_loss, t_loss, s_loss, a_loss, g_loss)]
        g_loss.backward(retain_graph=True)

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), args.clip)
        self.optimG.step()
        # return real_w_src, w_src, s_src, a_src, losses, txt_pad_mask
        return c_loss.item(), t_loss.item(), s_loss.item(), a_loss.item(), g_loss.item()

    def trainDS(self, s_src, w_src, label, pad_mask):
        self.netDS.train()
        args = self.args
        self.optimDS.zero_grad()
        pred = self.netDS(w_src, s_src, pad_mask) 
        loss = self.criterionD(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netDS.parameters(), args.clip)
        self.optimDS.step()
        return loss.item()

    def trainDA(self, a_src, w_src, label, pad_mask):
        self.netDA.train()
        args = self.args
        self.optimDA.zero_grad()
        pred = self.netDA(w_src, a_src, pad_mask)  # (batch_size, emsize)
        loss = self.criterionD(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netDA.parameters(), args.clip)
        self.optimDA.step()
        return loss.item()

    def train(self, data):
        Closs = Tloss = GSloss = GAloss = Gloss = DSloss = DAloss = tot_sample = 0
        batches = []  # store the middle result
        while True:
            batch = data.next_batch()  # (user, item, rating, seq, aspect, senti)
            batch_size = batch[0].size(0)
            tot_sample += batch_size

            label_real = torch.tensor([1.]*batch_size).to(self.device)
            label_fake = torch.tensor([0.]*batch_size).to(self.device)
            labelD = torch.cat([label_real, label_fake]).to(self.cpu)
            batches.append([batch, labelD])

            c_loss, t_loss, gs_loss, ga_loss, g_loss = self.trainG(batch, label_real)
            # c_loss, t_loss, gs_loss, ga_loss, g_loss = losses  # context, text, sentiment, aspect, generator
            Closs += batch_size * c_loss; Tloss += batch_size * t_loss
            GSloss += batch_size * gs_loss; GAloss += batch_size * ga_loss
            Gloss += batch_size * g_loss

            if data.step % self.args.log_interval != 0 and data.step != data.total_step:
                continue  # only update discriminator after `log_interval` steps
            else:
                print(now_time()+'Training:')
                print(f"Generator: context ppl {math.exp(Closs/tot_sample):4.4f} | text ppl {math.exp(Tloss/tot_sample):4.4f} | " + \
                      f"aspect loss {GAloss/tot_sample:4.4f} | sentiment loss {GSloss/tot_sample:4.4f} | " + \
                      f"loss {Gloss/tot_sample:4.4f} | {data.step:5d}/{data.total_step:5d} batches")
                Closs = Tloss = GSloss = GAloss = Gloss = 0

                # train discriminator for these batches
                for batch, labelD in batches:
                    """fake data from G"""
                    user, item, _, seq, aspect, senti = [i.to(self.device) for i in batch]
                    s_src = self.senti_embeddings(senti)  # (batch_size, emsize)
                    a_src = self.aspect_embeddings(aspect)
                    s_src = torch.cat([s_src, s_src]); a_src = torch.cat([a_src, a_src])
                    text = seq.t()[:-1]  # (tgt_len, batch_size)

                    log_word_prob, _, txt_pad_mask, real_w_src = self.netG(user, item, senti, aspect, text) 
                    txt_pad_mask_D = torch.cat([txt_pad_mask, txt_pad_mask])
                    word_prob = log_word_prob.exp()
                    fake_w_src = word_prob.matmul(self.word_embeddings.weight)
                    w_src = torch.cat([real_w_src, fake_w_src], dim=1)
                    ds_loss = self.trainDS(s_src, w_src, labelD.to(self.device), txt_pad_mask_D)
                    # get fake text second time to avoid setting `retain_graph=True`
                    log_word_prob, _, txt_pad_mask, real_w_src = self.netG(user, item, senti, aspect, text) 
                    word_prob = log_word_prob.exp()
                    fake_w_src = word_prob.matmul(self.word_embeddings.weight)
                    w_src = torch.cat([real_w_src, fake_w_src], dim=1)
                    da_loss = self.trainDA(a_src, w_src, labelD.to(self.device), txt_pad_mask_D)
                    DSloss += batch_size * ds_loss; DAloss += batch_size * da_loss
                print(now_time()+'Training:')
                print(f"Discriminator: aspect loss {DAloss/tot_sample:4.4f} | sentiment loss {DSloss/tot_sample:4.4f}")
                batches = []
                DSloss = DAloss = tot_sample = 0
            if data.step == data.total_step:
                break

    @torch.no_grad()
    def evaluate(self, data, msg="validation"):
        args = self.args
        self.netG.eval()
        self.netDS.eval()
        self.netDA.eval()
        Closs = Tloss = GSloss = GAloss = Gloss = DSloss = DAloss = tot_sample = 0
        while True:
            (user, item, rating, seq, aspect, senti) = data.next_batch() 
            batch_size = user.size(0)
            tot_sample += batch_size

            label_real = torch.tensor([1.]*batch_size).to(self.device)
            label_fake = torch.tensor([0.]*batch_size).to(self.device)
            labelD = torch.cat([label_real, label_fake])

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            senti = senti.to(self.device)  # (batch_size,)
            seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
            aspect = aspect.t().to(self.device)  # (1, batch_size)
            text = seq[:-1]  # (tgt_len, batch_size)
            
            """loss from G"""
            # (tgt_len, batch_size, ntoken), (batch_size, ntoken), ... , (tgt_len, batch_size, emsize)
            log_word_prob, log_context_dis, txt_pad_mask_G, real_w_src = self.netG(user, item, senti, aspect, text) 
            context_dis = log_context_dis.unsqueeze(0).repeat((self.tgt_len-1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            """loss from D"""
            word_prob = log_word_prob.exp()
            fake_w_src = word_prob.matmul(self.word_embeddings.weight)
            s_src = self.senti_embeddings(senti)  # (batch_size, emsize)
            a_src = self.aspect_embeddings(aspect)
            txt_pad_mask_D = torch.cat([txt_pad_mask_G, txt_pad_mask_G])
            pred_s = self.netDS(fake_w_src, s_src, txt_pad_mask_G)
            pred_a = self.netDA(fake_w_src, a_src, txt_pad_mask_G)

            c_loss = self.criterionG(context_dis.view(-1, self.ntoken), seq[1:-1].reshape((-1,)))
            t_loss = self.criterionG(log_word_prob.view(-1, self.ntoken), seq[1:].reshape((-1,)))
            gs_loss = self.criterionD(pred_s, label_real)
            ga_loss = self.criterionD(pred_a, label_real)
            g_loss = args.text_reg*t_loss + args.context_reg*c_loss + args.sentiment_reg*gs_loss + args.aspect_reg*ga_loss

            Closs += batch_size * c_loss.item(); Tloss += batch_size * t_loss.item()
            GSloss += batch_size * gs_loss.item(); GAloss += batch_size * ga_loss.item()
            Gloss += batch_size * g_loss.item()

            w_src = torch.cat([real_w_src, fake_w_src], dim=1)
            pred_s = self.netDS(w_src, torch.cat([s_src, s_src]), txt_pad_mask_D) 
            pred_a = self.netDA(w_src, torch.cat([s_src, s_src]), txt_pad_mask_D)
            ds_loss = self.criterionD(pred_s, labelD)
            da_loss = self.criterionD(pred_a, labelD)
            DSloss += batch_size * ds_loss; DAloss += batch_size * da_loss

            if data.step % args.log_interval == 0 or data.step == data.total_step:
                print('\n' + now_time() + f'Evaluating on {msg}: ')
                print(f"Generator: context ppl {math.exp(Closs/tot_sample):4.4f} | text ppl {math.exp(Tloss/tot_sample):4.4f} | " + \
                      f"aspect loss {GAloss/tot_sample:4.4f} | sentiment loss {GSloss/tot_sample:4.4f} | " + \
                      f"loss {Gloss/tot_sample:4.4f} | {data.step:5d}/{data.total_step:5d} batches")
                Closs = Tloss = GSloss = GAloss = Gloss = 0
                print(f"Discriminator: aspect loss {DAloss/tot_sample:4.4f} | sentiment loss {DSloss/tot_sample:4.4f}")
                DSloss = DAloss = tot_sample = 0
            if data.step == data.total_step:
                break
        return Gloss

    @torch.no_grad()
    def generate(self, data, word2idx, idx2word):
        self.netG.eval()
        args = self.args
        idss_predict = []
        context_predict = []
        while True:
            user, item, rating, seq, aspect, senti = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            senti = senti.to(self.device)
            bos = seq[:, 0].unsqueeze(0).to(self.device)  # (1, batch_size)
            aspect = aspect.t().to(self.device)  # (1, batch_size)
            text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for i in range(args.words):
                # produce a word at each step
                log_word_prob, _, _, _ = self.netG(user, item, senti, aspect, text, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

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