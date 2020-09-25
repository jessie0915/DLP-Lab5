from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import os
import random
from torch.utils import data
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 31

class Vocabulary(object):
    def __init__(self):
        self.char2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.char2count = {}
        self.index2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_chars = 4  # Count SOS and EOS

        for i in range(26):
            self.addChar(chr(ord('a') + i))

    def addWord(self, word):
        for char in self.split_sequence(word):
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def split_sequence(self, word):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, a, b, e, t]
        """
        return [char for char in word]

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2index['SOS']] if add_sos else []

        for char in self.split_sequence(sequence):
            if char not in self.char2index:
                index_sequence.append((self.char2index['UNK']))
            else:
                index_sequence.append(self.char2index[char])

        if add_eos:
            index_sequence.append(self.char2index['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for idx in indices:
            char = self.index2char[idx]
            if char == 'EOS':
                break
            else:
                sequence += char
        return sequence

class wordsDataset(data.Dataset):
    def __init__(self, train=True):
        if train:
            f = 'train.txt'
        else:
            f = 'test.txt'
        self.datas = np.loadtxt(f, dtype=np.str)

        if train:
            self.datas = self.datas.reshape(-1)
        else:
            '''
            sp -> p
            sp -> pg
            sp -> tp
            sp -> tp
            p  -> tp
            sp -> pg
            p  -> sp
            pg -> sp
            pg -> p
            pg -> tp
            '''
            self.targets = np.array([
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1],
            ])

        # self.tenses = ['sp', 'tp', 'pg', 'p']
        self.tenses = [
            'simple-present',
            'third-person',
            'present-progressive',
            'simple-past'
        ]
        self.charvocab = Vocabulary()

        self.train = train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.train:
            c = index % len(self.tenses)
            # c_one_hot = np.zeros(len(self.tenses), dtype=int)
            # c_one_hot[c] = 1
            return self.charvocab.sequence_to_indices(self.datas[index], add_eos=True), c
        else:
            inp = self.charvocab.sequence_to_indices(self.datas[index, 0], add_eos=True)
            cinp = self.targets[index, 0]
            # cinp_one_hot = np.zeros(len(self.tenses), dtype=int)
            # cinp_one_hot[cinp] = 1
            out = self.charvocab.sequence_to_indices(self.datas[index, 1], add_eos=True)
            cout = self.targets[index, 1]
            # cout_one_hot = np.zeros(len(self.tenses), dtype=int)
            # cout_one_hot[cout] = 1

            return inp, cinp, out, cout

    def indices_to_word(self, indexes):
        word = self.charvocab.indices_to_sequence(indexes)
        return word


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, condition_size, condi_embed_size, latent_size, num_layers=1, dropout=0.0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.condi_embed_size = condi_embed_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.condi_embedding = nn.Embedding(condition_size, condi_embed_size)
        # nn.init.normal_(self.embedding.weight, 0.0, 0.2)
        # nn.init.normal_(self.condi_embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, inp, hidden):
        # input = [input len, batch size]
        # embedded = self.dropout(self.embedding(input)).view(1, 1, -1)
        input_embedded = self.embedding(inp).view(1, 1, -1)

        # input_embedded = [input len, batch size, emb dim]
        # hidden = (hidden_condition, cell_condition)
        output, hidden = self.lstm(input_embedded, hidden)

        # get (1, 1, hidden_size)
        m_hidd = self.mean(hidden[0])
        logvar_hidd = self.logvar(hidden[0])

        normal_sample = self.sample_latent()
        z_hidd = normal_sample * torch.exp(logvar_hidd) ** 0.5 + m_hidd

        m = m_hidd
        logvar = logvar_hidd
        z = z_hidd

        return output, hidden, z, m, logvar


    def initHidden(self, condition):
        c = torch.LongTensor([condition]).to(device)
        condi_embedded = self.condi_embedding(c).view(1, 1, -1)

        h0 = torch.zeros(1, 1, (self.hidden_size - self.condi_embed_size)).to(device)
        c0 = torch.zeros(1, 1, self.hidden_size)

        h0 = torch.cat((h0, condi_embedded), dim=2)

        # hidden = (Variable(nn.Parameter(h0, requires_grad=True)).to(device),
        #           Variable(nn.Parameter(c0, requires_grad=True)).to(device))
        hidden = (Variable(h0).to(device),
                  Variable(c0).to(device))

        return hidden

    def sample_latent(self):
        return torch.normal(
            torch.FloatTensor([0] * self.latent_size),
            torch.FloatTensor([1] * self.latent_size)
        ).to(device)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, latent_size, condi_embed_size, num_layers=1, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.latent_size = latent_size
        self.condi_embed_size = condi_embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.latent_to_hidden = nn.Linear(
            latent_size + condi_embed_size, hidden_size
        )

        self.condi_embedding = nn.Embedding(condition_size, condi_embed_size)

        self.embedding = nn.Embedding(output_size, embedding_size)
        nn.init.normal_(self.embedding.weight, 0.0, 0.2)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        # nn.init.normal_(self.out.weight, 0.0, 0.2)
        # self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):

        # input = [1, batch size]
        # output = self.dropout(self.embedding(input)).view(1, 1, -1)
        output = self.embedding(inp).view(1, 1, -1)
        # embedded = [1, batch size, emb dim]

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        # output = self.softmax(self.out(output[0]))
        output = self.out(output[0])

        return output, hidden

    def initHidden(self, z, condition):
        c = torch.LongTensor([condition]).to(device)
        condi_embedded = self.condi_embedding(c).view(1,1,-1)

        latent_hidd = torch.cat((z, condi_embedded), dim=2)

        hidd = self.latent_to_hidden(latent_hidd)
        cell = torch.zeros(1, 1, self.hidden_size)
        cell = torch.FloatTensor(cell).to(device)

        hidden = (hidd, cell)

        return hidden


# KL(p||q) = 0.5 x {log(sigma_q / sigma_p) + [trans(mean_q - mean_p)*(mean_q - mean_p) / sigma_q]
#                   + tr{sigma_p / sigma_q} - N}
# Now, p~N(m,var), q~N(0,1), and N = 1
def KL_loss(m, logvar):
    kldiv = torch.sum(0.5 * (-logvar + (m ** 2) + torch.exp(logvar) - 1))
    return kldiv

def train(input_tensor, condition, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, kldiv_weights, teacher_forcing_ratio=0.5):

    encoder.train()
    decoder.train()

    encoder_hidden = encoder.initHidden(condition)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        output, encoder_hidden, encoder_latent, encoder_mean, encoder_logvar = encoder(
            input_tensor[ei], encoder_hidden)

    kldiv_loss = KL_loss(encoder_mean, encoder_logvar)
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = decoder.initHidden(encoder_latent, condition)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di].view(-1))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].view(-1))
            if decoder_input.item() == EOS_token:
                break

    (loss + (kldiv_weights * kldiv_loss)).backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), kldiv_loss.item()

def KLD_cost_annealing(type, iteration):
    # Monotonic
    if type == 0:
        slope = 0.001

        w = slope * iteration

        if w > 1.0:
            w = 1.0
    # Cyclic
    else:
        slope = 0.005
        period = 1.0 / slope * 2

        w = slope * (iteration % period)

        if w > 1.0:
            w = 1.0

    return w

def Teacher_Forcing_Ratio_Fcn(iteration):
    # from 1.0 to 0.0
    slope = 0.01
    level = 10
    w = 1.0 - (slope * (iteration // level))
    if w <= 0.0:
        w = 0.0

    return w

def inference(encoder, decoder, input_tensor, input_condition, target_condition, target_len):
    with torch.no_grad():

        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(input_condition)


        for ei in range(input_length):
            output, encoder_hidden, encoder_latent, encoder_mean, encoder_logvar = encoder(
                input_tensor[ei], encoder_hidden)

        # output, encoder_hidden, encoder_latent, encoder_mean, encoder_logvar = encoder(
        #     input_tensor[-1], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        encoder_latent = encoder_latent.view(1, 1, -1)
        decoder_hidden = decoder.initHidden(encoder_latent, target_condition)

        decoded_words = []

        for di in range(target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(train_dataset.charvocab.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def decode_inference(decoder, z, target_condition, target_len):
    with torch.no_grad():
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        z = z.view(1, 1, -1)
        decoder_hidden = decoder.initHidden(z, target_condition)

        decoded_words = []

        for di in range(target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(train_dataset.charvocab.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)

def evaluate_bleu_score(encoder, decoder, mode='test'):
    bleu_score = 0.0
    if mode == 'test':
        print('========================')
    for idx in range(len(test_dataset)):
        data = test_dataset[idx]
        if test_dataset.train:
            inputs, input_condition = data
            targets = inputs
            target_condition = input_condition
        else:
            inputs, input_condition, targets, target_condition = data

        if mode == 'test':
            print('input: ', test_dataset.indices_to_word(inputs))
            print('target:', test_dataset.indices_to_word(targets))

        input_tensor = torch.LongTensor(inputs).to(device)
        output_words = inference(encoder, decoder, input_tensor, input_condition, target_condition, len(targets))
        output_word = ''
        for k in range(len(output_words) - 1):
            output_word += str(output_words[k])
        if mode == 'test':
            print('pred:  ', output_word)
            print('========================')

        bleu_score += compute_bleu(output_word, test_dataset.indices_to_word(targets))

    bleu_score /= len(test_dataset)
    return bleu_score

"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""
def Gaussian_score(words):
    words_list = []
    score = 0
    path = 'train.txt'#should be your directory of train.txt
    with open(path,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def trainIters(encoder, decoder, epochs, print_every=1000, learning_rate=0.01):
    start = time.time()
    ary_losses = []
    ary_kldlosses = []
    ary_bleu_score = []
    print_loss_total = 0  # Reset every print_every
    print_kldivloss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(size_average=False)

    if not os.path.isdir('model'):
        os.mkdir('model')
    if not os.path.isdir('history'):
        os.mkdir('history')

    fp = open('history/history.txt', 'w')
    fp.close()
    data_len = len(train_dataset)
    for epoch in range(1, epochs+1, 1):
        kldiv_weights = KLD_cost_annealing(type=1, iteration=epoch)
        teacher_forcing_ratio = Teacher_Forcing_Ratio_Fcn(epoch)

        for iter in range(data_len):
            data = train_dataset[iter]
            inputs, condition = data

            input_tensor = torch.LongTensor(inputs).to(device)
            target_tensor = torch.LongTensor(inputs).to(device)

            loss, kldiv_loss = train(input_tensor, condition, target_tensor, encoder,
                                     decoder, encoder_optimizer, decoder_optimizer, criterion, kldiv_weights,
                                     teacher_forcing_ratio)
            print_loss_total += loss
            print_kldivloss_total += kldiv_loss


        print_loss_avg = print_loss_total / data_len
        print_loss_total = 0
        ary_losses.append(print_loss_avg)

        print_kldivloss_avg = print_kldivloss_total / data_len
        print_kldivloss_total = 0
        ary_kldlosses.append(print_kldivloss_avg)

        # bleu_score_train = evaluate_all(encoder, decoder, pairs, mode='train')
        bleu_score_test = evaluate_bleu_score(encoder, decoder, mode='train')
        ary_bleu_score.append(bleu_score_test)

        print('%s (%d %d%%): loss=%.4f, kldiv_loss=%.4f, test_bleu_score=%.4f'
              % (timeSince(start, epoch / epochs), epoch,
                 epoch / epochs * 100, print_loss_avg,
                 print_kldivloss_avg, bleu_score_test))

        f = open('history/history.txt', 'a')
        f.write(
            str(epoch) + ', ' + str(print_loss_avg) + ', ' + str(print_kldivloss_avg) + ', ' +
            str(bleu_score_test) + ', ' + str(kldiv_weights) + ', ' + str(teacher_forcing_ratio) + '\n')
        f.close()

        save_path_encoder = 'model/encoder_epoch_' + str(epoch) + '.dict'
        save_path_decoder = 'model/decoder_epoch_' + str(epoch) + '.dict'
        torch.save(encoder.state_dict(), save_path_encoder)
        torch.save(decoder.state_dict(), save_path_decoder)

    showPlot(ary_losses, 'loss.png')
    showPlot(ary_kldlosses, 'kldiv_loss.png')
    showPlot(ary_bleu_score, 'bleu.png')

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    if path == 'kldiv_loss.png':
        plt.plot(points, color="blue", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("loss value")
        plt.title("Loss Curve")
    elif path == 'loss.png':
        plt.plot(points, color="orahge", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("kldiv loss value")
        plt.title("KLDiv Loss Curve")
    else:
        plt.plot(points, color="red", linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("score")
        plt.title("BLEU Score Curve")
        plt.ylim(0.0, 1.0)

    plt.legend()
    plt.savefig(path)

def read_and_show_curve(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    index = []
    cross_entropy_loss = []
    kl_divers_loss = []
    bleu_score = []
    KL_weight = []
    teacher_force = []
    for i in range(len(lines)):
        res = re.split(', ', lines[i])
        index.append(int(res[0])-1)
        cross_entropy_loss.append(float(res[2]))
        kl_divers_loss.append(float(res[4]))
        bleu_score.append(float(res[6]))
        KL_weight.append(float(res[8]))
        teacher_force.append(float(res[10]))

    plt.figure(figsize=(10, 6))
    plt.title('Training\nLoss/Score/Weight Curve')

    plt.plot(index, kl_divers_loss, label='KLD', linewidth=3)
    plt.plot(index, cross_entropy_loss, label='CrossEntropy', linewidth=3)

    plt.xlabel('epoch')
    plt.ylabel('loss')

    h1, l1 = plt.gca().get_legend_handles_labels()

    ax = plt.gca().twinx()
    ax.plot(index, bleu_score, '.', label='BLEU4-score', c="C2")
    ax.plot(index, KL_weight, '--', label='KLD_weight', c="C3")
    ax.plot(index, teacher_force, '--', label='Teacher ratio', c="C4")
    ax.set_ylabel('score / weight')

    h2, l2 = ax.get_legend_handles_labels()

    ax.legend(h1 + h2, l1 + l2)
    plt.show()

def load_modle_and_evaluate(mode='test'):

    encoder1.load_state_dict(torch.load('encoder_best.dict'))
    decoder1.load_state_dict(torch.load('decoder_best.dict'))

    return evaluate_bleu_score(encoder1, decoder1, mode)

def load_modle_and_train():

    encoder1.load_state_dict(torch.load('encoder_best.dict'))
    decoder1.load_state_dict(torch.load('decoder_best.dict'))

    trainIters(encoder1, decoder1, 200, print_every=5000, learning_rate=LR)


def generate_word(decoder, z, condition, maxlen=20):
    decoder.eval()

    output_words = decode_inference(
        decoder, z, condition, target_len=maxlen
    )

    output_word = ''
    for k in range(len(output_words) - 1):
        output_word += str(output_words[k])

    return output_word


def show_noise(noise):
    plt.title('sample Z')
    plt.plot(list(noise))
    plt.show()


def generate_test(decoder, noise):

    show_noise(noise)

    strs = []
    for i in range(len(train_dataset.tenses)):
        output_str = generate_word(decoder, noise, i)
        strs.append(output_str)

    return strs


train_dataset = wordsDataset(train=True)
test_dataset = wordsDataset(train=False)

embedding_size = 256
hidden_size = 256
condition_size = 4
condi_embed_size = 8
latent_size = 32
LR=1e-3
encoder1 = EncoderRNN(train_dataset.charvocab.n_chars, embedding_size, hidden_size, condition_size, condi_embed_size, latent_size).to(device)
decoder1 = DecoderRNN(train_dataset.charvocab.n_chars, embedding_size, hidden_size, latent_size, condi_embed_size).to(device)
trainIters(encoder1, decoder1, 1000, print_every=1000, learning_rate=LR)

# read_and_show_curve('bhistory_1000/history.txt')
# #
# score_test = load_modle_and_evaluate(mode='test')
# # score_new_test = load_modle_and_evaluate(new_test_pairs, mode='test')
# print("BLEU-4 score (test): ", score_test)
# # print("BLEU-4 score (new test): ", score_new_test)
# # print("BLEU-4 score (Avg): ", (score_test + score_new_test)/2.0)
#
# # load_modle_and_train()
#
# # encoder = EncoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
# # decoder = DecoderRNN(data_vocab.n_chars, embedding_size, hidden_size, num_layers=1, dropout=0.3).to(device)
# #
# # encoder.load_state_dict(torch.load('encoder.dict'))
# # decoder.load_state_dict(torch.load('decoder.dict'))
# # while(True):
# #     word = input("Please enter the word << ")
# #     output_words = evaluate(encoder, decoder, word, max_length=MAX_LENGTH)
# #     output_word = ''
# #     for k in range(len(output_words) - 1):
# #         output_word += str(output_words[k])
# #     print(">>", output_word)
#
# words = []
# for k in range(100):
#     noise = encoder1.sample_latent()
#     four_tense_str = generate_test(decoder1, noise)
#     print(four_tense_str)
#     words.append(four_tense_str)
#
# avg_gaussian_score = Gaussian_score(words)
# print('Gaussian Score: ' + str(avg_gaussian_score))



