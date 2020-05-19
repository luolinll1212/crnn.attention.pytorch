# *_*coding:utf-8 *_*

import argparse
from PIL import Image

import torch

import src.utils as utils
import src.dataset as dataset

import models.crnn as crnn

import config as cfg


# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(cfg.alphabet)

transformer = dataset.ResizeNormalize(img_width=cfg.img_width, img_height=cfg.img_height)


def seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    prob = 1.0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()

    return words, prob


def main():
    image = Image.open(cfg.img_path).convert('RGB')
    image = transformer(image)
    if torch.cuda.is_available() and cfg.use_gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image)

    encoder = crnn.Encoder(3, cfg.hidden_size)
    # no dropout during inference
    decoder = crnn.Decoder(cfg.hidden_size, cfg.num_classes, dropout_p=0.0, max_length=cfg.max_width)

    if torch.cuda.is_available() and cfg.use_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    encoder.load_state_dict(torch.load(cfg.encoder, map_location=map_location))
    print('loading pretrained encoder models from {}.'.format(cfg.encoder))
    decoder.load_state_dict(torch.load(cfg.decoder, map_location=map_location))
    print('loading pretrained decoder models from {}.'.format(cfg.decoder))

    encoder.eval()
    decoder.eval()

    encoder_out = encoder(image)

    max_length = 20
    decoder_input = torch.zeros(1).long()
    decoder_hidden = decoder.initHidden(1)
    if torch.cuda.is_available() and cfg.use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_hidden = decoder_hidden.cuda()

    words, prob = seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length)
    print('predict_string: {} => predict_probility: {}'.format(words, prob))


if __name__ == "__main__":
    main()