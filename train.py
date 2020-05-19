# *_*coding:utf-8 *_*
import argparse
import random
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import src.utils as utils
import src.dataset as dataset

import models.crnn as crnn

import config as cfg

cudnn.benchmark = True


# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(cfg.alphabet)


def train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=1, cfg=None, epoch=0):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()

    train_iter = iter(train_loader)

    for i in range(len(train_loader)):
        cpu_images, cpu_texts = train_iter.next()
        batch_size = cpu_images.size(0)

        for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
            encoder_param.requires_grad = True
            decoder_param.requires_grad = True
        encoder.train()
        decoder.train()

        target_variable = converter.encode(cpu_texts)
        utils.load_data(image, cpu_images)

        # CNN + BiLSTM
        encoder_outputs = encoder(image)
        target_variable = target_variable.cuda()
        # start decoder for SOS_TOKEN
        decoder_input = target_variable[utils.SOS_TOKEN].cuda()
        decoder_hidden = decoder.initHidden(batch_size).cuda()

        loss = 0.0
        teach_forcing = True if random.random() > teach_forcing_prob else False
        if teach_forcing:
            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
        else:
            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_avg.add(loss)

        if i % cfg.interval == 0:
            print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, cfg.num_epochs, i, len(train_loader),
                                                                     loss_avg.val()))
            loss_avg.reset()

    # save checkpoint
    torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(cfg.model, epoch))
    torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(cfg.model, epoch))


def evaluate(image, text, encoder, decoder, data_loader, max_eval_iter=100):
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    for i in range(min(len(data_loader), max_eval_iter)):
        cpu_images, cpu_texts = val_iter.next()
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = encoder(image)
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()
        decoder_hidden = decoder.initHidden(batch_size).cuda()

        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == utils.EOS_TOKEN:
                decoded_label.append(utils.EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:, :]):
            if pred == target:
                n_correct += 1

        if i % 10 == 0:
            texts = cpu_texts[0]
            print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

    accuracy = n_correct / float(n_total)
    print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))


def main():
    if not os.path.exists(cfg.model):
        os.makedirs(cfg.model)

    # create train dataset
    train_dataset = dataset.TextLineDataset(text_line_file=cfg.train_list, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=sampler, num_workers=int(cfg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=cfg.img_height, img_width=cfg.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(text_line_file=cfg.eval_list,
                                           transform=dataset.ResizeNormalize(img_width=cfg.img_width,
                                                                             img_height=cfg.img_height))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
                                              num_workers=int(cfg.num_workers))

    # create crnn/seq2seq/attention network
    encoder = crnn.Encoder(channel_size=3, hidden_size=cfg.hidden_size)
    # for prediction of an indefinite long sequence
    decoder = crnn.Decoder(hidden_size=cfg.hidden_size, output_size=cfg.num_classes, dropout_p=0.1,
                           max_length=cfg.max_width)
    print(encoder)
    print(decoder)
    encoder.apply(utils.weights_init)
    decoder.apply(utils.weights_init)

    # create input tensor
    image = torch.FloatTensor(cfg.batch_size, 3, cfg.img_height, cfg.img_width)
    text = torch.LongTensor(cfg.batch_size)

    criterion = torch.nn.NLLLoss()

    assert torch.cuda.is_available(), "Please run \'train.py\' script on nvidia cuda devices."
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

    for epoch in range(cfg.num_epochs):
        # train crnn
        train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=cfg.teaching_forcing_prob, cfg=cfg, epoch=epoch)

        # do evaluation after training
        evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    main()


