import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
import parser
import models
import data_c
import numpy as np
from torchsummary import summary
from sklearn.metrics import accuracy_score
import itertools


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def evaluate(model, classifier, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):

            if torch.cuda.is_available():
                imgs = imgs.cuda()

            pred = classifier(model(imgs))

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)


if __name__ == '__main__':

    args = parser.arg_parse()

    source_data = 'svhn'
    target_data = 'mnistm'

    mnistm_dataset = data_c.Mnist(args, mode='train')
    svhn_dataset = data_c.Svhn(args, mode='train')

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    mnistm_loader = torch.utils.data.DataLoader(mnistm_dataset,
                                                batch_size=args.train_batch,
                                                num_workers=args.workers,
                                                shuffle=True)

    svhn_loader = torch.utils.data.DataLoader(svhn_dataset,
                                              batch_size=args.train_batch,
                                              num_workers=args.workers,
                                              shuffle=True)

    '''define source and target'''
    if source_data == 'mnistm':
        source = mnistm_loader
        target = svhn_loader
    else:
        source = svhn_loader
        target = mnistm_loader

    '''create directory to save trained model and other info'''
    paths = ["mnistm-svhn", "svhn-mnistm"]
    for path in paths:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' 1. Training phase '''
    print('===> prepare model for first training phase...')
    model = models.Adda_net(args)
    classifier = models.Adda_classifier(args)

    if torch.cuda.is_available():
        print("setze model auf Cuda")
        model.cuda()  # load model to
        classifier.cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0

    model.train()
    classifier.train()

    for epoch in range(1, args.epoch + 1):
        for idx, (img, cls) in enumerate(source):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(source))
            iters += 1

            ''' Move data to cuda '''
            if torch.cuda.is_available():
                img, cls = img.cuda(), cls.cuda()

            pred = classifier(model(img))

            loss = criterion(pred, cls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch%args.val_epoch == 0:

            ''' evaluate the model '''
            acc = evaluate(model, classifier, source)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                save_model(classifier, os.path.join(args.save_dir, 'classifier_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
        save_model(classifier, os.path.join(args.save_dir, 'classifier_{}.pth.tar'.format(epoch)))

    # ''' 2. Training phase '''
    # print('===> prepare model for second training phase...')
    #
    # source_model = models.Adda_net(args)
    # source_classifier = models.Adda_classifier(args)
    # target_model = models.Adda_net(args)
    # discriminator = models.Adda_discriminator(args)
    #
    # if torch.cuda.is_available():
    #     checkpoint = torch.load('./model_adda/{}_{}_model_best.pth.tar'.format(source_data, target_data))
    #     checkpoint_cls = torch.load('./model_adda/{}_{}_classifier_best.pth.tar'.format(source_data, target_data))
    # else:
    #     checkpoint = torch.load('./model_adda/{}_{}_model_best.pth.tar'.format(source_data, target_data),
    #                             map_location=torch.device('cpu'))
    #     checkpoint_cls = torch.load('./model_adda/{}_{}_classifier_best.pth.tar'.format(source_data, target_data),
    #                                 map_location=torch.device('cpu'))
    #
    # source_model.load_state_dict(checkpoint)
    # target_model.load_state_dict(checkpoint)
    #
    # source_classifier.load_state_dict(checkpoint_cls)
    #
    # for param in source_model.parameters():
    #     param.requires_grad = False
    #
    # source_model.eval()
    # source_classifier.eval()
    # target_model.train()
    # discriminator.train()
    #
    # if torch.cuda.is_available():
    #     source_model.cuda()
    #     source_classifier.cuda()
    #     target_model.cuda()
    #     discriminator.cuda()
    #
    # ''' define loss '''
    # criterion = nn.CrossEntropyLoss()
    #
    # ''' setup optimizer for target CNN and discriminator'''
    # optimizer_target_CNN = torch.optim.Adam(target_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    #
    # ''' setup tensorboard '''
    # writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))
    #
    # steps = min(len(source), len(target))
    # iters = 0
    # best_acc = 0
    #
    # for epoch in range(1, args.epoch + 1):
    #     total_accuracy_discriminator = 0
    #
    #     for idx, ((source_img, _),(target_img, _)) in enumerate(zip(source, target)):
    #         train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(source))
    #         iters += 1
    #
    #         if torch.cuda.is_available():
    #             source_img = source_img.cuda()
    #             target_img = target_img.cuda()
    #
    #         ''' train discriminator '''
    #         #print('===> train discriminator...')
    #         optimizer_discriminator.zero_grad()
    #
    #         pred_source = source_model(source_img)
    #         pred_target = target_model(target_img)
    #
    #         pred = torch.cat((pred_source, pred_target), 0).detach()
    #         pred_concat = discriminator(pred)
    #
    #         ''' labels to the domains '''
    #         source_lbl = torch.ones(pred_source.size(0)).long()
    #         target_lbl = torch.zeros(pred_target.size(0)).long()
    #         if torch.cuda.is_available():
    #             source_lbl = source_lbl.cuda()
    #             target_lbl = target_lbl.cuda()
    #         labels = torch.cat((source_lbl, target_lbl), 0)
    #
    #         loss_discriminator = criterion(pred_concat, labels)
    #         loss_discriminator.backward()
    #         optimizer_discriminator.step()
    #
    #         with torch.no_grad():
    #             predict = torch.max(pred_concat, 1)[1]
    #             acc = np.mean((predict == labels).cpu().numpy())
    #             total_accuracy_discriminator += acc
    #
    #         ''' write out information to tensorboard '''
    #         #writer.add_scalar('loss discriminator', loss_discriminator.data.cpu().numpy(), iters)
    #         train_info += ' loss discriminator: {:.4f}'.format(loss_discriminator.data.cpu().numpy())
    #
    #         #print(train_info)
    #
    #         ''' train target CNN '''
    #         #print('===> train target CNN...')
    #         optimizer_target_CNN.zero_grad()
    #         optimizer_discriminator.zero_grad()
    #
    #         pred_target = target_model(target_img)
    #         pred = discriminator(pred_target)
    #
    #         ''' Make flipped label '''
    #         # target_cls = source_cls
    #         # if torch.cuda.is_available():
    #         #     target_cls = target_cls.cuda()
    #         target_label = torch.ones(pred_target.size(0)).long()
    #         if torch.cuda.is_available():
    #             target_label = target_label.cuda()
    #
    #         loss_target_cnn = criterion(pred, target_label)
    #         loss_target_cnn.backward()
    #         optimizer_target_CNN.step()
    #
    #         ''' write out information to tensorboard '''
    #         writer.add_scalar('loss target CNN', loss_target_cnn.data.cpu().numpy(), iters)
    #         train_info += ' loss target CNN: {:.4f}'.format(loss_target_cnn.data.cpu().numpy())
    #
    #         print(train_info)
    #
    #     if epoch%args.val_epoch == 0:
    #         ''' evaluate the model '''
    #         helper = iters/idx
    #         acc_disc = total_accuracy_discriminator/helper
    #         acc = evaluate(target_model, source_classifier, target)
    #         writer.add_scalar('val_acc', acc, iters)
    #         print('')
    #         print('Epoch: [{}] ACC (of target CNN):{} ACC (of Discriminator):{}'.format(epoch, acc, acc_disc))
    #         print('')
    #
    #         ''' save best model '''
    #         if acc > best_acc:
    #             save_model(target_model, os.path.join(args.save_dir, 'target_model_best.pth.tar'))
    #             save_model(discriminator, os.path.join(args.save_dir, 'discriminator_best.pth.tar'))
    #             best_acc = acc
    #
    #     ''' save model '''
    #     save_model(target_model, os.path.join(args.save_dir, 'target_model_{}.pth.tar'.format(epoch)))
    #     save_model(discriminator, os.path.join(args.save_dir, 'discriminator_{}.pth.tar'.format(epoch)))