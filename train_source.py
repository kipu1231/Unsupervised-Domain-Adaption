import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
import parser
import models
import data_c
import numpy as np
from torchsummary import summary


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == '__main__':

    args = parser.arg_parse()

    mnistm_dataset = data_c.Mnist(args, mode='train')
    svhn_dataset = data_c.Svhn(args, mode='train')

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    mnistm_loader = torch.utils.data.DataLoader(mnistm_dataset,
                                                batch_size=args.train_batch,
                                                num_workers=args.workers,
                                                shuffle=False)

    svhn_loader = torch.utils.data.DataLoader(svhn_dataset,
                                              batch_size=args.train_batch,
                                              num_workers=args.workers,
                                              shuffle=False)

    '''define source and target'''
    source = [mnistm_loader, svhn_loader]
    target = [mnistm_loader, svhn_loader]

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

    ''' load model '''
    print('===> prepare model ...')
    model = models.DannSource(args)
    # checkpoint = torch.load('./log/model_best_adv.pth.tar')
    # model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        print("setze model auf Cuda")
        model.cuda()  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0

    ''' Loop through different source target combinations '''
    for i in range(len(source)):

        steps = min(len(source[i]), len(target[i]))

        weight_reset(model)
        model.train()

        for epoch in range(1, args.epoch + 1):
            train_loss = 0.0

            source_data = iter(source[i])
            #target_data = iter(target[i])

            for step in range(steps):
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, steps)
                iters += 1

                #p = float(step + epoch * steps) / (args.epoch * steps)
                #lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

                ''' SOURCE DATA '''
                ''' Move source data to cuda '''
                imgs, cls = next(iter(source_data))

                if torch.cuda.is_available():
                    imgs, cls = imgs.cuda(), cls.cuda()

                ''' Set domain of source data to 0 '''
                imgsize = imgs.size(0)
                domains = torch.zeros(imgsize).long()

                if torch.cuda.is_available():
                    domains = domains.cuda()

                ''' Calculate loss '''
                source_class_out = model(imgs)
                source_class_loss = criterion(source_class_out, cls)

                ''' Overall loss and  backpropagation'''
                #loss = source_class_loss + source_domain_loss + target_domain_loss
                loss = source_class_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                ''' write out information to tensorboard '''
                writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

                print(train_info)

            print("\n\nSource-Target: {}".format(paths[i]))
            print("Finish epoch {}\n".format(epoch))
            save_model(model, os.path.join(args.save_dir, paths[i] + '_model_{}.pth.tar'.format(epoch)))
