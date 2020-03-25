import torch
from PIL import Image
import parser
import models
import data_c
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    args = parser.arg_parse()

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    '''Distinguish different training patterns'''
    if "mnistm" == args.target_data:
        source_data = data_c.Svhn(args, mode='train', visualization=True)
        target_data = data_c.Mnist(args, mode='train', visualization=True)
        source = 'svhn'
        target = 'mnistm'
    elif "svhn" == args.target_data:
        source_data = data_c.Mnist(args, mode='train', visualization=True)
        target_data = data_c.Svhn(args, mode='train', visualization=True)
        source = 'mnistm'
        target = 'svhn'

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    source_loader = torch.utils.data.DataLoader(source_data,
                                                batch_size=args.test_batch,
                                                num_workers=args.workers,
                                                shuffle=False)

    target_loader = torch.utils.data.DataLoader(target_data,
                                                batch_size=args.test_batch,
                                                num_workers=args.workers,
                                                shuffle=False)

    tsne = TSNE(n_components=2, init="pca", random_state=0)

    ''' prepare tSNE '''
    tsne = TSNE(n_components=2, init="pca")
    x = np.array([]).reshape(0, 3200)
    y_cls = np.array([], dtype=np.int16).reshape(0, )
    y_dom = np.array([], dtype=np.int16).reshape(0, )

    # ''' prepare mode '''
    # model = models.Dann(args)
    # if torch.cuda.is_available():
    #     model.cuda()
    # checkpoint = torch.load("./model_dann/{}_{}_dann_model.pth.tar".format(source, target), map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint)
    # model.eval()

    # model = models.Adda_net(args)
    # #model_classifier = models.Adda_classifier(args)
    # #check_net = torch.load("svhn_mnistm_uda_target_model_best.pth", map_location=torch.device('cpu'))
    # #check_clas = torch.load("./model/{}-{}_model.pth.tar".format(source, target), map_location=torch.device('cpu'))
    # model.load_state_dict(check_net)
    # #model_classifier.load_state_dict(check_clas)
    # model.eval()
    # #model_classifier.eval()


    ''' run target and source data through feature extractor '''
    with torch.no_grad():
        steps = len(source_loader)
        model_src = models.Adda_net(args)
        checkpoint_src = torch.load("./model_adda/{}_{}_uda_source_model.pth.tar".format(source, target), map_location=torch.device('cpu'))
        model_src.load_state_dict(checkpoint_src)
        if torch.cuda.is_available():
            model_src.cuda()
        model_src.eval()

        model_tgt = models.Adda_net(args)
        checkpoint_tgt = torch.load("./model_adda/{}_{}_uda_target_model.pth.tar".format(source, target),
                                    map_location=torch.device('cpu'))
        model_tgt.load_state_dict(checkpoint_tgt)
        if torch.cuda.is_available():
            model_tgt.cuda()
        model_tgt.eval()

        for i, data in enumerate(source_loader):
            img, cls = data
            if torch.cuda.is_available():
                img = img.cuda()

            outputs = model_src.conv(img).contiguous().view(img.size(0), -1).cpu().numpy()
            cls = cls.numpy()

            x = np.vstack((x, outputs))
            y_cls = np.concatenate((y_cls, cls))
            y_dom = np.concatenate((y_dom, np.array([0 for _ in range(img.size(0))], dtype=np.int16)))

            print("Progress of Source data in steps: [{}/{}]".format(i, steps))

        print(x.shape)
        print(y_cls.shape)
        print(y_dom.shape)

        steps = len(target_loader)

        for i, data in enumerate(target_loader):
            img, cls = data
            if torch.cuda.is_available():
                img = img.cuda()

            outputs = model_tgt.conv(img).contiguous().view(img.size(0), -1).cpu().numpy()
            cls = cls.numpy()

            x = np.vstack((x, outputs))
            y_cls = np.concatenate((y_cls, cls))
            y_dom = np.concatenate((y_dom, np.array([1 for _ in range(img.size(0))], dtype=np.int16)))

            print("Progress of Target data in steps: [{}/{}]".format(i, steps))

        print(x.shape)
        print(y_cls.shape)
        print(y_dom.shape)

    ''' perform tSNE and get min, max and norm '''
    x_tsne = tsne.fit_transform(x)
    print("Data has the {} before tSNE and the following after tSNE {}".format(x.shape[-1], x_tsne.shape[-1]))
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)

    ''' plot results of tSNE '''
    colors = ['lightcoral', 'maroon', 'k', 'grey', 'orange', 'darkslategrey', 'lightskyblue', 'plum', 'yellow', 'plum']
    class_color = [colors[label] for label in y_cls]
    domain_color = [colors[label] for label in y_dom]

    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=1)
    plt.savefig("./adda_{}_{}_class.png".format(source, target))
    plt.title("(a) digit classes")
    plt.close("all")

    plt.figure(2, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=domain_color, s=1)
    plt.savefig("./adda_{}_{}_domain.png".format(source, target))
    plt.title("(b) domains")
    plt.close("all")