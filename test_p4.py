import torch
from PIL import Image
import parser
import models
import data_c
import numpy as np
import os
import csv


if __name__ == '__main__':
    args = parser.arg_parse()

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    '''Distinguish different training patterns'''
    if "mnistm" == args.target_data:
        print("jumps in here")
        dataset = data_c.Mnist(args, mode='test')
        source = 'svhn'
        target = 'mnistm'
    elif "svhn" == args.target_data:
        dataset = data_c.Svhn(args, mode='test')
        print("jumps in here")
        source = 'mnistm'
        target = 'svhn'

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=False)

    ''' prepare mode '''
    #model = models.Dann(args)
    model_net = models.Adda_net(args)
    model_classifier = models.Adda_classifier(args)

    if torch.cuda.is_available():
        model_net.cuda()
        model_classifier.cuda()

    with torch.no_grad():
        ''' resume save model '''
        if "mnistm" == args.target_data:
            checkpoint_cnn = torch.load(args.resume1)
            checkpoint_cls = torch.load(args.resume2)
        elif "svhn" == args.target_data:
            checkpoint_cnn = torch.load(args.resume3)
            checkpoint_cls = torch.load(args.resume4)

        model_net.load_state_dict(checkpoint_cnn)
        model_classifier.load_state_dict(checkpoint_cls)

        model_net.eval()
        model_classifier.eval()

        with open(args.csv_dir, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["image_name", "label"])

            for _, data in enumerate(test_loader):
                inputs, file_name = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                # outputs, _ = model(inputs, 0)
                outputs = model_classifier(model_net(inputs))

                predict = torch.max(outputs, 1)[1].cpu().numpy()

                for i in range(len(file_name)):
                    img_name = os.path.basename(file_name[i])
                    writer.writerow([img_name, predict[i]])






