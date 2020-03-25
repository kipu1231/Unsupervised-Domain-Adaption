wget https://www.dropbox.com/s/7nrv9dvcgvk4gti/svhn_mnistm_uda_target_model_best.pth.tar?dl=1
wget https://www.dropbox.com/s/99ief378nxrc196/svhn_mnistm_uda_classifier_best.pth.tar?dl=1
wget https://www.dropbox.com/s/bkbnh6cx4nuqc4j/mnistm_svhn_uda_target_model.pth.tar?dl=1
wget https://www.dropbox.com/s/y94jj4rju77wct5/mnistm_svhn_uda_classifier_best.pth.tar?dl=1
RESUME1='svhn_mnistm_uda_target_model_best.pth.tar?dl=1'
RESUME2='svhn_mnistm_uda_classifier_best.pth.tar?dl=1'
RESUME3='mnistm_svhn_uda_target_model.pth.tar?dl=1'
RESUME4='mnistm_svhn_uda_classifier_best.pth.tar?dl=1'
python3 test_p4.py --resume1 $RESUME1 --resume2 $RESUME2 --resume3 $RESUME3 --resume4 $RESUME4 --data_dir $1 --target_data $2 --csv_dir $3
