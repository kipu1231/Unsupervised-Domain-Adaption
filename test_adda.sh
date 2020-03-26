wget https://www.dropbox.com/s/hl6ath1bvdwba27/svhn_mnistm_uda_target_model_best.pth.tar?dl=1
wget https://www.dropbox.com/s/sifc6qzxszczfhf/svhn_mnistm_uda_classifier_best.pth.tar?dl=1
wget https://www.dropbox.com/s/prin5csfyt67vke/mnistm_svhn_uda_target_model.pth.tar?dl=1
wget https://www.dropbox.com/s/sqmgfby082kfu8m/mnistm_svhn_uda_classifier_best.pth.tar?dl=1
RESUME1='svhn_mnistm_uda_target_model_best.pth.tar?dl=1'
RESUME2='svhn_mnistm_uda_classifier_best.pth.tar?dl=1'
RESUME3='mnistm_svhn_uda_target_model.pth.tar?dl=1'
RESUME4='mnistm_svhn_uda_classifier_best.pth.tar?dl=1'
python3 test_adda.py --resume1 $RESUME1 --resume2 $RESUME2 --resume3 $RESUME3 --resume4 $RESUME4 --data_dir $1 --target_data $2 --csv_dir $3
