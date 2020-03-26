wget https://www.dropbox.com/s/sfvwirn2c78qpsu/svhn_mnistm_dann_model.pth.tar?dl=1
wget https://www.dropbox.com/s/t5psye7ogmpihzr/mnistm_svhn_dann_model.pth.tar?dl=1
RESUME1='svhn_mnistm_dann_model.pth.tar?dl=1'
RESUME2='mnistm_svhn_dann_model.pth.tar?dl=1'
python3 test_dann.py --resume1 $RESUME1 --resume2 $RESUME2 --data_dir $1 --target_data $2 --csv_dir $3
