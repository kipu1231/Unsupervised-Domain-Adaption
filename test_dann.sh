wget https://www.dropbox.com/s/0s6yxbij9r3swwj/svhn_mnistm_dann_model.pth.tar?dl=1
wget https://www.dropbox.com/s/bir21qbnnuyu228/mnistm_svhn_dann_model.pth.tar?dl=1
RESUME1='svhn_mnistm_dann_model.pth.tar?dl=1'
RESUME2='mnistm_svhn_dann_model.pth.tar?dl=1'
python3 test_dann.py --resume1 $RESUME1 --resume2 $RESUME2 --data_dir $1 --target_data $2 --csv_dir $3
