# Unsupervised-Domain-Adaption
Implementation of two models for unsupervised domain adaption aiming to realizize  transfer learning between the mnist-m and svhn digit data set.

### Results
To compare the results, DANN is additionally trained on source and target data only which serves as lower and upper bound. In the case of SVHN → MNIST-M, the accuracy lays above the one of DANN due to the fact that the training process of DANN is not very stable. An overview of the training accuracy is given in the following table.

<p align="center">
<img src="/Results/acc_dann.png" width="500" alt="DANN accuracy"/>
</p>

In order to visualize the output of the feature extractor (latent space), the test data was mapped to the 2D-space using t-SNE. The results are visualized in the following for (a) the different digit classes and (b) the different domains.

<p align="center">
<img src="/Results/DANN.png" width="500" alt="tSNE DANN"/>
</p>

In addition, ADDA was implemented to improve the results of DANN. The results of the models are displayed in the following table.

<p align="center">
<img src="/Results/acc_adda.pn" width="500" alt="ADDA accuracy"/>
</p>

The test data is also mapped to the 2D-space by applying t-SNE. In the following, the (a) different digit classes and the (b) domains are visualized for each of the two improved UDA models.

<p align="center">
<img src="/Results/ADDA.png" width="500" alt="tSNE ADDA"/>
</p>

# Usage

### Dataset
In order to download the used dataset, a shell script is provided and can be used by the following command.

    bash ./get_dataset.sh
    
The shell script will automatically download the dataset and store the data in a folder called `digits`. 

### Packages
The project is done with python3.6. For used packages, please refer to the requirments.txt for more details. All packages can be installed with the following command.

    pip3 install -r requirements.txt
    
### Training
The models can be trained using the following command. To distinguish the training of GAN and ACGAN, the two following commands can be used.

    python3 train_gan.py
    python3 train_acgan.py

### Testing & Visualization
To test the trained models, the provided script can be run by using the following command. Two plots will be generated and saved in predefined folder as output. 

    bash test_models.sh $1

-   `$1` is the folder to which the output `fig_gan.jpg` and `fig_acgan.jpg` is saved.
