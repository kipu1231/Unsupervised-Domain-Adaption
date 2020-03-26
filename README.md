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
<img src="/Results/acc_adda.png" width="500" alt="ADDA accuracy"/>
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
The models can be trained using the following commands. `train_source` and `train` can be utilized to train DANN. Target and source domains need to be adjusted beforehand. `train_adda` can be used for both, first and second round of training adda.

    bash train_source.sh digits
    bash train.sh digits
    bash train_adda.sh digits

### Evaluation
To test the trained models, the provided script can be run by using the following command. A csv with the repective predictions is saved to the path indicated in `$3`.

    bash test_dann.sh $1 $2 $3
    bash test_adda.sh $1 $2 $3

-   `$1` is the directory of testing images in the **target** domain (e.g. `digits/svhn/test`).
-   `$2` is a string that indicates the name of the target domain, which will be either `mnistm` or `svhn`. 
	- Note: Run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `svhn`, make the prediction using "mnistm→svhn" model, **NOT** "svhn→mnistm→" model.
-   `$3` is the path to the output prediction file (e.g. `digits/svhn/test_pred.csv`).

To evaluate your the models, the evaluation script can be run by using the following command.

    python3 eval.py $1 $2

 - `$1` is the path to the predicted results (e.g. `digits/svhn/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `/digits/svhn/test.csv`)

### tSNE
All above visualizations are made with tSNE to visualize the output of the feature extractor.

