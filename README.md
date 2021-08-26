# Master Thesis: Popularity Bias and Fairness in Recommendation

## Acknowledgment
- The implementation of ALS is modified from (https://github.com/tushushu/imylu/tree/master/imylu/recommend)
- The implementation of VAE-CF is modified from (https://github.com/dawenl/vaecf)
- The implementation of Long-tail GAN is modified from (https://github.com/CrowdDynamicsLab/NCF-GAN)

## Final Dissertation
The final version of dissertation can be found [here](Final_Dissertation.pdf), which all the explanation and figures are presented to show own work.

## Experiments
### Setup Environment
First of all, install all the required packages using 
```
conda install numpy=1.16.0 tensorflow-gpu=1.14.0 scikit-learn=0.23.1 seaborn=0.11.1 matplotlib=3.3.2 pandas=1.1.3 bottleneck=1.3.2 psutil=5.8.0 scipy=1.5.2 configparser=5.0.2
```
### Dataset Processing
Before running the experiments, please check if the dataset exists in the first level directory [raw_data](raw_data/). As long as the dataset exist, the [DataConvert_dat_to_csv.ipynb](DataConvert_dat_to_csv.ipynb) can be used for dataset converting.
Then, [process_rawdata.ipynb](process_rawdata.ipynb) is used to split dataset(with holdout ratings) by first approach and generated different group of users. The second approach is implemented together with [VAE_CF(holdout users).ipynb](VAE_CF/VAE_CF(holdout_users).ipynb)
### ALS
In order to run the experiments for ALS-MF, please use the [ALS.ipynb](ALS/ALS.ipynb) file under the first level directory ALS. Test section is alse include in this file.
### VAE-CF
In order to run the experiments for VAE-CF, please use the [VAE_CF(holdout_ratings).ipynb](VAE_CF/VAE_CF(holdout_ratings).ipynb) and [VAE_CF(holdout_users).ipynb](VAE_CF/VAE_CF(holdout_users).ipynb). Test section is alse include in this file.
### Long-tail GAN
In order to run the experiments for long-tail GAN, please use the following commend lines to run the experiments.
For training:
```
	cd long-tail_GAN/Codes
	python train.py ../Dataset/ml_1m_holdout_user
```
For testing:
```
	cd long-tail_GAN/Codes
	python test.py ../Dataset/ml_1m_holdout_user
```
