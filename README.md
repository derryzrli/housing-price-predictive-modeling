# Project - Ames Housing Price Prediction


### Problem Statement

About 4 years ago, Zillow decided to make a major commitment to “iBuying,” the computer-automated purchasing of homes to flip using a pricing algorithm, with the launch of Zillow Offers (“ZO”). This led to a series of expected financial loss caused by high error rates from ZO, creating turmoils in the housing market and within Zillow. 
__This project aims to build a model with a wide array of features for the purpose of prediction, in order to help businesses assess housing prices more robustly, and avoid scenerios similar to the one endured by Zillow.__
We define that a robust model good for business use is one that is better than the simple model and accounts for over 92% of the variation in Sale Price. 

### Dataset Used
* [`train.csv`](datasets/train.csv): Training data set contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010, with Sale Price
* [`test.csv`](datasets/test.csv): Testing data set contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010, without Sale Price

### Data Dictionary
* [Data Description](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt) for the Ames Iowa Housing Dataset

### The Analysis + Modeling Process

- Data Cleaning + EDA
- Ordinal Encoding on selected ordinal features 
- Log Transformation for skewed features
- One Hot Encoding on categorical features
- Grid Search over pipeline of (StandardScalar() $\rightarrow$ KNN-Imputer $\rightarrow$ LASSO) twice to tune $k$ and $\alpha$
- Used KNN-Imputer (with best $k$) and LASSO (with best $\alpha$) to impute null values and perform feature selection
- Outlier identification and removal if |standardized residuals| > 3
- Fitted and evaluated LASSO, Ridge, ElasticNet, MLR

Note: a graphic illustration of the process is available in the [presentation pdf](GA_DSI_Project_2.pdf) on slides 21-23.


### Conclusion and Future Work

As our robust model performance shows, 94.93% of the variation in Sale Price in the training set and 91.13% of the variation in Sale Price in the test set can be explained by the model. This statisfies our defination of a robust model which accounts for over 92% of the target variation. This model is fairly good at predictive tasks, with an $R^2$ 0.23 larger than that of our base model, meaning the robust model accounts for 23% more variation in Sale Price than the base model.

As for future work, since we still see slight overfitting in our robust model, the use of clustering to group and lower OHE feature counts may help us reduce model complexity in hope for better bias-variance tradeoff. In addition, more experimentation as to how we perform the train-test-split (i.e. at what ratio) may lead to model performance improvement, as it is a critical 'hyperparametre' which we did not tune using GridSearch. 