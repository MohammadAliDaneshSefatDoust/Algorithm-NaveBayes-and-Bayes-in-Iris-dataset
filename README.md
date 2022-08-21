# NaveBayes-and-Bayes-in-Iris-dataset
________________________________________________________________
NaveBayes and Bayes in Iris dataset "https://archive.ics.uci.edu/ml/datasets/iris"

In this project, using Iris data (from the UCI collection), we implement Bayes and Naïve Bayes algorithm in Python.
________________________________________________________________
https://archive.ics.uci.edu/ml/datasets/iris
________________________________________________________________
The data contains 150 data with 4 dimensions (4 features) in 3 classes, each class has 50 data.

1- The data has been randomly divided into two parts: training and testing. In this way, we have set aside 70% of the data from each class for training and 30% for testing.We have constructed a Gaussian PDF for multidimensional variables using training data. We produce the Bayesian classification by considering the uniform (equal) background distribution for all classes. (ML estimation). We repeat this random selection and training twice. (The code of this part is in part-A)

2- We repeat the previous part with 4-fold cross validation. It means that every time 75% of the data participate in training and 25% in testing. (The code for this part is in Part-B.)

3- Now we repeat the previous two parts with Gaussian distribution with diagonal covariance (the code of this part is in part-C)

4- Finally, we implemented Gaussian naïve Bayes algorithm on the training data of part B
