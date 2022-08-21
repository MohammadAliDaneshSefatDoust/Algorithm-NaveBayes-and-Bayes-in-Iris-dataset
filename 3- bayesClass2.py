import numpy as np
class bayesClass1:
    def __init__(self, className, x1, x2, x3, x4, countTotal):
        self.className = className
        
        self.mean = [np.mean(x1), np.mean(x2), np.mean(x3), np.mean(x4)]
        self.std = [np.std(x1), np.std(x2), np.std(x3), np.std(x4)]
        self.prob = len(x1) / countTotal
        
        
    def calculatLikelihood(self, inputData):
        result = np.log(self.prob)
        for i in range(len(self.mean)):
            result += np.log(self.gaussian(self.mean[i], self.std[i], inputData[i]))
        return result
    
    def gaussian(self, mu, std, var):
        return (1.0 / (std * np.power((2 * np.pi), 0.5))) * np.exp(-np.power(var - mu, 2.) / (2 * np.power(std, 2.)))

        