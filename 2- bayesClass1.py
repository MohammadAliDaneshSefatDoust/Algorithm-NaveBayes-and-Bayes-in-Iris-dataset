import numpy as np
class bayesClass1:
    def __init__(self, className, x1, x2, x3, x4):
        self.className = className
        
        data = np.array([x1, x2, x3, x4])

        self.cov = np.cov(data, bias=True)
        self.mean = [np.mean(x1), np.mean(x2), np.mean(x3), np.mean(x4)]
        
    def calculatLikelihood(self, inputData):
        x = inputData - self.mean
        likelihood = -1 * np.dot(np.dot(x, np.linalg.inv(self.cov)), x.T)
        return likelihood
    
    def makeDiagonal(self):
        self.cov = np.diag(np.diag(self.cov))
        