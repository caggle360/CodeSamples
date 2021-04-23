# Courtney Simmons
# November 4, 2020

import numpy as np
# Import for dataset (breast cancer dataset)
from sklearn.datasets import load_breast_cancer 
# Import for dataset (boston housing dataset)
from sklearn.datasets import load_boston
# import to get train/test data
from sklearn.model_selection import train_test_split
# import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# import ridge regression scikit-learn module
from sklearn.linear_model import Ridge 


# Create a decision tree that splits 4 ways (splits on 2 variables at the same time)
class QuarternaryDecisionTree:

    def __init__(self, max_depth = 2, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.y_pred = None
        self.y_prob = None

    # Calculate Entropy
    def Entropy(self, p):
        if p == 0 or p == 1:
            return 0
        else:
            return -(p * np.log2(p)) - ((1-p) * np.log2(1-p))

    # Get and resutls size of partition and p
    def findP(self, split):
        # Find size of partition
        n = split.size

        if n == 0: # since we cannot divide by 0
            p = 0 
        else:
            p = np.count_nonzero(split == 1)/n
        return n, p

    # Calculate Information Gain
    def infoGain(self, y, y_split1, y_split2, y_split3, y_split4):

        # Get total size of y (N)
        N = y.size
        # Calculate parent p
        Parent_p = np.count_nonzero(y == 1)/N
        # Calculate parent entropy
        parentEntropy = self.Entropy(Parent_p)

        # Find n and p for these partitions
        n1, split1_p = self.findP(y_split1)
        n2, split2_p = self.findP(y_split2)
        n3, split3_p = self.findP(y_split3)
        n4, split4_p = self.findP(y_split4)

        # Get entropy for each partition
        split1_entropy = self.Entropy(split1_p)
        split2_entropy = self.Entropy(split2_p)
        split3_entropy = self.Entropy(split3_p)
        split4_entropy = self.Entropy(split4_p)

        # Calculate Information gain
        IG = parentEntropy - ((n1/N)*split1_entropy + (n2/N) *split2_entropy + (n3/N)* split3_entropy + (n4/N) * split4_entropy)

        return IG

    def getBestSplit(self, X, y):
        
        n, m = X.shape

        # Initialize best splitting variables and values, and max Info Gain
        x1_best = -1
        x2_best = -1
        s1_value = -1
        s2_value = -1
        IG_max = -np.inf
        # Iterate over first feature
        for feature1 in range(m):
            # Iterate over a second feature
            for feature2 in range(feature1 + 1, m):
                # iterate over all unique values in feature 1
                for value1 in np.unique(X[:,feature1]):
                    # iterate over all unique values in feature 2
                    for value2 in np.unique(X[:, feature2]):
                        # Get the indices for the split for this feature and value
                        split1_ind = (X[:, feature1] <= value1) & (X[:, feature2] <= value2) 
                        split2_ind = (X[:, feature1] <= value1) & (X[:, feature2] > value2)
                        split3_ind = (X[:, feature1] > value1) & (X[:, feature2] <= value2)
                        split4_ind = (X[:, feature1] > value1) & (X[:, feature2] > value2) 

                        # Split the data using the indices
                        X_split1, y_split1 = X[split1_ind,:], y[split1_ind]
                        X_split2, y_split2 = X[split2_ind,:], y[split2_ind]
                        X_split3, y_split3 = X[split3_ind,:], y[split3_ind]
                        X_split4, y_split4 = X[split4_ind,:], y[split4_ind]

                        # Get information gain
                        IG = self.infoGain(y, y_split1, y_split2, y_split3, y_split4)

                        # If information gain for this set of splits is larger than the current largest info gain,
                        # update best splitting features and values for this node.
                        if IG > IG_max:
                            IG_max = IG
                            x1_best = feature1
                            x2_best = feature2
                            s1_value = value1
                            s2_value = value2

        return x1_best, x2_best, s1_value, s2_value
    
    def grow(self, X, y, depth):
        n,m = X.shape
        # Stopping rules for stopping growing the tree
        if (n < self.min_samples_split or depth == 0 or np.mean(y) == 1 or np.mean(y) == 0):
            return np.mean(y)

        # Get the best splitting variables and values and return
        bestVar1, bestVar2, bestVal1, bestVal2 = self.getBestSplit(X, y)

        # If the splitting variables were never updated from initialization
        if bestVar1 == -1 or bestVar2 == -1:
            return np.mean(y) 

        # Get indexes for split using our best variable and value split we found above
        split1_ind = (X[:, bestVar1] <= bestVal1) & (X[:, bestVar2] <= bestVal2) 
        split2_ind = (X[:, bestVar1] <= bestVal1) & (X[:, bestVar2] > bestVal2)
        split3_ind = (X[:, bestVar1] > bestVal1) & (X[:, bestVar2] <= bestVal2)
        split4_ind = (X[:, bestVar1] > bestVal1) & (X[:, bestVar2] > bestVal2) 

        # Split the data using the indices
        X_split1, y_split1 = X[split1_ind,:], y[split1_ind]
        X_split2, y_split2 = X[split2_ind,:], y[split2_ind]
        X_split3, y_split3 = X[split3_ind,:], y[split3_ind]
        X_split4, y_split4 = X[split4_ind,:], y[split4_ind]

        # Recursively grow the tree and return the splitting variables and values at each split
        return {"bestVar1": bestVar1, 
        "bestVar2": bestVar2,
        "bestVal1": bestVal1,
        "bestVal2": bestVal2,
        "split1": self.grow(X_split1, y_split1, depth - 1),
        "split2": self.grow(X_split2, y_split2, depth - 1),
        "split3": self.grow(X_split3, y_split3, depth - 1),
        "split4": self.grow(X_split4, y_split4, depth - 1)
        }

    # fits the quarternary decision tree and stores the tree inside
    def fit(self, X, y):
        self.tree = self.grow(X, y, self.max_depth)
        return self.tree
        

    # outputs the predictions for X (testing data)
    def predict(self, X):
         
        n, m = X.shape

        # Initialize predictions
        predictions = np.zeros(n)
        probability = np.zeros(n)
        
        # Loop through each row (observation)
        for i in range(n):
            row = X[i, :]
            probability[i], predictions[i] = self.getPrediction(row)

        self.y_prob = probability
        self.y_pred = predictions

        return self.y_prob, self.y_pred

    def getPrediction(self, row):
        tree_level = self.tree

        # While we are still splitting 
        while isinstance(tree_level, dict):
            if (row[tree_level['bestVar1']] <= tree_level['bestVal1']) & (row[tree_level['bestVar2']] <= tree_level['bestVal2']):
                tree_level = tree_level['split1'] 
            elif (row[tree_level['bestVar1']] <= tree_level['bestVal1']) & (row[tree_level['bestVar2']] > tree_level['bestVal2']):
                tree_level = tree_level['split2'] 
            elif (row[tree_level['bestVar1']] > tree_level['bestVal1']) & (row[tree_level['bestVar2']] <= tree_level['bestVal2']):
                tree_level = tree_level['split3'] 
            elif (row[tree_level['bestVar1']] > tree_level['bestVal1']) & (row[tree_level['bestVar2']] > tree_level['bestVal2']):
                tree_level = tree_level['split4'] 
        else:
            return tree_level, np.round(tree_level)


def test_QuarternaryDecisionTree():

    # Import Class
    model = QuarternaryDecisionTree()

    # Load breast cancer data
    data = load_breast_cancer()
    print("The feature names of the breast cancer data are", str(data.feature_names))
    X = data.data
    y = data.target
    n, m = X.shape

    # Subset the data to make it faster to run (I'm only using the first 5 variables)
    X = X[:, 0:4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22)

    # Using QuarternaryDecisionTree, will grow 2 levels
    tree = model.fit(X_train, y_train)
    print("The following is the printed tree", str(tree))

    predict = model.predict(X_test)
    y_prob  = predict[0]
    y_preds = predict[1]

    # Print AUROC 
    AUROC = roc_auc_score(y_test, y_prob)
    print("The AUROC is", str(AUROC))
    # 0.897

    # Print accuracy
    Accuracy = accuracy_score(y_test, y_preds)
    print("The Accuracy is", str(Accuracy))
    # 0.865

    # Print Precision
    Precision = precision_score(y_test, y_preds)
    print("The Precision is", str(Precision))
    # 0.889

# At each node, fit a Ridge linear regression instead of mean value
class DaRDecisionTree:

    def __init__(self, min_samples_split = 2):
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.y_pred = None

    # Get the variance at a certain split
    def getVariance(self, y, y_left, y_right):

        n_left = y_left.size
        n_right = y_right.size

        if n_left == 0:
            left_var = 0 # Because we cannot divide by 0
        else: 
            left_var = np.var(y_left)

        if n_right == 0:
            right_var = 0
        else:
            right_var = np.var(y_right)

        return (n_left*left_var) + (n_right*right_var)

    # Gets the best split at a node based on minimum variance
    def getBestSplit(self, X, y):
        
        n, m = X.shape

        # Initialize best splitting variables and values, and min variance
        x1_best = -1
        s1_value = -1
        var_min = np.inf
        # Iterate over features
        for feature in range(m):
            # Iterate over all unique values in the feature
            for value in np.unique(X[:,feature]):
                # Get the indices for the split for this feature and value
                left_ind = (X[:, feature] <= value)  
                right_ind = (X[:, feature] > value)

                # Split the data using the indices
                X_left, y_left = X[left_ind,:], y[left_ind]
                X_right, y_right = X[right_ind,:], y[right_ind]

                # Calculate variance for split
                variance = self.getVariance(y, y_left, y_right)

                # If the variance at the split is less than the minimum variance we've seen so
                # far, update the minimum variance, best splitting feature and value
                if variance < var_min:
                    var_min = variance
                    x1_best = feature
                    s1_value = value

        return x1_best, s1_value
    
    def grow(self, X, y, depth):
        n,m = X.shape
        # Stopping criteria. If any of these are true, fit a ridge regression
        if (n < self.min_samples_split or depth == 0 or np.var(y) == 0):
            RidgeReg = Ridge(normalize = True, random_state = 4).fit(X,y)
            return RidgeReg

        # Get the best splitting variable and value and return
        bestVar, bestVal = self.getBestSplit(X, y)

        # If best splitting variable was never replaced from initialized value
        if bestVar == -1:
            RidgeReg = Ridge(normalize = True, random_state = 4).fit(X,y)
            return RidgeReg

        # Get indexes for split using our best variable and value split we found above
        left_ind = (X[:, bestVar] <= bestVal)  
        right_ind = (X[:, bestVar] > bestVal)

        # Split the data using the indices
        X_left, y_left = X[left_ind,:], y[left_ind]
        X_right, y_right = X[right_ind,:], y[right_ind]

        # Grow tree recursively
        return {"bestVar": bestVar, 
        "bestVal": bestVal,
        "left": self.grow(X_left, y_left, depth - 1),
        "right": self.grow(X_right, y_right, depth - 1)
        }

    # fits the quarternary decision tree and stores the tree inside
    def fit(self, X, y, max_depth):
        self.tree = self.grow(X, y, max_depth)
        return self.tree

    # outputs the predictions for X (testing data)
    def predict(self, X):
         
        n, m = X.shape

        # Initialize predictions
        predictions = np.zeros(n)
        
        # Loop through each row (observation)
        for i in range(n):
            row = X[i, :]
            predictions[i] = self.getPrediction(row)

        self.y_pred = predictions

        return self.y_pred

    def getPrediction(self, row):
        tree_level = self.tree

        # While we are still splitting
        while isinstance(tree_level, dict):
            if (row[tree_level['bestVar']] <= tree_level['bestVal']):
                tree_level = tree_level['left'] 
            elif (row[tree_level['bestVar']] > tree_level['bestVal']):
                tree_level = tree_level['right'] 
        else:
            # This returns the prediction based on the Ridge regression model that was stored at each leaf
            return tree_level.predict(row.reshape(1, -1))


def test_DaRDecisionTree():

    # Import Class
    model2 = DaRDecisionTree()

    # Load breast cancer data
    data = load_boston()
    print("The feature names of the boston housing dataset are", str(data.feature_names))
    X = data.data
    y = data.target
    n, m = X.shape

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22)

    # Using DaRDecisionTree, the third argument is the maximum number of levels
    tree = model2.fit(X_train, y_train, 2)
    print("The following is the printed tree", str(tree))

    y_preds = model2.predict(X_test)

    mse = mean_squared_error(y_test, y_preds)
    print("The MSE is", str(mse)) 
    # 19.314

    rmse = np.sqrt(mse)
    print("The RMSE is", str(rmse))
    # 4.395

    # Get the R-squared
    r2 = r2_score(y_test, y_preds)
    print("The R-squared is", str(r2))
    # The R-squared is good
    # 0.797
 
if __name__ == "__main__":

    test_QuarternaryDecisionTree()
    test_DaRDecisionTree()