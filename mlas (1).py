import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")


X = df.drop(columns=['play','day'])
y = df['play']


def id3(X, y):
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if len(X.columns) == 0:
        return y.mode()[0]
    info_gain = []
    for feature in X.columns:
        entropy = 0
        for value in np.unique(X[feature]):
            sub_y = y[X[feature] == value]
            prob = len(sub_y) / len(y)
            entropy += prob * np.log2(prob)
        entropy = -entropy
        # calculate information gain
        info_gain.append((feature, entropy))
    best_feature = max(info_gain, key=lambda x: x[1])[0]
    # create the tree
    tree = {best_feature: {}}
    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(columns=[best_feature])
        sub_y = y[X[best_feature] == value]
        tree[best_feature][value] = id3(sub_X, sub_y)
    return tree



# define the C4.5 function
def c45(X, y):
    # base case: all target variables are the same
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    # base case: no more features to split
    if len(X.columns) == 0:
        return y.mode()[0]
    # calculate the information gain ratio for each feature
    info_gain_ratio = []
    for feature in X.columns:
        # calculate entropy for each value of the feature
        entropy = 0
        for value in np.unique(X[feature]):
            sub_y = y[X[feature] == value]
            prob = len(sub_y) / len(y)
            entropy += prob * np.log2(prob)
        entropy = -entropy
        # calculate information gain
        info_gain = entropy
        for value in np.unique(X[feature]):
            sub_y = y[X[feature] == value]
            prob = len(sub_y) / len(y)
            sub_entropy = 0
            for sub_value in np.unique(sub_y):
                sub_prob = len(sub_y[sub_y == sub_value]) / len(sub_y)
                sub_entropy += sub_prob * np.log2(sub_prob)
            sub_entropy = -sub_entropy
            info_gain -= prob * sub_entropy
        # calculate split information
        split_info = 0
        for value in np.unique(X[feature]):
            sub_y = y[X[feature] == value]
            prob = len(sub_y) / len(y)
            split_info -= prob * np.log2(prob)
        # calculate information gain ratio
        info_gain_ratio.append((feature, info_gain / split_info))
    best_feature = max(info_gain_ratio, key=lambda x: x[1])[0]
    # create the tree
    tree = {best_feature: {}}
    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(columns=[best_feature])
        sub_y = y[X[best_feature] == value]
        tree[best_feature][value] = c45(sub_X, sub_y)
    return tree



# define the CART function
def cart(X, y):
    # base case: all target variables are the same
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    # base case: no more features to split
    if len(X.columns) == 0:
        return y.mode()[0]
    # calculate the Gini impurity for each feature
    gini_impurity = []
    for feature in X.columns:
        # calculate Gini impurity for each value of the feature
        impurity = 0
        for value in np.unique(X[feature]):
            sub_y = y[X[feature] == value]
            prob = len(sub_y) / len(y)
            if len(np.unique(sub_y)) == 1:
                impurity += prob * 0
            else:
                impurity += prob * (1 - (len(sub_y[sub_y == np.unique(sub_y)[0]]) / len(sub_y))**2 - (len(sub_y[sub_y == np.unique(sub_y)[1]]) / len(sub_y))**2)
        # calculate weighted Gini impurity
        gini_impurity.append((feature, impurity))
    best_feature = min(gini_impurity, key=lambda x: x[1])[0]
    # create the tree
    tree = {best_feature: {}}
    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(columns=[best_feature])
        sub_y = y[X[best_feature] == value]
        tree[best_feature][value] = cart(sub_X, sub_y)
    return tree



# call the ID3 function
tree1 = id3(X, y)

# call the c4.5 function
tree2 = c45(X, y)

# call the CART function
tree3 = cart(X, y)

print(tree1)
print(tree2)
print(tree3)
