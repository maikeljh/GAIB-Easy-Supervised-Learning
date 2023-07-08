import numpy as np
import pandas as pd

class ID3:
    def __init__(self):
        # Constructor
        self.tree = None

    def entropy(self, labels):
        # Calculate the entropy of a list of labels
        # Count unique values
        _, counts = np.unique(labels, return_counts=True)

        # Calculate probabilities
        probabilities = counts / len(labels)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def information_gain(self, data, feature, label):
        # Calculate the information gain when splitting on a specific feature
        # Calculate total entropy
        total_entropy = self.entropy(data[label])

        # Get all unique feature values
        feature_values = data[feature].unique()
        
        # Initialize variable
        weighted_entropy = 0

        # Calculate weighted entropy
        for value in feature_values:
            # Find all rows with feature = current feature value
            subset = data[data[feature] == value]

            # Compute weight
            weight = len(subset) / len(data)

            # Update weighted entropy
            weighted_entropy += weight * self.entropy(subset[label])

        # Calculate information gain
        information_gain = total_entropy - weighted_entropy

        return information_gain

    def find_best_split(self, data, features, label):
        # Find the best feature to split the data based on information gain
        # Initialize variables
        best_feature = None
        best_gain = -1

        # Iterate each feature to find the best feature
        for feature in features:
            gain = self.information_gain(data, feature, label)
            if gain > best_gain:
                # Save best gain and feature
                best_gain = gain
                best_feature = feature

        return best_feature

    def create_tree(self, data, features, label):
        # Recursively create the decision tree
        # Get labels
        labels = data[label]
        
        # Check if all rows have the same label, return a leaf node
        if len(labels.unique()) == 1:
            return labels.iloc[0]
        
        # Check if there are no more features to split on
        if len(features) == 0:
            return labels.value_counts().idxmax()
        
        # Find the best feature to split on
        best_feature = self.find_best_split(data, features, label)

        # Create tree
        tree = {best_feature: {}}

        # Create remaining features
        remaining_features = [feat for feat in features if feat != best_feature]

        # Iterate over the unique values of the best feature
        for value in data[best_feature].unique():
            # Find all rows with best_feature = current feature unique value
            subset = data[data[best_feature] == value]

            # Check if there were no rows found
            if len(subset) == 0:
                tree[best_feature][value] = labels.value_counts().idxmax()
            else:
                # Create subtree
                tree[best_feature][value] = self.create_tree(subset, remaining_features, label)

        return tree

    def fit(self, X_train, y_train):
        # Convert the training data to a pandas DataFrame
        data = pd.DataFrame(X_train)
        data['label'] = y_train

        # Get all features
        features = list(data.columns[:-1])

        # Get label
        label = data.columns[-1]

        # Create ID3 Tree
        self.tree = self.create_tree(data, features, label)

    def predict_row(self, row, tree):
        # Recursively traverse the decision tree to predict the label of a row
        # Check if current node leaf or not
        if isinstance(tree, str):
            return tree

        # Get feature from node and feature value from row
        feature = list(tree.keys())[0]
        value = row[feature]

        # Check if feature value found in branch tree
        if value not in tree[feature]:
            return list(tree[feature].values())[0]
        
        # Get subtree
        subtree = tree[feature][value]

        # Predict through subtree
        return self.predict_row(row, subtree)

    def predict(self, X_test):
        # Make predictions for X_test
        # Initialize variables
        data = pd.DataFrame(X_test)
        predictions = []

        # Predict each row
        for _, row in data.iterrows():
            prediction = self.predict_row(row, self.tree)
            predictions.append(prediction)

        return predictions
