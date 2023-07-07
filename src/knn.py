import numpy as np

class KNN:
    def __init__(self, k = 3):
        # Constructor
        self.k = k

    def euclidean_distance(self, row_1, row_2):
        # Calculate euclidean distance between two rows
        return np.sqrt(np.sum((row_1 - row_2) ** 2))
    
    def change_k(self, k):
        # Change k number
        self.k = k

    def get_nearest_neighbours(self, test):
        # Return k nearest neighbours
        distances = []

        # Find all euclidean distance between train set and X_predict
        for i, row in enumerate(self.X_train):
            distance = self.euclidean_distance(row, test)
            distances.append((i, distance))

        # Sort by distance descending
        distances.sort(key=lambda x: x[1])

        # Filter to k nearest neighbours
        neighbours = []
        for i in range(self.k):
            neighbours.append(distances[i][0])

            # Handle k over available data
            if i == len(distances) - 1:
                break

        return neighbours
    
    def fit(self, X_train, y_train):
        # Save train_set
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        # Predictions
        y_pred = []

        # Predict for each row in test_set
        for row in X_test:
            # Get k nearest neighbours
            neighbours = self.get_nearest_neighbours(row)
            
            # Get all labels of neighbours
            labels = [self.y_train.iloc[neighbour] for neighbour in neighbours]

            # Find mode
            prediction = max(set(labels), key=labels.count)

            # Add prediction to list
            y_pred.append(prediction)
        
        return y_pred
