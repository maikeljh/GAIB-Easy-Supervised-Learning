import numpy as np

class LogisticRegression:
    def __init__(self, batch_size=100, epochs=1000, learning_rate=0.01):
        # Constructor
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train):
        # Fit X_train and y_train to model
        self.X_train = X_train
        self.y_train = y_train

        # Train model
        # Initialize variables
        n_rows, n_cols = self.X_train.shape
        weights = np.zeros(n_cols)
        bias = 0
        y = self.y_train.reshape(n_rows, 1)
        losses = []

        # Training loop
        for _ in range(self.epochs):
            # Iterate over batches
            for i in range((n_rows - 1) // self.batch_size + 1):
                # Get start index and end index of current batch
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size

                # Get features and labels of current batch
                xb = self.X_train[start_i:end_i]
                yb = self.y_train[start_i:end_i]

                # Calculate the predicted probabilities
                y_hat = self.sigmoid(np.dot(xb, weights) + bias)

                # Calculate gradients
                dw, db = self.gradients(xb, yb, y_hat)

                # Update weights
                weights -= self.learning_rate * dw.reshape(-1)

                # Update bias
                bias -= self.learning_rate * db

            # Compute loss for current epoch
            l = self.loss(y, self.sigmoid(np.dot(self.X_train, weights) + bias))

            # Save loss
            losses.append(l)

        # Output of train
        self.weights = weights
        self.bias = bias
        self.losses = losses

    def sigmoid(self, z):
        # Sigmoid function
        return 1.0 / (1 + np.exp(-z))

    def loss(self, y, y_hat):
        # Compute the logistic loss function
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def gradients(self, X, y, y_hat):
        # Compute the gradients of the logistic loss function
        n_rows = X.shape[0]
        dw = (1 / n_rows) * np.dot(X.T, (y_hat - y))
        db = (1 / n_rows) * np.sum(y_hat - y)
        return dw, db

    def predict(self, X):
        # Predict class labels for input features
        preds = self.sigmoid(np.dot(X, self.weights) + self.bias)
        pred_class = [1 if pred >= 0.5 else 0 for pred in preds]
        return pred_class