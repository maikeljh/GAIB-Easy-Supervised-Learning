import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from src.knn import KNN
from src.lr import LogisticRegression
from src.id3 import ID3

def parse_arguments():
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specify the model")
    parser.add_argument("--data", type=str, help="Specify the data file")
    parser.add_argument("--k_nearest", type=int, help="Specify the value of k nearest neighbours for KNN model")
    parser.add_argument("--lr", type=int, help="Specify the value of learning rate for logistic regression model")
    parser.add_argument("--epochs", type=int, help="Specify the value of epochs for logistic regression model")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Validation Arguments
    if args.model == None:
        print("Error! Please specify the model in the args with --model")
        exit()
    elif args.data == None:
        print("Error! Please specify the dataset in the args with --data")
        exit()

    # Load dataset
    print("Load Dataset\n")
    df = pd.read_csv(args.data)
    print(df)
    print()

    # Split train and test set
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Engineering
    if args.model == "knn" or args.model == "log_reg":
        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if args.model == "log_reg":
        # Encode target
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)

    # Machine Learning
    # Model
    if args.model == "knn":
        model = KNN(k=args.k_nearest)
    elif args.model == "log_reg":
        model = LogisticRegression()
    elif args.model == "id_3":
        model = ID3()
    
    # Fit model
    model.fit(X_train, y_train)
    
    print("Evaluation")

    # Prediction and Evaluation
    y_pred = model.predict(X_test)

    # Predicted Classes
    # Label back to category
    if args.model == "log_reg":
        y_pred = encoder.inverse_transform(y_pred)

    # Creating DF for comparison between predicted and actual classes
    y_pred_series = pd.Series(y_pred, name='y_pred')
    y_test_concat = y_test.reset_index(drop=True).rename("y_actual")
    result_df = pd.concat([y_pred_series, y_test_concat], axis=1)

    print("Predicted and Actual Classes:")
    print(result_df)
    print()

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print()

    # Classification Report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    print()

if __name__ == "__main__":
    main()  