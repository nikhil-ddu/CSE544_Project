import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from linear_regression_model import LinearRegression
pd.options.mode.chained_assignment = None

grades = {'A': 10, 'B': 20, 'C': 30, 'D': 40, 'E': 50, 'F': 60, 'G': 70}
y = 'borrower_rate'


# Method to predict borrower_rate
# Training set contains all "non-current" loans and the test set contains all "current" loans
def exec_regression(df_train, df_test,x0, x1, x2):
    X_train = df_train[[x1, x2]]
    temp_grade = df_train[x0]
    if x0 == 'grade':
        temp_grade = df_train[x0].map(grades)
    X_train[x0] = temp_grade
    y_train = df_train[y]
    reg = LinearRegression().fit(X_train, y_train)

    y_true = df_test[y]
    X_test = df_test[[x1, x2]]
    temp_grade = df_test[x0]
    if x0 == 'grade':
        temp_grade = df_test[x0].map(grades)
    X_test[x0] = temp_grade
    y_test = reg.predict(X_test)

    error_list = [(abs(i - j)*100/i) for i,j in zip(y_true.tolist(), y_test.tolist())]
    print("Error Mean: " + x0 + ", " + x1 + ", " + x2 + " = ", np.mean(error_list))

    plt.style.use('fivethirtyeight')
    plt.scatter(y_true, abs(y_true - y_test), color="blue", s=0.02, label='Test data')
    plt.hlines(y=0, xmin=0, xmax=0.4, linewidth=2)
    plt.legend(loc='upper right')
    plt.title("Residual errors: " + x0 + ", " + x1 + ", " + x2)
    plt.xlabel('true borrower rate')
    plt.ylabel('residual error')
    plt.show()


def main(filePath):
    df = pd.read_csv(filePath)

    # Run1: Regressors: amount_borrowed, term, grade
    df1 = df[['amount_borrowed', 'term', 'borrower_rate', 'grade', 'loan_status_description']]
    df_train = df1.loc[df1['loan_status_description'] != 'CURRENT']
    df_test = df1.loc[df1['loan_status_description'] != 'CURRENT']
    exec_regression(df_train, df_test, 'grade', 'amount_borrowed', 'term')

    # Run2: Regressors: amount_borrowed, installment, grade
    df1 = df[['amount_borrowed', 'installment', 'borrower_rate', 'grade', 'loan_status_description']]
    df_train = df1.loc[df1['loan_status_description'] != 'CURRENT']
    df_test = df1.loc[df1['loan_status_description'] != 'CURRENT']
    exec_regression(df_train, df_test, 'grade', 'amount_borrowed', 'installment')

    # Run3: Regressors: term, installment, grade
    df1 = df[['term', 'installment', 'borrower_rate', 'grade', 'loan_status_description']]
    df_train = df1.loc[df1['loan_status_description'] != 'CURRENT']
    df_test = df1.loc[df1['loan_status_description'] != 'CURRENT']
    exec_regression(df_train, df_test, 'grade', 'term', 'installment')

    # Run3: Regressors: term, installment, amount_borrowed
    df1 = df[['term', 'installment', 'borrower_rate', 'amount_borrowed', 'loan_status_description']]
    df_train = df1.loc[df1['loan_status_description'] != 'CURRENT']
    df_test = df1.loc[df1['loan_status_description'] != 'CURRENT']
    exec_regression(df_train, df_test, 'term', 'amount_borrowed', 'installment')

main(sys.argv[1])