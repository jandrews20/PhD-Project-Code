import ember

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ember.create_vectorized_features("C:/Users/40237845/Downloads/ember_2017_2")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("C:/Users/40237845/Downloads/ember_2017_2")

    print(X_train)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
