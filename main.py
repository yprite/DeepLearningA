#http://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220991490995&parentCategoryNo=49&categoryNo=&viewDate=&isShowPopularPosts=true&from=search
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#상관관계 분석
def process_relation():
    # -1.0 ≤ r ≤ -0.7: 매우 강한 음의 상관관계
    # -0.7 < r ≤ -0.3: 강한 음의 상관관계
    # -0.3 < r ≤ -0.1: 약한 음의 상관관계
    # -0.1 < r ≤ 0.1: 상관관계 없음
    # 0.1 < r ≤ 0.3: 약한 양의 상관관계
    # 0.3 < r ≤ 0.7: 강한 양의 상관관계
    # 0.7 < r ≤ 1.0: 매우 강한 양의 상관관계
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    df = pd.read_csv('./data/data1.csv', sep=',')
    cols = ['samsung', 'ksp', 'bond', 'memory', 'smartphone']
    # sns.set(style='whitegrid', context='notebook')

    # sns.pairplot(df[cols], size=2.5)
    # plt.show()
    # sns.reset_orig()

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
    plt.show()

#LSTM
def YLSTM():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense, Activation
    import datetime

    data = pd.read_csv('./data/samsung_2010.csv')
    high_prices = data['고가'].values
    low_prices = data['저가'].values
    mid_prices = (high_prices + low_prices) / 2

    seq_len = 90 #window 값이 90
    sequence_length = seq_len + 1
    result = []
    for index in range(len(mid_prices) - sequence_length):
        result.append(mid_prices[index:index + sequence_length])

    normalized_data = []
    for window in result:
        normalized_window = [[(float(p) / float(window[0])) - 1] for p in window]
        normalized_data.append(normalized_window)
    result = np.array(normalized_data)

    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:,:-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    y_test = result[row:, -1]

    print(x_train.shape, x_test.shape)


if __name__ == '__main__':
    print_hi('PyCharm')
    #process_relation()
    YLSTM()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
