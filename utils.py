
import numpy as np
import pandas as pd

def split_val_parts(dfnow, n_split, labelcol='label'):

    hoplen = 1 / n_split

    starts = np.arange(0 ,1 ,hoplen)
    ends = starts + hoplen

    probs_random = np.random.rand(len(dfnow))

    dfs = []
    for i in range(n_split):
        print(i)
        msknow = (probs_random >= starts[i]) & (probs_random < ends[i])
        dftmp = dfnow[msknow]
        print(dftmp.shape)
        print(dftmp.groupby(labelcol).label.count())
        dfs.append(dftmp)
        print("")

    return dfs



def cross_validate(df_list, df_comparisons, classifier_func, colname, columns, n_split, labelcol='label'):
    models = []

    for i in range(n_split):
        split_inds = [ii for ii in range(n_split)]
        split_inds.remove(i)
        df_train_now = pd.concat([df_list[ii] for ii in split_inds], axis=0)
        df_test_now = df_list[i]

        Xtrain = df_train_now[columns].values
        ytrain = df_train_now[labelcol].values

        Xtest = df_test_now[columns].values
        # ytest = df_test_now[labelcol].values

        model = classifier_func()
        model = model.fit(Xtrain, ytrain)

        yhat = model.predict(Xtest)
        df_comparisons.loc[df_test_now.index, colname] = yhat

        models.append(model)

    return models, df_comparisons

