
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df):
    categorical = ['PULocationID', 'DOLocationID']
    #numerical = ['trip_distance']
    target = 'duration'
    y_train = df[target].values

    dv = DictVectorizer()
    train_dicts = df[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)

    print(lr.intercept_)

    
    return X_train, lr


