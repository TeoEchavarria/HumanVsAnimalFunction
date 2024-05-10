import pandas as pd
import statsmodels.api as sm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def create_model(df, outcome_col, age_group, gender):
    column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['age_group', 'gender'])
    ],
    remainder='passthrough'
)

    pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('regressor', LogisticRegression(max_iter=1000))
    ])
    
    X = df[['age_group', 'gender']]
    y = df[outcome_col]
    
    # Training the model
    pipeline.fit(X, y)

    input_data = pd.DataFrame({'age_group': [age_group], 'gender': [gender]})
    prediction = pipeline.predict(input_data)[0]
    return prediction
    
    

def model_train(df, name, age_group, gender):
    df = df.astype(float)
    df = df.T
    df.columns = ['win', 'lose', 'unknown']
    df = df.reset_index()
    df.rename(columns={'index': 'group'}, inplace=True)
    
    mapping = {
    'total': ('all', 'all'),
    'women': ('female', 'all'),
    'man': ('male', 'all'),
    '18-34': ('all', '18-34'),
    '35-54': ('all', '35-54'),
    '55+': ('all', '55+')
    }

    df['gender'], df['age_group'] = zip(*df['group'].map(mapping))
    df = df[(df['gender'] != 'all') | (df['age_group'] != 'all')]
    models = {outcome: create_model(df, outcome, age_group, gender) for outcome in ['win', 'lose', 'unknown']}
    return models

