import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

links = {
    "rat" :"https://www.animallama.com/wp-content/uploads/2018/03/rat-behavior-1.jpg", 
    "house cat" : "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRS7dFEWwxUZdC9ZZeEIBHe91KB3qqPjxYb83R7KA3UpBN2cJuO", 
    "medium dog" : "https://okcfox.com/resources/media/59d14bb8-fc9a-4784-9f88-c2a622f64784-medium16x9_IMG_1191.jpg?1680654992640", 
    "large dog" : "https://qph.cf2.quoracdn.net/main-qimg-b9a5765d2847dde72f6d57df12cab50d", 
    "kangaroo" : "https://qph.cf2.quoracdn.net/main-qimg-17397d577daa166d60cb3d98f947883c-lq", 
    "eagle" : "https://cdn.britannica.com/92/152292-050-EAF28A45/Bald-eagle.jpg", 
    "grizzly bear" : "https://i.pinimg.com/736x/36/75/3c/36753c9ee53ac48f19dea69e96648e85.jpg", 
    "wolf" : "https://cdn.britannica.com/79/195879-138-964134FE/wolves-species-time-country-Denmark-1813.jpg?w=400&h=225&c=crop", 
    "lion" : "https://www.krugerpark.co.za/images/lion-facts-786x446.jpg", 
    "gorilla" : "https://media.istockphoto.com/id/462046881/photo/silverback-gorilla-beating-chest.jpg?s=612x612&w=0&k=20&c=rMRgQ0NDKgwbC13sM0k_-W2-0cc5Td4NymibZW_jByg=", 
    "chimpanzee" : "https://i.guim.co.uk/img/media/11256d1a68a9b73e412e67380a85ceabc8d20cd9/0_0_5576_3347/master/5576.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=5b697482a111c62cbbe5fbed961daa5a", 
    "king cobra" : "https://www.naturesafariindia.com/wp-content/uploads/2023/10/King-cobra-in-Kaziranga-national-park-india.jpg", 
    "elephant" : "https://cdn.mos.cms.futurecdn.net/uiCrUgVCf64TzEdTM8x9aD-1200-80.jpg", 
    "crocodile" : "https://e3.365dm.com/23/06/2048x1152/skynews-crocodile-file-pic_6180857.jpg", 
    "goose" : "https://i.cbc.ca/1.6845820.1684283175!/fileImage/httpImage/canada-goose-attack.jpg"
}
    

def model_train(df, age_group, gender, name):
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
    '55': ('all', '55')
    }

    df['gender'], df['age_group'] = zip(*df['group'].map(mapping))
    df = df[(df['gender'] != 'all') | (df['age_group'] != 'all')]
    models = {outcome: create_model(df, outcome, age_group, gender) for outcome in ['win', 'lose', 'unknown']}
    models["imageUrl"] = links[name]
    return models