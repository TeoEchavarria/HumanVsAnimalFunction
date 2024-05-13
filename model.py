import pandas as pd
from joblib import dump, load
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_model(df, outcome_col, age_group, gender, name):   
    df['interaction'] = df['age_group'] + '_' + df['gender']
    X = df[['age_group', 'gender', 'interaction']]
    y = df[outcome_col]
    
    def add_interaction(df):
        df = df.copy()
        df['interaction'] = df['age_group'] + "_" + df['gender']
        return df
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['age_group', 'gender', 'interaction'])
        ],
        remainder='passthrough'
    )

    rf_pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    
    gbm_pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('classifier', GradientBoostingClassifier(n_estimators=100))
    ])
        
    # Training the model
    rf_pipeline.fit(X, y)
    gbm_pipeline.fit(X, y)
    
    dir_path_rm = f'models/rf/{name}'
    dir_path_gbm = f'models/gbm/{name}'

    # Check if the directory exists
    if not os.path.exists(dir_path_rm):
        os.makedirs(dir_path_rm)
        print("Directory created:", dir_path_rm)
    if not os.path.exists(dir_path_gbm):
        os.makedirs(dir_path_gbm)
        print("Directory created:", dir_path_gbm)

        
    dump(rf_pipeline, f'models/rf/{name}/{outcome_col}.joblib')  # Save the model to a file
    dump(gbm_pipeline, f'models/gbm/{name}/{outcome_col}.joblib')

    # input_data = pd.DataFrame({'age_group': [age_group], 'gender': [gender], 'interaction': [age_group + '_' + gender]})
    
    # try:
    #     rf_prediction = rf_pipeline.predict(input_data)
    #     gbm_prediction = gbm_pipeline.predict(input_data)
    # except Exception as e:
    #     print("An error occurred during prediction:", e)
    return None #(rf_prediction+gbm_prediction)/2

def predict(outcome_col, age_group, gender, name):
    rf_model_loaded = load(f'models/rf/{name}/{outcome_col}.joblib')
    gbm_model_loaded = load(f'models/rf/{name}/{outcome_col}.joblib')

    # Now you can use the loaded model to make predictions
    # Prepare input data as before
    input_data = pd.DataFrame({
        'age_group': [age_group],
        'gender': [gender],
        'interaction': [f'{age_group}_{gender}']
    })

    # Predict using the loaded models
    rf_prediction = rf_model_loaded.predict(input_data)[0]
    gbm_prediction = gbm_model_loaded.predict(input_data)[0]
    
    return (rf_prediction+gbm_prediction)/2



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
    

def model_train(age_group, gender, name):
    models = {outcome: predict(outcome, age_group, gender, name) for outcome in ['win', 'lose', 'unknown']}
    models["imageUrl"] = links[name]
    return models