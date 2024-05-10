import pandas as pd
import statsmodels.api as sm
import joblib

def predict_outcome(name, attributes):
    models = {
        "win": joblib.load(f'{name}-win.pkl'),
        "lose": joblib.load(f'{name}-lose.pkl'),
        "unknown": joblib.load(f'{name}-unknown.pkl'),
    }
    data = {f'group_{attr}': 1 for attr in attributes.split(', ')}
    df_input = pd.DataFrame([data])
    df_input = sm.add_constant(df_input.reindex(columns=models['win'].params.index, fill_value=0))
    predictions = {outcome: model.predict(df_input)[0] for outcome, model in models.items()}
    return predictions

