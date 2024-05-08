import pandas as pd
import statsmodels.api as sm

# Tu DataFrame original
data = {
    'total': [72.0, 17.0, 11.0, 69.0, 18.0, 13.0, 49.0, 30.0, 20.0, 23.0, 58.0, 19.0, 14.0, 68.0, 18.0, 30.0, 50.0, 20.0, 6.0, 85.0, 9.0, 12.0, 75.0, 13.0, 8.0, 83.0, 9.0, 8.0, 83.0, 9.0, 17.0, 65.0, 19.0, 15.0, 69.0, 15.0, 8.0, 82.0, 9.0, 9.0, 79.0, 12.0, 61.0, 24.0, 15.0],
    'women': [76.0, 14.0, 10.0, 74.0, 16.0, 10.0, 60.0, 22.0, 18.0, 31.0, 49.0, 20.0, 17.0, 62.0, 21.0, 38.0, 41.0, 22.0, 7.0, 84.0, 9.0, 16.0, 68.0, 16.0, 7.0, 83.0, 10.0, 9.0, 81.0, 10.0, 22.0, 59.0, 19.0, 23.0, 58.0, 18.0, 9.0, 82.0, 9.0, 10.0, 76.0, 13.0, 71.0, 16.0, 13.0],
    'man': [68.0, 19.0, 13.0, 64.0, 20.0, 16.0, 39.0, 38.0, 22.0, 15.0, 67.0, 18.0, 11.0, 74.0, 16.0, 23.0, 60.0, 17.0, 6.0, 85.0, 9.0, 9.0, 82.0, 10.0, 8.0, 84.0, 8.0, 8.0, 84.0, 8.0, 12.0, 70.0, 18.0, 8.0, 80.0, 12.0, 8.0, 83.0, 10.0, 8.0, 82.0, 10.0, 51.0, 32.0, 17.0],
    '18-34': [61.0, 25.0, 15.0, 57.0, 27.0, 16.0, 48.0, 32.0, 20.0, 28.0, 53.0, 18.0, 22.0, 60.0, 18.0, 32.0, 49.0, 18.0, 12.0, 72.0, 15.0, 17.0, 67.0, 16.0, 13.0, 72.0, 15.0, 16.0, 71.0, 13.0, 22.0, 57.0, 20.0, 23.0, 58.0, 19.0, 16.0, 69.0, 14.0, 17.0, 67.0, 16.0, 53.0, 31.0, 16.0],
    '35-54': [77.0, 11.0, 12.0, 71.0, 16.0, 13.0, 51.0, 27.0, 22.0, 26.0, 53.0, 21.0, 13.0, 67.0, 20.0, 31.0, 48.0, 21.0, 5.0, 86.0, 9.0, 13.0, 74.0, 13.0, 7.0, 84.0, 9.0, 7.0, 82.0, 10.0, 14.0, 65.0, 21.0, 12.0, 73.0, 15.0, 8.0, 82.0, 10.0, 7.0, 82.0, 11.0, 64.0, 21.0, 16.0],
    '55+': [78.0, 14.0, 9.0, 77.0, 12.0, 11.0, 49.0, 31.0, 20.0, 16.0, 65.0, 18.0, 8.0, 75.0, 17.0, 28.0, 53.0, 19.0, 2.0, 94.0, 4.0, 8.0, 82.0, 10.0, 4.0, 92.0, 4.0, 3.0, 92.0, 5.0, 14.0, 71.0, 15.0, 12.0, 76.0, 12.0, 2.0, 93.0, 5.0, 4.0, 87.0, 8.0, 66.0, 21.0, 13.0]
}

df = pd.DataFrame(data)

# Crear un diccionario para almacenar los DataFrames generados
dfs = {}

# Dividir el DataFrame original en sub-DataFrames de tres filas cada uno
for i in range(0, len(df), 3):
    dfs[f'df_{i//3}'] = df.iloc[i:i+3]

# Transpose and rename the index to outcome type
dfs['df_0'] = dfs['df_0'].astype(float)
df = dfs['df_0'].T
df.columns = ['win', 'lose', 'unknown']
df = df.reset_index()
df.rename(columns={'index': 'group'}, inplace=True)

# Melt the DataFrame to long format
df_long = df.melt(id_vars='group', value_vars=['win', 'lose', 'unknown'], var_name='Outcome', value_name='Probability')

# Create dummy variables for group and outcome
df_long = pd.get_dummies(df_long, columns=['group', 'Outcome'])

# Example model for 'win' outcome
def create_model(df, outcome_col):
    X = df.drop(['Probability'] + [col for col in df.columns if col.startswith('Outcome_') and not col.endswith(outcome_col)], axis=1)
    y = df['Probability']
    X = sm.add_constant(X)  # Adds a constant term to the predictors
    model = sm.OLS(y, X).fit()
    return model

# Create models
models = {outcome: create_model(df_long, outcome) for outcome in ['win', 'lose', 'unknown']}

# Function to make predictions
def predict_outcome(attributes):
    data = {f'group_{attr}': 1 for attr in attributes.split(', ')}
    df_input = pd.DataFrame([data])
    df_input = sm.add_constant(df_input.reindex(columns=models['win'].params.index, fill_value=0))
    predictions = {outcome: model.predict(df_input)[0] for outcome, model in models.items()}
    return predictions

# Example usage
attributes = 'man, 35-54'
predictions = predict_outcome(attributes)
print("Predicted probabilities:", predictions)