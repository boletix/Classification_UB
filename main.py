import click
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# Load CSV file
data = pd.read_csv('adult.csv')

@click.command()
@click.option("--cat_imputer", default="most_frequent")
@click.option("--n_iter", default=1000)
@click.option("--max_depth", default=3)

# Transform income to target
target_names = data.income.unique()
data['income'] = data['income'].map({target_names[0]: 0, target_names[1]: 1})

def replace_2nan(df):
    df[df == '?'] = np.nan
    return df

def preprocessing(data):
    # Replacing ? per np.nan
    data = data.apply(replace_2nan)

    # Integrate Human Development Index
    url = 'https://en.wikipedia.org/wiki/List_of_countries_by_Human_Development_Index'
    hdi_country = pd.read_html(url)[1]

    # Sorted aplhabetical
    sorted_hdi = hdi_country.sort_values(by='Country or territory')

    # Fix names that were wrong to use the same names as on wikipedia
    data['native.country'] = data['native.country'].str.replace('-', ' ')
    data.loc[data['native.country'] == 'Trinadad&Tobago', 'native.country'] = 'Trinidad and Tobago'
    data.loc[data['native.country'] == 'South', 'native.country'] = 'South Korea'
    data.loc[data['native.country'] == 'Holand Netherlands', 'native.country'] = 'Netherlands'
    uk = ['England', 'Ireland', 'Scotland']
    data.loc[data['native.country'].isin(uk), 'native.country'] = 'United Kingdom'
    data.loc[data['native.country'] == 'Hong', 'native.country'] = 'Hong Kong'
    data.loc[data['native.country'] == 'Columbia', 'native.country'] = 'Colombia'
    data.loc[data['native.country'] == 'Yugoslavia', 'native.country'] = 'Russia'
    data.loc[data['native.country'] == 'Outlying US(Guam USVI etc)', 'native.country'] = 'United States'

    # Filter ordered list by countries that are present in dataset
    mask_sorted_hdi = sorted_hdi[sorted_hdi['Country or territory'].isin(data['native.country'].unique())]

    # Preparando para hacer el map
    hdi_mapping = dict(zip(mask_sorted_hdi['Country or territory'], mask_sorted_hdi['HDI']))

    # Insertar la neva columna mediante mapping
    data['hdi.country'] = data['native.country'].map(hdi_mapping)
    return data

# Apply the preprocessing
data_preprocess = preprocessing(data)

# Prepare the Pipeline with all the steps
imputer = SimpleImputer(strategy="median")
cat_encoder = OneHotEncoder(handle_unknown='ignore')

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

one_hot_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy=cat_imputer)),
    ('encoder', OneHotEncoder())
])

# Split columns in categorical and numerical
num_vars = data_preprocess.drop(columns='income').select_dtypes(exclude=['object']).columns
cat_vars = data_preprocess.drop(columns='income').select_dtypes(include=['object']).columns

full_pipe = ColumnTransformer([
    ('num', num_pipe, num_vars),
    ('cat', one_hot_pipe, cat_vars)
])

# Split features and target variables
X = data_preprocess.drop('income', axis=1)
y = data_preprocess['income']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply Transformation
full_pipe.fit(X)
X_train_ready = full_pipe.transform(X_train)
X_valid_ready = full_pipe.transform(X_valid)

def display_matrix(y_preds, y_valid, norm='true'):
    ConfusionMatrixDisplay.from_predictions(y_valid, y_preds, normalize=norm, display_labels=target_names)
    plt.show()

def model_eval(model, Xt, Xv, yt, yv):
    model.fit(Xt, yt)
    y_preds = model.predict(Xv)

    display_matrix(y_preds, yv)
    accu = accuracy_score(yv, y_preds)

    return model, accu

# Create the models
log_reg = LogisticRegression(max_iter=n_iter)
decission_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=6)

print('Predictions using the Logistic Regressor')
log_reg, logistic_accu = model_eval(log_reg, X_train_ready, X_valid_ready, y_train, y_valid)

print('Predictions using the Decission Tree')
decission_tree, tree_accu = model_eval(decission_tree, X_train_ready, X_valid_ready, y_train, y_valid)

# Log the metrics to MLFlow
mlflow.log_metric("logReg_accuracy", logistic_accu)
mlflow.log_metric("tree_accuracy", tree_accu)

# Log the model
mlflow.sklearn.log_model(log_reg, "LogReg_model")
mlflow.sklearn.log_model(decission_tree, "Tree_model")

if __name__ == "__main__":
    main()
