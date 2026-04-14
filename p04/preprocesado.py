import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder  # Librería estándar para Target Encoding
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectKBest, f_classif


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
def eliminacion_duplicados_train(df):
    cols = df.columns.drop('Target_Risco')
    df = df.drop_duplicates(subset=cols)

    # Eliminar columnas inútiles (ID)
    df = df.drop(columns=['ID_Cliente'])
    df = df.drop(columns=['Data_Solicitude'])
    return df
def eliminacion_duplicados_test(df):
    # Eliminar columnas inútiles (ID)
    df = df.drop(columns=['ID_Cliente'])
    df = df.drop(columns=['Data_Solicitude'])
    return df



COLS_OUTLIER = [
    'Idade', 'Lonxitude_Nome', 'Num_Fillos', 'Anos_Emprego',
    'Ingresos_Anuais', 'Tempo_Web_Minutos', 'Distancia_Oficina_Km',
    'Patrimonio_Total', 'Debeda_Total', 'Numero_Tarxetas',
    'Utilizacion_Credito', 'Consultas_Risco_6M', 'Limite_Credito_Total',
    'Cota_Mensual_Prestamos', 'Ratio_Cota_Ingresos', 'Prestamos_Activos',
    'Antiguedade_Cliente_Anos', 'Saldo_Medio_3M', 'Variacion_Saldo_6M',
    'Fondo_Emerxencia_Meses', 'Indice_Estres_Financeiro'
]

COLS_IMPUTAR = [
    'Idade', 'Lonxitude_Nome', 'Num_Fillos', 'Anos_Emprego',
    'Ingresos_Anuais', 'Tempo_Web_Minutos', 'Distancia_Oficina_Km',
    'Patrimonio_Total', 'Debeda_Total', 'Numero_Tarxetas',
    'Ratio_Deuda_Patrimonio',
    'Ratio_Deuda_Ingresos', 'Tarjetas_Antiguedad', 'Capacidad_Ahorro',
    'Utilizacion_Credito', 'Consultas_Risco_6M', 'Limite_Credito_Total',
    'Cota_Mensual_Prestamos', 'Ratio_Cota_Ingresos', 'Prestamos_Activos',
    'Antiguedade_Cliente_Anos', 'Saldo_Medio_3M', 'Variacion_Saldo_6M',
    'Fondo_Emerxencia_Meses', 'Indice_Estres_Financeiro'
]


# ── Definición de columnas por tipo ──────────────────────────────────────────
COLS_NUMERIC = [
    'Idade', 'Lonxitude_Nome', 'Num_Fillos', 'Anos_Emprego',
    'Ingresos_Anuais', 'Tempo_Web_Minutos', 'Distancia_Oficina_Km',
    'Patrimonio_Total', 'Debeda_Total', 'Numero_Tarxetas', 'Ratio_Deuda_Patrimonio', 'Ratio_Deuda_Ingresos', 
    'Tarjetas_Antiguedad', 'Capacidad_Ahorro',
    'Utilizacion_Credito', 'Consultas_Risco_6M', 'Limite_Credito_Total',
    'Cota_Mensual_Prestamos', 'Ratio_Cota_Ingresos', 'Prestamos_Activos',
    'Antiguedade_Cliente_Anos', 'Saldo_Medio_3M', 'Variacion_Saldo_6M',
    'Fondo_Emerxencia_Meses', 'Indice_Estres_Financeiro'
]

COLS_BINARY    = ['Subscricion_Email', 'Historial_Impagos']  # 0/1

COLS_ORDINAL   = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude']

COLS_ONEHOT    = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude']

# Nueva categoría para evitar la explosión de columnas del OneHot
COLS_TARGET_ENC = ['Codigo_Postal'] 

# ── Preprocessor A: para árboles (XGBoost, RF, LightGBM) ─────────────────────
preprocessor_tree = ColumnTransformer(transformers=[
    ('num',  'passthrough',                          COLS_NUMERIC),
    ('cat',  OrdinalEncoder(handle_unknown='use_encoded_value',
                            unknown_value=-1),       COLS_ORDINAL),
    ('target', TargetEncoder(smoothing=10),          COLS_TARGET_ENC), # Codificamos CP por su riesgo medio
    ('bin',  'passthrough',                          COLS_BINARY),
], remainder='drop')

# ── Preprocessor B: para modelos lineales / KNN / MLP ────────────────────────
preprocessor_linear = ColumnTransformer(transformers=[
    ('num',  RobustScaler(),                         COLS_NUMERIC),
    ('target', TargetEncoder(smoothing=10),          COLS_TARGET_ENC), # CP como valor numérico continuo
    ('cat',  OneHotEncoder(handle_unknown='ignore',
                           sparse_output=False),     COLS_ONEHOT),
    ('bin',  'passthrough',                          COLS_BINARY),
], remainder='drop')


"""
## FUNCIÓN DE EXPLICACIÓN DO PORQUÉ SE SELECCIONARON OS PARÁMETROS DECIDIDOS

# Detección de valores atípicos modificando el multiplicador del IQR
df_copia = df_train.copy()
df_copia = eliminacion_duplicados_train(df_copia)
total_celdas = len(df_copia) * len(COLS_OUTLIER)

# Inicializamos contador de atípicos por fila
df_copia["num_atipicos"] = 0

for m in [1.5, 3, 6]:
    total_atipicos = 0

    for col in COLS_OUTLIER:
        Q1 = df_copia[col].quantile(0.25)
        Q3 = df_copia[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - m * IQR
        upper = Q3 + m * IQR

        cond= (df_copia[col] < lower) | (df_copia[col] > upper)
        total_atipicos += cond.sum()
        df_copia.loc[cond, "num_atipicos"] += 1

    porcentaje = 100 * total_atipicos / total_celdas
    distribucion_atipicos = df_copia["num_atipicos"].value_counts().sort_index()

    print(f"Multiplicador {m}: {total_atipicos} valores atípicos " f"({porcentaje:.2f}% do total)")
    print(distribucion_atipicos)
    print()

"""

def deteccion_outliers(df_train, df_test, iqr_multiplier=1.5):
    # ── Aprender límites IQR SOLO en train ───────────────────────────────────────
    iqr_limits = {}
    for col in COLS_OUTLIER:
        Q1 = df_train[col].quantile(0.25)
        Q3 = df_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        iqr_limits[col] = (lower, upper)

    # ── Reemplazar atípicos por NaN (train y test con los mismos límites) ─────────
    def replace_outliers_with_nan(df, limits):
        df = df.copy()
        for col, (lower, upper) in limits.items():
            mask = (df[col] < lower) | (df[col] > upper)
            df.loc[mask, col] = np.nan
        return df

    df_train = replace_outliers_with_nan(df_train, iqr_limits)
    df_test  = replace_outliers_with_nan(df_test,  iqr_limits)
    return df_train, df_test

def ingenieria_variables(df_train, df_test):
    # Aquí se pueden crear nuevas variables a partir de las existentes
    # Ejemplo: ratio de deuda a ingresos
    df_train['Ratio_Deuda_Ingresos'] = df_train['Debeda_Total'] / (df_train['Ingresos_Anuais'] + 1e-6)
    df_test['Ratio_Deuda_Ingresos'] = df_test['Debeda_Total'] / (df_test['Ingresos_Anuais'] + 1e-6)

    #df_train['Patrimoinio_Neto_Real'] = df_train['Patrimonio_Total'] - df_train['Debeda_Total']
    #df_test['Patrimoinio_Neto_Real'] = df_test['Patrimonio_Total'] - df_test['Debeda_Total']
    df_train['Ratio_Deuda_Patrimonio'] = df_train['Debeda_Total'] / (df_train['Patrimonio_Total'] + 1)
    df_test['Ratio_Deuda_Patrimonio'] = df_test['Debeda_Total'] / (df_test['Patrimonio_Total'] + 1)
    
    # Ejemplo: interacción entre número de tarjetas y antigüedad del cliente
    df_train['Tarjetas_Antiguedad'] = df_train['Numero_Tarxetas'] * df_train['Antiguedade_Cliente_Anos']
    df_test['Tarjetas_Antiguedad'] = df_test['Numero_Tarxetas'] * df_test['Antiguedade_Cliente_Anos']

    # Capacidad de ahorro
    df_train['Capacidad_Ahorro'] = df_train['Ingresos_Anuais'] - df_train['Cota_Mensual_Prestamos'] * 12
    df_test['Capacidad_Ahorro'] = df_test['Ingresos_Anuais'] - df_test['Cota_Mensual_Prestamos'] * 12
    
    return df_train, df_test


def eliminar_o_imputar(df_train, df_val, max_faltantes=3):
    # ── 1. Eliminar filas con demasiados NaN ──────────────────────────────────
    mask = df_train.isnull().sum(axis=1) < max_faltantes
    df_train = df_train[mask].copy()

    # ── 2. Configurar el imputador (regresión iterativa) ──────────────────────
    imputer_pro = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=50, random_state=42),
        max_iter=10,
        random_state=42,
        initial_strategy='median',
        imputation_order='ascending'
    )
    imputer_ligero = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=42,
        initial_strategy='median',
        imputation_order='ascending'
    )

    # Alternativa: usar la mediana para imputar (más rápido pero menos preciso)
    #df_train[COLS_IMPUTAR] = df_train[COLS_IMPUTAR].fillna(df_train[COLS_IMPUTAR].median())
    #df_val[COLS_IMPUTAR]   = df_val[COLS_IMPUTAR].fillna(df_train[COLS_IMPUTAR].median())

    # ── 3. Fitear SOLO en train, transformar ambos ────────────────────────────
    df_train[COLS_IMPUTAR] = imputer_ligero.fit_transform(df_train[COLS_IMPUTAR])
    df_val[COLS_IMPUTAR]   = imputer_ligero.transform(df_val[COLS_IMPUTAR])

    return df_train, df_val

def seleccion_caracteristicas_check(X_train_linear, y_train, target='Target_Risco', num_features=20):
    # 1. Configuramos el selector (usamos f_classif para clasificación)
    selector = SelectKBest(score_func=f_classif, k='all') # k='all' para ver todas

    # 2. Ajustamos con los datos ya transformados (importante: datos numéricos)
    selector.fit(X_train_linear, y_train)

    # 3. Creamos un DataFrame con los resultados
    df_scores = pd.DataFrame({
        'Característica': X_train_linear.columns,
        'Puntuación': selector.scores_
    })

    # 4. Ordenamos de mayor a menor importancia
    df_scores = df_scores.sort_values(by='Puntuación', ascending=False)

    return df_scores


def preprocesado(df_train_pre, df_test_pre, target='Target_Risco', iqr_multiplier=3, max_faltantes=5):
    df_train = eliminacion_duplicados_train(df_train_pre.copy())
    df_test = eliminacion_duplicados_test(df_test_pre.copy())

    
    df_train, df_test = deteccion_outliers(df_train, df_test, iqr_multiplier=iqr_multiplier)
    df_train, df_test = ingenieria_variables(df_train, df_test)
    df_train, df_test = eliminar_o_imputar(df_train, df_test, max_faltantes=max_faltantes)


    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_test = df_test

    X_train_tree = pd.DataFrame(
        preprocessor_tree.fit_transform(X_train, y_train),
        columns=preprocessor_tree.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_tree = pd.DataFrame(
        preprocessor_tree.transform(X_test),
        columns=preprocessor_tree.get_feature_names_out(),
        index=X_test.index,
    )

    X_train_linear = pd.DataFrame(
        preprocessor_linear.fit_transform(X_train, y_train),
        columns=preprocessor_linear.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_linear = pd.DataFrame(
        preprocessor_linear.transform(X_test),
        columns=preprocessor_linear.get_feature_names_out(),
        index=X_test.index,
    )

    df_train_tree = pd.concat([X_train_tree, y_train], axis=1)
    df_test_tree = X_test_tree.copy()
    df_train_linear = pd.concat([X_train_linear, y_train], axis=1)
    df_test_linear = X_test_linear.copy()

    #EXTRA: Dropear la columna Patrimonio_Total
    df_train_linear = df_train_linear.drop(columns=['num__Patrimonio_Total'])
    df_test_linear = df_test_linear.drop(columns=['num__Patrimonio_Total'])
    df_train_tree = df_train_tree.drop(columns=['num__Patrimonio_Total'])
    df_test_tree = df_test_tree.drop(columns=['num__Patrimonio_Total'])

    return df_train_tree, df_train_linear, df_test_tree, df_test_linear