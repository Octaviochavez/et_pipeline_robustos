import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC

class CorrelationFilter(BaseEstimator, TransformerMixin):
  """
  Eliminación de variables correlacionadas

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  columns_to_drop_ : array-like
    Nombres de las columnas a eliminar.
  threshold : float
    Umbral de correlación.
  Returns
  -------
  DataFrame
    Conjunto de datos sin variables correlacionadas.
  """
  def __init__(self, threshold=0.9):
    self.threshold = threshold
    self.columns_to_drop_ = None

  def fit(self, X, y=None):
    X_df = pd.DataFrame(X)

    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    self.columns_to_drop_ = [
        col for col in upper.columns if any(upper[col] > self.threshold)
    ]

    return self

  def transform(self, X):
    X_df = pd.DataFrame(X)
    X_filtered = X_df.drop(columns=self.columns_to_drop_, errors="ignore")
    return X_filtered.values

class DataFrameConverter(BaseEstimator, TransformerMixin):
  """
  Convierte un array en un DataFrame

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  feature_names_ : array-like
    Nombres de las columnas.
  Returns
  -------
  DataFrame
    Conjunto de datos con nombres de columnas.
  """
  def __init__(self, preprocessor):
    self.preprocessor = preprocessor
    self.feature_names_ = None

  def fit(self, X, y=None):
    # Obtener nombres después de fit del preprocessor
    self.feature_names_ = self.preprocessor.get_feature_names_out()
    return self

  def transform(self, X):
    return pd.DataFrame(X, columns=self.feature_names_)

class Winsorizer(BaseEstimator, TransformerMixin):
  """
  Winsorizacion de variables numericas.

  Parametros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  limits : tuple of float
    Limites inferior y superior para el recorte de colas.
  lower_bounds_ : Series
    Cuantiles inferiores por columna (calculados en fit).
  upper_bounds_ : Series
    Cuantiles superiores por columna (calculados en fit).
  columns_ : array-like
    Nombres de columnas usadas para transformar.
  Returns
  -------
  DataFrame
    Conjunto de datos con valores recortados.
  """
  def __init__(self, limits=(0.05, 0.05)):
    """
    Inicializa el transformador.

    Parametros
    ----------
    limits : tuple of float, default=(0.05, 0.05)
      Porcentaje de recorte en las colas inferior y superior.
    """
    self.limits = limits

  def fit(self, X, y=None):
      """
      Calcula y almacena los limites por columna.

      Parametros
      ----------
      X : array-like o DataFrame
        Conjunto de datos de entrada.
      y : Ignorado

      Returns
      -------
      self
        Instancia entrenada.
      """
      X_df = pd.DataFrame(X)
        # Calculamos y guardamos los límites matemáticos AQUÍ (Fase de aprendizaje)
      self.lower_bounds_ = X_df.quantile(self.limits[0])
      self.upper_bounds_ = X_df.quantile(1 - self.limits[1])
      self.columns_ = X_df.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
      return self

  def transform(self, X):
      """
      Aplica el recorte de valores segun los limites aprendidos.

      Parametros
      ----------
      X : array-like o DataFrame
        Conjunto de datos de entrada.

      Returns
      -------
      DataFrame
        Conjunto de datos con valores recortados.
      """
      X_df = pd.DataFrame(X, columns=self.columns_).copy()
      X_df = X_df.astype("float64")
        # Aplicamos los límites guardados
      for col in self.columns_:
        X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

  def get_feature_names_out(self, input_features=None):
      """
      Devuelve nombres de caracteristicas.

      Parametros
      ----------
      input_features : array-like, optional
        Nombres de entrada a devolver.

      Returns
      -------
      ndarray
        Nombres de las columnas.
      """
      if input_features is None:
        return np.array(self.columns_)
      else:
        return np.array(input_features)

# Define función para eliminar duplicados
def tratar_duplicados(X : pd.DataFrame, drop = True):
  """
  Tratamiento de duplicados

  Parámetros
  ----------
  X : DataFrame
    Conjunto de datos.
  drop : bool
    Si se deben eliminar los duplicados.

  Retorna
  -------
  DataFrame
    Conjunto de datos sin duplicados.
  """
  return X.drop_duplicates() if drop else X

def separar_objetivo_features(
  X: pd.DataFrame,
  target: str,
  drop_duplicates: bool = True,
):
  """
  Elimina duplicados y separa objetivo y features.

  Parametros
  ----------
  X : DataFrame
    Conjunto de datos.
  target : str
    Nombre de la variable objetivo.
  drop_duplicates : bool, default=True
    Si True, elimina duplicados antes de separar.

  Returns
  -------
  tuple
    (data_sin_dup, var_dep, var_indep)
  """
  data_sin_dup = tratar_duplicados(X, drop=drop_duplicates).reset_index(drop=True)
  var_dep = data_sin_dup[target].reset_index(drop=True)
  var_indep = data_sin_dup.drop(columns=[target])
  return data_sin_dup, var_dep, var_indep

def eliminar_nulos_objetivo(X: pd.DataFrame, target: str) -> pd.DataFrame:
  """
  Elimina filas con valores nulos en la variable objetivo.

  Parametros
  ----------
  X : DataFrame
    Conjunto de datos.
  target : str
    Nombre de la variable objetivo.

  Returns
  -------
  DataFrame
    Conjunto de datos sin nulos en la variable objetivo.
  """
  return X.dropna(subset=[target]).reset_index(drop=True)

def detectar_inconsistencias(
  X: pd.DataFrame,
  columnas_negativas_check=None,
  columnas_ceros_check=None,
  verbose=True,
):
  """
  Detecta valores negativos o ceros en columnas que deberian ser positivas.

  Parametros
  ----------
  X : DataFrame
    Conjunto de datos.
  columnas_negativas_check : list or None
    Columnas a revisar por valores negativos.
  columnas_ceros_check : list or None
    Columnas a revisar por valores cero.
  verbose : bool
    Si es True, imprime el resumen de inconsistencias.

  Returns
  -------
  tuple
    (conteo_negativos, conteo_ceros)
  """
  if columnas_negativas_check is None:
    columnas_negativas_check = [
      "ingreso_mensual", "gasto_mensual", "deuda_total", "score_crediticio",
      "edad", "antiguedad_meses", "frecuencia_compra", "ultima_compra_dias",
      "num_productos", "hora_registro",
    ]
  if columnas_ceros_check is None:
    columnas_ceros_check = [
      "ingreso_mensual", "gasto_mensual", "deuda_total", "score_crediticio",
      "edad", "antiguedad_meses",
    ]

  conteo_negativos = {}
  conteo_ceros = {}

  for col in columnas_negativas_check:
    if col in X.columns:
      conteo_negativos[col] = int((X[col] < 0).sum())

  for col in columnas_ceros_check:
    if col in X.columns:
      conteo_ceros[col] = int((X[col] == 0).sum())

  conteo_negativos = {k: v for k, v in conteo_negativos.items() if v > 0}
  conteo_ceros = {k: v for k, v in conteo_ceros.items() if v > 0}

  if verbose:
    print("Negativos detectados:", conteo_negativos if conteo_negativos else "ninguno")
    print("Ceros detectados:", conteo_ceros if conteo_ceros else "ninguno")

  return conteo_negativos, conteo_ceros

def corregir_valores_negativos(
  X: pd.DataFrame,
  columns_to_correct=None,
  inplace=False,
  verbose=True,
):
  """
  Corrige valores negativos convirtiendolos a valor absoluto.

  Parametros
  ----------
  X : DataFrame
    Conjunto de datos.
  columns_to_correct : list or None
    Columnas a corregir.
  inplace : bool
    Si True, modifica el DataFrame original.
  verbose : bool
    Si es True, imprime un mensaje de confirmacion.

  Returns
  -------
  DataFrame
    Conjunto de datos con valores negativos corregidos.
  """
  if columns_to_correct is None:
    columns_to_correct = [
      "ingreso_mensual", "gasto_mensual", "deuda_total", "score_crediticio",
      "edad", "antiguedad_meses", "frecuencia_compra", "ultima_compra_dias",
      "num_productos", "hora_registro",
    ]

  X_out = X if inplace else X.copy()

  for col in columns_to_correct:
    if col in X_out.columns:
      X_out[col] = X_out[col].abs()

  if verbose:
    print("Valores negativos corregidos a valores absolutos en las columnas relevantes.")

  return X_out

class FeatureEngineering(BaseEstimator, TransformerMixin):
  """
  Ingeniería de características

  Parámetros
  ----------
  BaseEstimator : Clase base para estimadores en scikit-learn.
  TransformerMixin : Clase base para transformadores en scikit-learn.

  Atributos
  ---------
  columns_ : array-like
    Nombres de las columnas a transformar.
  Returns
  -------
  DataFrame
    Conjunto de datos con nuevas características.
  """
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()

    # razón de endeudamiento
    X["ratio_endeudamiento"] = X["deuda_total"] / X["ingreso_mensual"]

    # porcentaje de gasto respecto al ingreso
    X['porcentaje_gasto'] = X['gasto_mensual'] / X['ingreso_mensual']

    return X

#Pipelines
def build_preprocessor(
    numerical_features,
    categorical_nominales,
    categorical_ordinales,
    orden_tipo_plan,
    orden_uso_app,
):
    """
    Construye pipelines de transformacion y el preprocesador combinado.

    Parameters
    ----------
    numerical_features : list
        Columnas numericas.
    categorical_nominales : list
        Columnas categoricas nominales.
    categorical_ordinales : list
        Columnas categoricas ordinales.
    orden_tipo_plan : list
        Orden para la variable ordinal tipo_plan.
    orden_uso_app : list
        Orden para la variable ordinal uso_app.

    Returns
    -------
    tuple
        (pipeline_numerical_features, pipeline_nominales, pipeline_ordinales, preprocesador)
    """
    pipeline_numerical_features = Pipeline(steps=[
        ("winsorizer", Winsorizer(limits=(0.05, 0.05))),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    pipeline_nominales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pipeline_ordinales = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[orden_tipo_plan, orden_uso_app])),
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ("num_limpios", pipeline_numerical_features, numerical_features),
            ("cat_nom", pipeline_nominales, categorical_nominales),
            ("cat_ord", pipeline_ordinales, categorical_ordinales),
        ],
        remainder="drop",
    )

    return pipeline_numerical_features, pipeline_nominales, pipeline_ordinales, preprocesador

def build_cleaning_pipeline(preprocesador):
    """
    Construye el pipeline de limpieza con duplicados, features y preprocesamiento.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador combinado.

    Returns
    -------
    Pipeline
        Pipeline de limpieza.
    """
    return Pipeline(
        steps=[
            ("duplicados", FunctionTransformer(tratar_duplicados, kw_args={"drop": True})),
            ("feature_engineering", FeatureEngineering()),
            ("preprocesamiento", preprocesador),
            ("conversion", DataFrameConverter(preprocesador)),
        ]
    )

def build_cleaning_pipeline_reg(preprocesador):
    """
    Construye el pipeline de limpieza con duplicados, features y preprocesamiento.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador combinado.

    Returns
    -------
    Pipeline
        Pipeline de limpieza.
    """
    return Pipeline(
        steps=[
            ("duplicados", FunctionTransformer(tratar_duplicados, kw_args={"drop": True})),
            ("preprocesamiento", preprocesador),
            ("conversion", DataFrameConverter(preprocesador)),
        ]
    )

def build_model_pipeline_regresion(modelo, threshold=0.9):
    """
    Construye un pipeline de regresion con filtro de colinealidad.

    Parameters
    ----------
    modelo : BaseEstimator
        Modelo de regresion.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline de regresion.
    """
    return Pipeline(
        steps=[
            ("colinealidad", CorrelationFilter(threshold=threshold)),
            ("modelo", modelo),
        ]
    )


def build_model_pipeline_classifier(modelo, threshold=0.9):
    """
    Construye un pipeline de clasificacion con filtro de colinealidad.

    Parameters
    ----------
    modelo : BaseEstimator
        Modelo de clasificacion.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline de clasificacion.
    """
    return Pipeline(
        steps=[
            ("colinealidad", CorrelationFilter(threshold=threshold)),
            ("modelo", modelo),
        ]
    )

def build_linear_regression_pipeline(threshold=0.9):
    """
    Pipeline para regresion lineal.

    Parameters
    ----------
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline de regresion lineal.
    """
    return build_model_pipeline_regresion(LinearRegression(), threshold=threshold)

def build_decision_tree_regressor_pipeline(
    max_depth=7,
    min_samples_leaf=15,
    random_state=42,
    threshold=0.9,
):
    """
    Pipeline para DecisionTreeRegressor.

    Parameters
    ----------
    max_depth : int, default=7
        Profundidad maxima del arbol.
    min_samples_leaf : int, default=15
        Minimo de muestras por hoja.
    random_state : int, default=42
        Semilla aleatoria.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline con DecisionTreeRegressor.
    """
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return build_model_pipeline_regresion(modelo, threshold=threshold)


def build_logistic_regression_classifier_pipeline(
    max_iter=1000,
    class_weight="balanced",
    threshold=0.9,
):
    """
    Pipeline para regresion logistica.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximo de iteraciones.
    class_weight : str or dict, default="balanced"
        Peso de clases.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline con LogisticRegression.
    """
    modelo = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    return build_model_pipeline_classifier(modelo, threshold=threshold)


def build_decision_tree_classifier_pipeline(
    max_depth=5,
    class_weight="balanced",
    min_samples_leaf=1,
    min_samples_split=2,
    threshold=0.9,
):
    """
    Pipeline para DecisionTreeClassifier.

    Parameters
    ----------
    max_depth : int, default=5
        Profundidad maxima del arbol.
    class_weight : str or dict, default="balanced"
        Peso de clases.
    min_samples_leaf : int, default=1
        Minimo de muestras por hoja.
    min_samples_split : int, default=2
        Minimo de muestras para dividir.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline con DecisionTreeClassifier.
    """
    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
    )
    return build_model_pipeline_classifier(modelo, threshold=threshold)


def build_svm_classifier_pipeline(
    kernel="rbf",
    class_weight="balanced",
    probability=True,
    random_state=42,
    threshold=0.9,
):
    """
    Pipeline para SVM (SVC).

    Parameters
    ----------
    kernel : str, default="rbf"
        Tipo de kernel.
    class_weight : str or dict, default="balanced"
        Peso de clases.
    probability : bool, default=True
        Habilita probabilidades.
    random_state : int, default=42
        Semilla aleatoria.
    threshold : float, default=0.9
        Umbral de correlacion.

    Returns
    -------
    Pipeline
        Pipeline con SVC.
    """
    modelo = SVC(
        kernel=kernel,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
    )
    return build_model_pipeline_classifier(modelo, threshold=threshold)

def aplicar_pipeline_limpieza(pipeline_limpieza, var_indep, var_dep, target_col="deuda_total"):
    """
    Aplica el pipeline de limpieza y devuelve el dataset transformado con objetivo.

    Parameters
    ----------
    pipeline_limpieza : Pipeline
        Pipeline de limpieza entrenable.
    var_indep : DataFrame
        Variables independientes.
    var_dep : Series
        Variable dependiente.
    target_col : str, default="deuda_total"
        Nombre de la columna objetivo en el resultado.

    Returns
    -------
    DataFrame
        Dataset transformado con la columna objetivo.
    """
    data_transformada = pipeline_limpieza.fit_transform(var_indep)
    if isinstance(data_transformada, pd.DataFrame):
        df = data_transformada.copy()
    else:
        df = pd.DataFrame(data_transformada)

    if hasattr(df, "columns"):
        df.columns = df.columns.str.replace("num_limpios__", "", regex=False)
        df.columns = df.columns.str.replace("cat_nom__", "", regex=False)
        df.columns = df.columns.str.replace("cat_ord__", "", regex=False)

    df[target_col] = var_dep.to_numpy()
    return df


def _limpiar_prefijos_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina prefijos generados por el ColumnTransformer en los nombres de columnas.
    """
    if hasattr(df, "columns"):
        df = df.copy()
        df.columns = df.columns.str.replace("num_limpios__", "", regex=False)
        df.columns = df.columns.str.replace("cat_nom__", "", regex=False)
        df.columns = df.columns.str.replace("cat_ord__", "", regex=False)
    return df


def aplicar_pipeline_limpieza_train_test(pipeline_limpieza, X_train, X_test):
    """
    Ajusta el pipeline con train y transforma train/test sin leakage.

    Parameters
    ----------
    pipeline_limpieza : Pipeline
        Pipeline de limpieza entrenable.
    X_train : DataFrame
        Variables independientes de entrenamiento.
    X_test : DataFrame
        Variables independientes de prueba.

    Returns
    -------
    tuple
        (X_train_t, X_test_t) transformados y con columnas normalizadas.
    """
    X_train_t = pipeline_limpieza.fit_transform(X_train)
    X_test_t = pipeline_limpieza.transform(X_test)

    if not isinstance(X_train_t, pd.DataFrame):
        X_train_t = pd.DataFrame(X_train_t)
    if not isinstance(X_test_t, pd.DataFrame):
        X_test_t = pd.DataFrame(X_test_t)

    X_train_t = _limpiar_prefijos_columnas(X_train_t)
    X_test_t = _limpiar_prefijos_columnas(X_test_t)

    return X_train_t, X_test_t