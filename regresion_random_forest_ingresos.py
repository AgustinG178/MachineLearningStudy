import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generar un Dataset Sintético
np.random.seed(42) # Para reproducibilidad

# Número de muestras
n_samples = 500

# Generar características
# Edad (entre 22 y 65 años)
edad = np.random.randint(22, 65, n_samples)

# Años de Educación (después de la secundaria, entre 0 y 10 años)
anos_educacion = np.random.randint(0, 11, n_samples)

# Horas de Trabajo por Semana (entre 20 y 60 horas)
horas_trabajo_semana = np.random.randint(20, 61, n_samples)

# Puntuación en una Habilidad Específica (escala 1-100)
puntuacion_habilidad = np.random.randint(1, 101, n_samples)

# Nivel de Experiencia en el Sector (años, entre 0 y 40, relacionado con la edad)
# Aseguramos que la experiencia no sea mayor que edad - 22 (edad mínima de inicio laboral)
max_experiencia = np.clip(edad - 22, 0, 40) # Límite superior de experiencia
anos_experiencia_sector = np.array([np.random.randint(0, max_exp + 1 if max_exp >= 0 else 1) for max_exp in max_experiencia])


# Generar Ingreso Anual Estimado (variable objetivo)
# Introducimos no linealidades y relaciones más complejas
# Base del ingreso
ingreso = 15000  # Ingreso base

# Contribución de la educación (crece más con más años)
ingreso += anos_educacion * 2000 + (anos_educacion**2) * 150

# Contribución de la experiencia (efecto cuadrático, disminuye después de cierto punto)
ingreso += anos_experiencia_sector * 1000 - (anos_experiencia_sector - 20)**2 * 20

# Contribución de las horas de trabajo
ingreso += horas_trabajo_semana * 300

# Contribución de la habilidad (efecto multiplicador)
ingreso *= (1 + puntuacion_habilidad / 200)

# Contribución de la edad (efecto no lineal, un ligero aumento y luego estabilización)
# Usamos una función sigmoide escalada para modelar esto
ingreso_edad_factor = 1 + 0.5 / (1 + np.exp(-(edad - 40) / 5))
ingreso *= ingreso_edad_factor


# Interacción entre educación y experiencia
ingreso += anos_educacion * anos_experiencia_sector * 50

# Añadir ruido gaussiano para simular variabilidad no explicada
ruido = np.random.normal(0, 15000, n_samples) # Desviación estándar del ruido
ingreso_anual_estimado = ingreso + ruido

# Asegurarse de que el ingreso no sea negativo
ingreso_anual_estimado = np.maximum(ingreso_anual_estimado, 10000) # Ingreso mínimo

# Crear DataFrame
data = pd.DataFrame({
    'Edad': edad,
    'Anos_Educacion': anos_educacion,
    'Horas_Trabajo_Semana': horas_trabajo_semana,
    'Puntuacion_Habilidad': puntuacion_habilidad,
    'Anos_Experiencia_Sector': anos_experiencia_sector,
    'Ingreso_Anual_Estimado': ingreso_anual_estimado.astype(int) # Convertir a entero
})

# Guardar en CSV
csv_filename = 'datos_ingresos_sinteticos.csv'
data.to_csv(csv_filename, index=False)

print(f"Dataset sintético generado y guardado como '{csv_filename}'")
print("Primeras filas del dataset generado:")
print(data.head())

# 2. Cargar y Explorar los Datos
# Cargar el CSV (aunque ya lo tenemos en 'data', esto simula un flujo normal)
df = pd.read_csv(csv_filename)

print("\nInformación del DataFrame cargado:")
df.info()

print("\nEstadísticas descriptivas del DataFrame:")
print(df.describe())

# Visualizaciones para exploración
# Pairplot para ver relaciones bivariadas y distribuciones
# Tomará un momento en datasets grandes, pero con 500 muestras es manejable.
print("\nGenerando pairplot (puede tardar un momento)...")
# sns.pairplot(df, diag_kind='kde')
# plt.suptitle('Pairplot de Características e Ingreso', y=1.02)
# plt.savefig('pairplot_ingresos.png')
# print("Pairplot guardado como 'pairplot_ingresos.png'")
# plt.close() # Cerrar la figura para no mostrarla si se ejecuta en un entorno sin GUI

# Scatter plots individuales del ingreso contra cada característica
features = ['Edad', 'Anos_Educacion', 'Horas_Trabajo_Semana', 'Puntuacion_Habilidad', 'Anos_Experiencia_Sector']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=df[feature], y=df['Ingreso_Anual_Estimado'])
    plt.title(f'Ingreso vs {feature}')
plt.tight_layout()
plt.savefig('scatter_plots_ingresos.png')
print("Scatter plots individuales guardados como 'scatter_plots_ingresos.png'")
plt.close()

# Correlación heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlación')
plt.savefig('correlation_heatmap_ingresos.png')
print("Heatmap de correlación guardado como 'correlation_heatmap_ingresos.png'")
plt.close()

# 3. Preprocesar los Datos
# Seleccionar características (X) y variable objetivo (y)
X = df[features] # features fue definido antes: ['Edad', 'Anos_Educacion', ...]
y = df['Ingreso_Anual_Estimado']

# Dividir los datos en conjuntos de entrenamiento y prueba
# test_size=0.2 indica que el 20% de los datos se usarán para el conjunto de prueba
# random_state asegura la reproducibilidad de la división
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nForma de los conjuntos de datos:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Nota: Para Random Forest, el escalado de características no es estrictamente necesario
# ya que es un modelo basado en árboles y no es sensible a la magnitud de las características.
# Si usáramos SVR o Regresión Lineal con regularización, el escalado sería importante.

# 4. Crear y Entrenar el Modelo (Random Forest Regressor)
# Crear una instancia del modelo Random Forest Regressor
# n_estimators: número de árboles en el bosque.
# random_state: para reproducibilidad.
# n_jobs=-1: usa todos los procesadores disponibles para el entrenamiento (más rápido).
# max_depth: profundidad máxima de los árboles. Ayuda a prevenir el overfitting.
# min_samples_split: número mínimo de muestras requeridas para dividir un nodo interno.
# min_samples_leaf: número mínimo de muestras requeridas en un nodo hoja.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,
                                 max_depth=10, min_samples_split=4, min_samples_leaf=2)

# Entrenar el modelo con los datos de entrenamiento
print("\nEntrenando el modelo Random Forest Regressor...")
rf_model.fit(X_train, y_train)
print("Modelo entrenado.")

# (Opcional) Se podría realizar una búsqueda de hiperparámetros (GridSearchCV o RandomizedSearchCV)
# para encontrar la mejor combinación de hiperparámetros, pero para este ejemplo, usamos valores razonables.

# 5. Realizar Predicciones y Evaluar el Modelo
# Realizar predicciones sobre el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo en el conjunto de prueba:")
print(f"Error Cuadrático Medio (MSE): {mse:,.2f}") # Formateado para mejor lectura
print(f"Coeficiente de Determinación (R^2): {r2:.4f}")

# Visualizar la importancia de las características
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Importancia de las Características (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance_ingresos.png')
print("Gráfico de importancia de características guardado como 'feature_importance_ingresos.png'")
plt.close()

# Graficar valores reales vs. predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Línea de y=x
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs. Predicciones (Random Forest)')
plt.grid(True)
plt.tight_layout()
plt.savefig('real_vs_predicted_ingresos.png')
print("Gráfico de reales vs. predichos guardado como 'real_vs_predicted_ingresos.png'")
plt.close()
