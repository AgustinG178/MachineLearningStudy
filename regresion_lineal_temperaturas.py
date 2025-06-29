# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Cargar y explorar los datos
# Cargar el archivo CSV en un DataFrame de pandas
df = pd.read_csv('temperaturas.csv')

# Mostrar las primeras filas del DataFrame para entender su estructura
print("Primeras filas del DataFrame:")
print(df.head())

# Obtener información básica sobre los datos
print("\nInformación del DataFrame:")
df.info()

# Mostrar estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Renombrar columnas para que sean más descriptivas si es necesario
# En este caso, los nombres 'celsius' y 'fahrenheit' son claros.
# Si fueran, por ejemplo, 'col1' y 'col2', podríamos hacer:
# df.rename(columns={'col1': 'Year', 'col2': 'Temperature'}, inplace=True)

# Para este ejemplo, vamos a predecir 'fahrenheit' a partir de 'celsius'.
# X serán los grados Celsius (nuestra característica o variable independiente)
# y serán los grados Fahrenheit (nuestra variable objetivo o dependiente)

# 2. Preprocesar los datos
# Seleccionar las características (X) y la variable objetivo (y)
# Usamos df[['celsius']] para mantener X como un DataFrame 2D, requerido por scikit-learn
X = df[['celsius']]
y = df['fahrenheit']

# Dividir los datos en conjuntos de entrenamiento y prueba
# test_size=0.2 significa que el 20% de los datos se usarán para pruebas
# random_state se usa para asegurar que la división sea la misma cada vez que se ejecuta el código
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nForma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)

# 3. Crear y entrenar el modelo de regresión lineal
# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo utilizando los datos de entrenamiento (X_train, y_train)
# El método fit() ajusta el modelo a los datos
model.fit(X_train, y_train)

# Una vez entrenado, el modelo ha aprendido los coeficientes de la regresión.
# El coeficiente (pendiente) se almacena en model.coef_
# La intercepción (ordenada al origen) se almacena en model.intercept_
print("\nModelo entrenado.")
print(f"Coeficiente (pendiente): {model.coef_[0]}")
print(f"Intercepción (ordenada al origen): {model.intercept_}")

# 4. Realizar predicciones
# Utilizar el modelo entrenado para hacer predicciones sobre el conjunto de prueba (X_test)
y_pred = model.predict(X_test)

# y_pred ahora contiene las temperaturas en Fahrenheit predichas por el modelo
# para las temperaturas en Celsius dadas en X_test.
print("\nPredicciones realizadas sobre el conjunto de prueba.")
# Mostramos algunas predicciones junto con los valores reales para comparar
print("Primeras 5 predicciones vs valores reales:")
for i in range(min(5, len(y_test))): # Asegurarse de no exceder el tamaño de y_test
    print(f"Predicho: {y_pred[i]:.2f}, Real: {y_test.iloc[i]:.2f}")

# 5. Evaluar el modelo
# Calcular métricas de evaluación.
# Error Cuadrático Medio (MSE): Promedio de los errores al cuadrado.
# Un valor más bajo indica un mejor ajuste.
mse = mean_squared_error(y_test, y_pred)
print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")

# Coeficiente de Determinación (R^2): Proporción de la varianza en la variable dependiente
# que es predecible a partir de la variable independiente.
# Varía entre 0 y 1, donde 1 indica un ajuste perfecto.
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

# Visualizar los resultados
# Graficar los puntos de datos reales del conjunto de prueba
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Valores Reales')
# Graficar la línea de regresión (predicciones)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicciones (Línea de Regresión)')
plt.title('Regresión Lineal: Celsius vs Fahrenheit (Conjunto de Prueba)')
plt.xlabel('Temperatura en Celsius (°C)')
plt.ylabel('Temperatura en Fahrenheit (°F)')
plt.legend()
plt.grid(True)
# Guardar el gráfico en un archivo
plt.savefig('regresion_lineal_prediccion.png')
print("\nGráfico de regresión guardado como 'regresion_lineal_prediccion.png'")
# plt.show() # Descomentar si se ejecuta en un entorno con capacidad gráfica interactiva
