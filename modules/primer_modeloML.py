import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar el conjunto de datos para entrenar el modelo
datos = pd.read_csv("temperaturas.csv")
datos.info()
datos.head()

plt.scatter(x="celsius", y="fahrenheit", data=datos, color="blue", label="Datos")
plt.show()

#Caracteristicas (X), etiqueta (y)
X = datos["celsius"]
y = datos["fahrenheit"]

#transformar los datos a un formato adecuado para el modelo
#reshape los datos para que sklearn pueda procesarlos
X_procesada = X.values.reshape(-1,1)
y_procesada = y.values.reshape(-1,1)

modelo = LinearRegression()

modelo.fit(X_procesada, y_procesada)

celsus = 0

while celsus != -999:
    
    celsius = float(input("Ingrese la temperatura en grados Celsius: "))
    prediccion = modelo.predict([[celsius]])
    print(f"{celsius} grados celsius son {prediccion} grados fahrenheit")

    modelo.score(X_procesada, y_procesada)
    
    if celsus == -999:
        print("terminando de predecir...")
        break
        
        
        