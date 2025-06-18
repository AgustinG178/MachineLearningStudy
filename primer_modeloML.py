import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression

# Cargar el conjunto de datos para entrenar el modelo
datos = pd.read_csv("temperaturas.csv")
datos.info()
datos.head()

sb.scatterplot(x="celsius", y="fahrenheit", data=datos)

#Caracteristicas (X), etiqueta (y)
X = datos["celsius"]
y = datos["fahrenheit"]

X_procesada = X.values.reshape(-1,1)
y_procesada = y.values.reshape(-1,1)


modelo = LinearRegression()

modelo.fit(X_procesada, y_procesada)

celsius = 0

while celsius != -999:
    
    celsius = float(input("Ingrese la temperatura en grados Celsius: "))
    prediccion = modelo.predict([[celsius]])
    print(f"{celsius} grados celsius son {prediccion} grados fahrenheit")

    modelo.score(X_procesada, y_procesada)
    
    if celsius == -999:
        print("terminando de predecir...")
        break
        
        
        