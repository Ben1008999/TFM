# TFM
Códigos Matlab, Python y AWK desarrollados para el Trabajo de Final de Máster con título "Estudio de la predictibilidad del tráfico en Internet para la detección de anomalías sutiles"

El repositorio se divide en tres partes:
1. Códigos AWK para la obtención de series temporales (ejecutar en Linux): en primer lugar, es necesario acceder a la página web de la UGR'16 (https://nesg.ugr.es/nesg-ugr16/index.php) y descargar los ficheros en extensión .csv con los datos de calibración o de test con los que se desee realizar las pruebas (por ejemplo, es posible descargar el fichero March Week#4 de calibración). El archivo descargado serán datos en formato de flujos con extensión tar.gz.
El siguiente paso es descomprimirlo en un directorio cualquiera y llevar y ejecutar en ese mismo directorio el Script tref.sh escrito con AWK que, tras procesar el fichero en formato de flujos, imprime en la terminal el primer segundo (en POSIX) del que se tienen datos (también llamado tref o tiempo de referencia).
Después, se deberá llevar el Script process.sh al mismo directorio donde esté el fichero de flujos descomprimido y el Script tref.sh y abrir dicho Script (process.sh) y modificar la variable tref con el tiempo de referencia que se haya obtenido con tref.sh. Con la variable modificada, se puede ejecutar process.sh, que obtendrá como salida un fichero con extensión .txt con tres columnas: la primera, el segundo. La segunda, el número de bytes transmitidos desde el segundo anterior hasta el segundo actual. La tercera, el número de paquetes transmitidos desde el segundo anterior hasta el segundo actual. En resumen, el fichero BPSyPPS.txt es la serie temporal con el ancho de banda.

Una vez se dispone de las series temporales (.txt) de diferentes datos de calibración, se deberán organizar en directorios distintos, todos ellos dentro de una ruta concreta. Por ejemplo, en la ruta "X" deberán estar los directorios march_week3, march_week4, march_week5... Y cada uno deberá contener su correspondiente fichero BPSyPPS.txt. En esa misma ruta "X" deberá encontrarse el fichero Matlab para la obtención de los datos de entrenamiento para la red LSTM/Holt-Winter's.

Este proceso se debe hacer para las siguientes semanas de tráfico, que son las que se han utilizado en el TFM (y, por tanto, las que se leen en el fichero de Matlab): march_week3, march_week4, march_week5, april_week2, april_week4, may_week1, may_week3, june_week1, june_week2, june_week3, july_week1.

2. Código Matlab para la obtención de la Base de Datos con las dinámicas de la tendencia (fichero TrendDynamics.m): Este fichero tomará todas las series temporales y las organizará semanalmente (de Lunes a Domingo). A continuación, por cada serie temporal, irá deslizando una ventana y, por cada ventana, obtendrá los parámetros theta de la regresión polinómica y los parámetros alpha-estable de cada ventana tras restar a los datos dicha tendencia. El resultado son dos ficheros: TPX_Y.txt (Theta Parameters X = Tamaño de ventana deslizante usado Y = orden polinómico usado) y APX_Y.txt (Alpha Parameters; X e Y significan lo mismo que para TPX_Y.txt). Estos ficheros almacenan la información con el siguiente formato:

Parámetros theta:
- [theta0 theta1 theta2...] semana 1 ventana 1 | [theta0 theta1 theta2...] semana 2 ventana 1 | ... | [theta0 theta1 theta2...] semana M ventana 1
- [theta0 theta1 theta2...] semana 1 ventana 2 | [theta0 theta1 theta2...] semana 2 ventana 2 | ... | [theta0 theta1 theta2...] semana M ventana 2
- [theta0 theta1 theta2...] semana 1 ventana 3 | [theta0 theta1 theta2...] semana 2 ventana 3 | ... | [theta0 theta1 theta2...] semana M ventana 3
- ...
- ...
- [theta0 theta1 theta2...] semana 1 ventana N | [theta0 theta1 theta2...] semana 2 ventana N | ... | [theta0 theta1 theta2...] semana M ventana N
   
Por tanto, las variables de entrada del fichero TrendDynamics.m son:
- computeParams (0 o 1) para decidir si leer los ficheros TP y AP de un .txt ya existente (computeParams = 0) o bien computarlos y escribirlos (computeParams = 1)
- TPfilename y APfilename: Nombres de los ficheros TP y AP de los que leer los datos.
- Tventana [min]: (Tamaño en minutos de la ventana deslizante T)
- n: Grado para la regresión polinómica
- Granularidad_deteccion: es el alcance del sistema, que se debe conocer para el dominio en el que realizar la regresión polinómica

Las salidas del fichero TrendDynamics.m son los ficheros TP y APX_Y.txt. Adicionalmente, se escribirá en un fichero "All_series.txt" todas las series temporales en forma de matriz (lo que sería la matriz 'agregado' en TrendDynamics.m) ordenadas semanalmente.

3. Código Python para la programación de la red LSTM y para la predicción de la tendencia. Una vez se dispone de los ficheros de entrenamiento para la red LSTM (TPX_Y.txt) y de las propias series temprales All_series.txt, se puede ejecutar el código mainThreads_multistep.py para simular la predicción sobre una ventana de test. Los inputs del fichero son los siguientes:
- Tsventana: tamaño de ventana en segundos que se usó para sacar los coeficientes y parámetros alpha-stable en TrendDynamics.m
- n: grado de la regresión polinómica que se usó cuando se obtuvieron estos parámetros theta en TrendDynamics.m
- timesteps_future: puntos futuros de la predicción de los parámetros theta_i (sería k_steps,future)
- timesteps_future_recurrent: puntos futuros de la predicción de los parámetros theta_i cuando se realiza la predicción recurrente (por ejemplo, se pueden predecir los timesteps_future = 10 puntos futuros de 1, en 1, 2 en 2, de 5 en 5... De forma realimentada, usando cada predicción como input para la siguiente predicción. Por ejemplo, si se quieren predecir 10 puntos de 2 en 2, entonces timesteps_future_recurrent = 2).
- Diezmado: período de muestreo (Delta) en segundos para realizar el diezmado de las series temporales theta_i
- recurrent_forecast: Indica si se desea realizar la predicción de forma recurrente. 0: no se desea predicción recurrente. 1: se desea predicción recurrente
- normalization: tipo de normalización que se usará para la predicción de las series temporales theta_i. 0: MinMax. 1: tanh. 2: zscore
- CNN: indica si se desea utilizar la LSTM con capa convolucional o no. 1: CNN + LSTM. 0: LSTM simple, sin capa convolucional.
- velocity: indica si se desea trabajar con la derivada de las series temporales theta_i para la predicción (FUNCIÓN NO IMPLEMENTADA: NO USAR). El código no soporta esta función actualmente, aunque se han realizado pruebas intermedias). Poner a 0.
- filename_thetaParams: nombre del fichero en el que se encuentran los parámetros theta (TPX_Y.txt).
- filename_network_traffic_series: nombre del fichero en el que se encuentran las series temporales organizadas semanalmente (All_series.txt)
- semana: número de semana que se desea utilizar como test. En las pruebas realizadas para el Trabajo, cada índice representa lo siguiente:

0 = march_week3

1 = march_week4

2 = march_week5

3 = april_week2

4 = april_week4

5 = may_week1

6 = may_week3

7 = june_week1

8 = june_week2

9 = june_week3

10 = july_week1


