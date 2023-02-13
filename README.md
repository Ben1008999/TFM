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

3. Código Python para la programación de la red LSTM y para la predicción de la tendencia. El código desarrollado está diseñado para 8 redes LSTM (grado polinómico n = 7), pero se puede aplicar lo mismo para órdenes superiores o inferiores. Una vez se dispone de los ficheros de entrenamiento para la red LSTM (TPX_Y.txt) y de las propias series temprales All_series.txt, se puede ejecutar el código mainThreads_multistep.py para simular la predicción sobre una ventana de test (este código requiere utilizar algunas funciones definidas en dataNormalization.py). Los inputs del fichero son los siguientes:
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
- semana: número o índice de semana que se desea utilizar como test (se usa para acceder a esa semana en la matriz con el agregado del tráfico semanal). En las pruebas realizadas para el Trabajo, cada índice representa lo siguiente (no obstante, dependerá de qué semanas haya descargado el usuario):

semana 0 = march_week3

semana 1 = march_week4

semana 2 = march_week5

semana 3 = april_week2

semana 4 = april_week4

semana 5 = may_week1

semana 6 = may_week3

semana 7 = june_week1

semana 8 = june_week2

semana 9 = june_week3

semana 10 = july_week1

- tiempo_final: último instante de tiempo (o segundo) incluido del que se conocen datos de la serie temporal theta_i para llevar a cabo la simulación de nuevos parámetros theta_i. Por ejemplo, si tiempo_final = 15901, quiere decir que se tomará la serie temporal con la evolución de los parámetros theta_i hasta el segundo 15901 incluido (en otras palabras, la última ventana conocida de la que se tienen datos con un sliding de 1s sería la 15901). Si por ejemplo tiempo_final = 1, quiere decir que el código asume que conoce la primera ventana (la 0, que va desde el Lunes [00:00:00] hasta el Lunes [00:00:00] + Tsventana -1) y la segunda (la 1, que va desde el Lunes [00:00:00] + Tsventana +1s hasta el [00:00:00] + Tsventana o, lo que es lo mismo, solo se tienen datos de los coeficientes theta de esa ventana 0 (theta_i_0) y de esa ventana 1 (theta_i_1), es decir, solo se tienen 2 puntos en las series temporales theta_i para el entrenamiento).
NOTA: El tiempo final de simulación podría no ser compatible con el diezmado. Si el diezmado es de 180s, quiere decir que las series temporales theta_i se muestrean cada 180s, lo que implica que solo se tienen los theta_i de la ventana 0 (Lunes [00:00:00] a Lunes [00:00:00] + Tsventana-1), de la ventana 180 (Lunes [00:00:00]+180 a Lunes [00:00:00] + Tsventana-1 + 180), de la 360 (Lunes [00:00:00]+360 a Lunes [00:00:00] + Tsventana-1 + 360) y así sucesivamente. Por tanto, por ejemplo, la ventana 145 no se podría simular; en cambio, el código arregla automáticamente el tiempo final para que sea múltiplo del diezmado mediante el siguiente trozo de código:

final = round(tiempo_final/diezmado); #Punto final
tiempo_final = final*diezmado;

- time_past0: tiempo en segundos de memoria de la red LSTM. Por ejemplo, si la red LSTM tiene una memoria de 60 segundos, quiere decir que tomará en cuenta la evolución de las series theta_i en los 60s anteriores para obtener la predicción. Si se establece un diezmado de 2s en ese caso, entonces equivalentemente la red tomaría los 60/2 = 30 puntos anteriores para la predicción.
- T_train0: tamaño en segundos de la ventana de entrenamiento. Si por ejemplo T_train0 = 540000 segundos y el diezmado es de 180s, entonces quiere decir que la ventana de entrenamiento tiene 540000/180 = 3000 puntos.
- time_neighbour_points0: número de puntos vecinos para la convolución (en caso de usar capa CNN).
- epoch: número de épocas de entrenamiento para las redes LSTM. Dado que cada serie theta_i podría requerir distinto entrenamiento, se puede configurar este valor por cada red LSTM en el cuarto argumento del siguiente trozo de código que refleja cada hilo de entrenamiento de cada LSTM:

my_thread0 = threading.Thread(target=trainLSTM, args=(history_list, X0, y0, epoch, 0, in_seq0_test_norm, in_seq0_truth_norm, model0, 0))

my_thread1 = threading.Thread(target=trainLSTM, args=(history_list, X1, y1, epoch, 1, in_seq1_test_norm, in_seq1_truth_norm, model1, 1))

my_thread2 = threading.Thread(target=trainLSTM, args=(history_list, X2, y2, epoch, 0, in_seq2_test_norm, in_seq2_truth_norm, model2, 2))

my_thread3 = threading.Thread(target=trainLSTM, args=(history_list, X3, y3, epoch, 0, in_seq3_test_norm, in_seq3_truth_norm, model3, 3))

my_thread4 = threading.Thread(target=trainLSTM, args=(history_list, X4, y4, epoch, 0, in_seq4_test_norm, in_seq4_truth_norm, model4, 4))

my_thread5 = threading.Thread(target=trainLSTM, args=(history_list, X5, y5, epoch, 0, in_seq5_test_norm, in_seq5_truth_norm, model5, 5))

my_thread6 = threading.Thread(target=trainLSTM, args=(history_list, X6, y6, epoch, 0, in_seq6_test_norm, in_seq6_truth_norm, model6, 6))

my_thread7 = threading.Thread(target=trainLSTM, args=(history_list, X7, y7, epoch, 0, in_seq7_test_norm, in_seq7_truth_norm, model7, 7))

El quinto argumento se refiere a la verbosidad (0: si no se quiere mostrar la barra de progreso del entrenamiento, 1: en caso contrario). Se puede mostrar el progreso para cualquiera de las redes LSTM.


NOTA: El código cuenta con algunas pruebas realizadas con PCA comentadas (para reducción de dimensionalidad en los coeficientes theta de los polinomios) y otros trozos de código necesarios para obtener Figuras intermedias para la memoria.

La salida del fichero Python con el entrenamiento puede ser importada en Matlab para ejecutar la etapa final: la detección y el cómputo del cociente de Funciones Características. Para ello, se deberá utilizar el fichero resultadosFinalesAlgortimo.m.
Por otro lado, en este repositorio también se encuentra un Notebook de Jupyter con el código de prueba utilizado para el modelo Holt-Winters. Este código utiliza la Base de Datos TPX_Y.txt y "All_series.txt".
