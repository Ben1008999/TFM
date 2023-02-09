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
