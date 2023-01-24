# TFM
Códigos Matlab, Python y AWK desarrollados para el Trabajo de Final de Máster con título "Estudio de la predictibilidad del tráfico en Internet para la detección de anomalías sutiles"

El repositorio se divide en tres partes:
1. Códigos AWK para la obtención de series temporales (ejecutar en Linux): en primer lugar, es necesario acceder a la página web de la UGR'16 (https://nesg.ugr.es/nesg-ugr16/index.php) y descargar los ficheros en extensión .csv con los datos de calibración o de test con los que se desee realizar las pruebas (por ejemplo, es posible descargar el fichero March Week#4 de calibración). El archivo descargado serán datos en formato de flujos con extensión tar.gz.
El siguiente paso es descomprimirlo en un directorio cualquiera y llevar y ejecutar en ese mismo directorio el Script tref.sh escrito con AWK que, tras procesar el fichero en formato de flujos, imprime en la terminal el primer segundo (en POSIX) del que se tienen datos (también llamado tref o tiempo de referencia).
Después, se deberá llevar el Script process.sh al mismo directorio donde esté el fichero de flujos descomprimido y el Script tref.sh y abrir dicho Script (process.sh) y modificar la variable tref con el tiempo de referencia que se haya obtenido con tref.sh. Con la variable modificada, se puede ejecutar process.sh, que obtendrá como salida un fichero con extensión .txt con tres columnas: la primera, el segundo. La segunda, el número de bytes transmitidos desde el segundo anterior hasta el segundo actual. La tercera, el número de paquetes transmitidos desde el segundo anterior hasta el segundo actual. En resumen, se el fichero BPSyPPS.txt es la serie temporal con el ancho de banda.
En este repositorio se tiene como ejemplo 

2. Código Matlab para la obtención de la Base de Datos con las dinámicas de la tendencia (fichero 
