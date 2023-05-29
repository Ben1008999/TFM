#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"
#Documentación de cómo extraer una variable dentro de un bloque awk a la shell: https://stackoverflow.com/questions/9708028/awk-return-value-to-shell-script
read referencia_tiempo <<< $(awk -F '[-:,]' 'BEGIN {c = 0}{
	if(c != 0){
		if($1 " " $2 " " $3 " " $4 " " $5 != tfecha_was){
			tf = mktime($1 " " $2 " " $3 " " $4 " " $5)
		}
		
		if($6 == 0){
			ti_is = tf - 1
		}
		else{
			ti_is = int(tf - $6)
		}
		
		if(ti_is < ti_first){
			ti_first = ti_is
		}
		tfecha_was = $1 " " $2 " " $3 " " $4 " " $5
	}
	else{
		if($1 > 0){
			tfecha_was = $1 " " $2 " " $3 " " $4 " " $5
			tf = mktime(tfecha_was)
			
			if($6 == 0){
				ti_first = tf - 1
			}
			else{
				ti_first = int(tf - $6)
			}
			c = 1
		}
	}
}END{print int(sprintf("%d", ti_first))}' april.week4.csv.uniqblacklistremoved)

echo $referencia_tiempo #Para hacer un print de la variable de referencia_tiempo
#Resultado: referencia_tiempo = 1462744732
