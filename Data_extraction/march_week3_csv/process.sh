export LC_NUMERIC=en_US.UTF-8
awk -F '[-:,]' -v tref=1458294472 '{
	tf_flujo = mktime($1 " " $2 " " $3 " " $4 " " $5)-tref
	dur_flujo = $6
	
	npaquetes = $15
	nbits = $16*8
	if(dur_flujo >= 0 && dur_flujo <= 1){
		bps[tf_flujo] += nbits
		pps[tf_flujo] += npaquetes
	}
	else{
		ti_flujo_eff = tf_flujo - dur_flujo
		ti_flujo = int(sprintf("%d", ti_flujo_eff))
		for(i = ti_flujo + 1; i <= tf_flujo; i++){
			if(i == ti_flujo + 1){
				bps[i] += (1 - (ti_flujo_eff - ti_flujo))*nbits/dur_flujo
				pps[i] += (1 - (ti_flujo_eff - ti_flujo))*npaquetes/dur_flujo
			}
			else{
				bps[i] += nbits/dur_flujo
				pps[i] += npaquetes/dur_flujo
			}
		}
	}
}END{for(i=1; i<=length(bps); i++){printf("%d, %.9f, %.9f\n", i+tref, bps[i], pps[i]);}}' march.week3.csv.uniqblacklistremoved > BPSyPPS.txt
