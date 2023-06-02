function [outputJSON] = buildJSONinfo(Tventana, n, Granularidad_deteccion, bitsPaquetes, NTotalWindows, agregado, labels, theta_params, alpha_params, domainFIT)
    %Create a JSON structure to store output data:
    outputJSON = struct(labels{1}(1), agregado(1,:));
    
    outputJSON.("Tsventana") = Tventana*60;
    outputJSON.("n") = n;
    outputJSON.("Scope") = Granularidad_deteccion;
    outputJSON.("BitsOrPackets") = bitsPaquetes;
    outputJSON.("Number_of_simulated_windows") = NTotalWindows;
    outputJSON.("domainFIT") = domainFIT';
    for i=1:length(labels{1})
        outputJSON.(labels{1}(i)) = agregado(i,:);
    end
    thetas = cell2mat(theta_params);
    alphas = cell2mat(alpha_params);
    outputJSON.(strcat(strcat(strcat("TP", string(Tventana)), '_'), string(n))) = thetas;
    outputJSON.(strcat(strcat(strcat("AP", string(Tventana)), '_'), string(n))) = alphas;
    %Formato de nombre:
    %APX_Y.txt
    %   X = Tamaño de ventana usado (min)
    %   Y = Orden de la regresión polinómica usado
end