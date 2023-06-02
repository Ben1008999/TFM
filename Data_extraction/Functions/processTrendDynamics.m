function [theta_params, alpha_params] = processTrendDynamics(agregado, Tsventana, n, NTotalWindows, domainFIT)
    if(NTotalWindows == -1)
        NTotalWindows = size(data,2) - Tsventana + 1;
    end
    theta_params = cell(NTotalWindows, size(agregado,1)-1);
    alpha_params = cell(NTotalWindows, size(agregado,1)-1);
    for i=1:NTotalWindows
        WNormal = agregado(2:end, i:i+Tsventana-1);
        for j=1:size(WNormal,1) %Por cada serie temporal
            if(sum(isnan(WNormal(j,:))) >= 1) %Si hay 1 NaN o más no se puede hacer fit
                theta_params{i,j} = NaN*ones(1, n+1);
                alpha_params{i,j} = NaN*ones(1, 4);
            else
                %Parámetros theta:
                regressiontype = strcat('poly', string(n));
                h = fit(domainFIT, WNormal(j,:)', regressiontype);
                h_accurate = fit([1:Tsventana]', WNormal(j,:)', 'poly9'); %Para los alpha stables siempre usamos regresión de orden 9!
                theta_params{i,j} = flip(coeffvalues(h));
                %Parámetros alpha:
                Xt = WNormal(j,:)-h_accurate(1:Tsventana)';
                try
                    Parametros_AlphaStable = fitdist(Xt', 'Stable');
                    alpha = Parametros_AlphaStable.alpha;
                    beta = Parametros_AlphaStable.beta;
                    gamma = Parametros_AlphaStable.gam;
                    delta = Parametros_AlphaStable.delta;
                catch e
                    alpha = NaN; beta = NaN; gamma = NaN; delta = NaN;
                end
                alpha_params{i,j} = [alpha, beta, gamma, delta];
            end
        end
        fprintf("Computing params... %.3f [%%]\n", i*100/NTotalWindows);
    end
end