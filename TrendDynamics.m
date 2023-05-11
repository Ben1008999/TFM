%Obtener parámetros alpha-stable y coeficientes de regresión polinómica
%para ventanas de Tventana minutos que se deslizan cada segundo:
clear all; close all; clc; warning off

%PARAMETROS DE ENTRADA:----------------------------------------------------
computeParams = 1; %Indica si se desea computar los parámetros alpha-estable y los coeficientes del polinomio (1) o bien si se desea leer dichos parámetros de un fichero de texto (0)
TPfilename = ""; APfilename = ""; %Nombres de los ficheros (si existieran) en caso de que se desee leer los parámetros alpha-estable y los coeficientes del polinomio de un fichero de texto ya existente
Tventana = 29; %[min] (Tamaño de ventana deslizante T)
n = 6; %Grado de la regresión polinómica
Granularidad_deteccion = 180; %= scope del sistema (alcance o tiempo de incertidumbre de predicción)
bitsPaquetes = 3; %Indica si trabajar con bits/s (2) o packets/s (3)
%--------------------------------------------------------------------------
%Obtener la matriz con todas las series temporales de cada semana
%NOTA: Dado que todas las series se ordenan semanalmente, algunas
%no tienen datos para ciertos días de la semana (por ejemplo, tal vez la
%semana 'X' del mes 'Y' no tenga datos para el Lunes y el Martes). Por esta
%razón, es posible ver valores NaN al comienzo en las bases de datos de los
%parámetros theta y alpha.
%LA PRIMERA FILA DE LA MATRIZ DE AGREGADO ES EL DOMINIO:
domain = 1:7*24*60*60; %[1 = Lunes 00:00:01 -> 7*24*60*60 = Lunes (semana siguiente) 00:00:00]
march_week3 = load('./march_week3_csv/BPSyPPS.txt');
march_week4 = load('./march_week4_csv/BPSyPPS.txt');
march_week5 = load('./march_week5_csv/BPSyPPS.txt');
april_week2 = load('./april_week2_csv/BPSyPPS.txt');
april_week3 = load('./april_week3_csv/BPSyPPS.txt');
april_week4 = load('./april_week4_csv/BPSyPPS.txt');
may_week1 = load('./may_week1_csv/BPSyPPS.txt');
may_week3 = load('./may_week3_csv/BPSyPPS.txt');
june_week1 = load('./june_week1_csv/BPSyPPS.txt');
june_week2 = load('./june_week2_csv/BPSyPPS.txt');
june_week3 = load('./june_week3_csv/BPSyPPS.txt');
july_week1 = load('./july_week1_csv/BPSyPPS.txt');

agregado = zeros(1, length(domain));
agregado(1,:) = domain;
agregado = addTimeSeriesToWeek(march_week3, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(march_week4, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(march_week5, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(april_week2, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(april_week4, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(may_week1, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(may_week3, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(june_week1, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(june_week2, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(june_week3, agregado, bitsPaquetes);
agregado = addTimeSeriesToWeek(july_week1, agregado, bitsPaquetes);
agregado(find(agregado <= 0)) = NaN;

%Obtener series temporales con las dinámicas de la tendencia polinómica:
Tsventana = Tventana*60;
NTotalWindows = size(agregado,2) - Tsventana + 1;
NTotalWindows = 90000; %Si se quiere usar un número de ventanas concreto y no esperar a que se procese toda la serie completa (tarda días)
%Definir el dominio de la regresión:
domainFIT = [[-(Tsventana-1):0] + ceil(Granularidad_deteccion/2)]';
stepdomainFIT = 1/(2*domainFIT(end));
domainFIT = domainFIT*stepdomainFIT;

if(computeParams == 1)
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

    %Exportarlo a txt:
    fileNameOutput_TP = strcat(strcat(strcat(strcat("TP", string(Tventana)), '_'), string(n)), ".txt");
    writecell(theta_params(1:i-1, :), fileNameOutput_TP);
    fileNameOutput_AP = strcat(strcat(strcat(strcat("AP", string(Tventana)), '_'), string(n)), ".txt");
    writecell(alpha_params(1:i-1, :), fileNameOutput_AP);
    %Formato de nombre de archivo:
    %APX_Y.txt
    %   X = Tamaño de ventana usado (min)
    %   Y = Orden de la regresión polinómica usado
    thetas = cell2mat(theta_params);
    alphas = cell2mat(alpha_params);
else
    %%Leer los parámetros y representarlos:
    thetas = cell2mat(readcell(TPfilename));
    alphas = cell2mat(readcell(APfilename));
    NTotalWindows = size(thetas,1);
end
%Formato de almacenamiento de los datos:
%Parámetros theta:
%   [theta0 theta1 theta2...] serie 1 ventana 1 | [theta0 theta1 theta2...] serie 2 ventana 1
%   [theta0 theta1 theta2...] serie 1 ventana 2 | [theta0 theta1 theta2...] serie 2 ventana 2
%   [theta0 theta1 theta2...] serie 1 ventana 3 | [theta0 theta1 theta2...] serie 2 ventana 3
%   ...
%   ...
%   [theta0 theta1 theta2...] serie 1 ventana N | [theta0 theta1 theta2...]
%   serie 2 ventana N
%Lo mismo con los parámetros alpha

%Representación:
for c=1:n+1 %Por cada coeficiente:
    figure;
    for i=1:size(thetas,2)/(n+1) %Sacaremos size(thetas,2)/(n+1) gráficas, una por cada serie temporal
        %Por cada serie temporal:
        thetas_window = thetas(:, (i-1)*(n+1)+c);
        %Evolución temporal de los parámetros theta:
        plot([1:NTotalWindows], thetas_window); axis tight; grid on; title('Theta(#ventana)'); xlabel('#ventana'); ylabel(strcat('Theta_', string(c-1)));
        hold on;
    end
    hold off;
end

for a=1:4 %Por cada parámetro alpha-stable posible (solo hay 4)
    figure;
    for i=1:size(alphas,2)/4 %Por cada serie temporal
        alphas_window = alphas(:,(i-1)*4+a);
        plot([1:NTotalWindows], alphas_window); axis tight; grid on; title('Alphas(#ventana)'); xlabel('#ventana');
        if(a == 1)
            ylabel('Alpha');
        end
        if(a == 2)
            ylabel('Beta');
        end
        if(a == 3)
            ylabel('Gamma');
        end
        if(a == 4)
            ylabel('Delta');
        end
        hold on;
    end
    hold off;
end