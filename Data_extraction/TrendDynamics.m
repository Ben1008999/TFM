%Obtener parámetros alpha-stable y coeficientes de regresión polinómica
%para ventanas de Tventana minutos que se deslizan cada segundo:
clear all; close all; clc; warning off
addpath('./Functions')

%PARAMETROS DE ENTRADA:----------------------------------------------------
JSONoutput_filename = "trendDynamicsOutput.json"; %Nombre del fichero JSON de salida. Ejemplo: outputJSON.json
Tventana = 30; %[min] (Tamaño de ventana deslizante T)
n = 7; %Grado de la regresión polinómica
Granularidad_deteccion = 180; %= scope del sistema (alcance o tiempo de incertidumbre de predicción)
bitsPaquetes = 3; %Indica si trabajar con bits/s (2) o packets/s (3)
NTotalWindows = 100; %Si se quiere usar un número de ventanas concreto y no esperar a que se procese toda la serie completa (tardaría días). En caso de querer procesar todas las ventanas, poner -1
filenames = ["./TimeSeriesData/march_week3_csv/BPSyPPS.txt";
             "./TimeSeriesData/march_week4_csv/BPSyPPS.txt";
             "./TimeSeriesData/march_week5_csv/BPSyPPS.txt";
             "./TimeSeriesData/april_week2_csv/BPSyPPS.txt";
             "./TimeSeriesData/april_week3_csv/BPSyPPS.txt";
             "./TimeSeriesData/april_week4_csv/BPSyPPS.txt";
             "./TimeSeriesData/may_week1_csv/BPSyPPS.txt";
             "./TimeSeriesData/may_week3_csv/BPSyPPS.txt";
             "./TimeSeriesData/june_week1_csv/BPSyPPS.txt";
             "./TimeSeriesData/june_week2_csv/BPSyPPS.txt";
             "./TimeSeriesData/june_week3_csv/BPSyPPS.txt";
             "./TimeSeriesData/july_week1_csv/BPSyPPS.txt"];
%--------------------------------------------------------------------------
%Obtener la matriz con todas las series temporales de cada semana
%NOTA: Dado que todas las series se ordenan semanalmente, algunas
%no tienen datos para ciertos días de la semana (por ejemplo, tal vez la
%semana 'X' del mes 'Y' no tenga datos para el Lunes y el Martes). Por esta
%razón, es posible ver valores NaN al comienzo en las bases de datos de los
%parámetros theta y alpha.
%LA PRIMERA FILA DE LA MATRIZ DE AGREGADO ES EL DOMINIO:
domain = 1:7*24*60*60; %[1 = Lunes 00:00:01 -> 7*24*60*60 = Lunes (semana siguiente) 00:00:00]
labels = {['domain', getLabelsFromFilenames(filenames)]'};
agregado = getAggregateNetTrafficMatrix(filenames, bitsPaquetes, domain);

%Obtener series temporales con las dinámicas de la tendencia polinómica y los parámetros alpha-estable:
%Definir el dominio de la regresión:
Tsventana = Tventana*60;
domainFIT = getDomainFIT(Tsventana, Granularidad_deteccion);
[theta_params, alpha_params] = processTrendDynamics(agregado, Tsventana, n, NTotalWindows, domainFIT);

%Exportar las dinámicas de la tendencia y los parámetros alpha-stable:
outputJSON = buildJSONinfo(Tventana, n, Granularidad_deteccion, bitsPaquetes, NTotalWindows, agregado, labels, theta_params, alpha_params, domainFIT);
writeJSON(strcat("./Data_extraction_output/", JSONoutput_filename), outputJSON);
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
thetas = cell2mat(theta_params);
alphas = cell2mat(alpha_params);
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