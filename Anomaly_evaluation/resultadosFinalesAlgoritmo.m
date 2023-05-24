clear all; close all; clc; warning('off');

%Variables de input:--------------------------------------------------------------------------------------------------------------------
tiempofinal = 67900; %Valor numérico del nombre de archivo de los ficheros de output del predictor (ej.: polPredicted.txt)
Tsventana = 1800; %[s] Tiempo en segundos de la ventana de simulación (mismo valor que se utilizó en el extractor de datos)
n = 7; %Orden del polinomio
tolerancia = 0.2; %Cuando el módulo es inferior a la tolerancia, se deja de tomar valores. Este valor determina el ancho de banda de evaluación de la anomalía en el cociente de Funciones Características
granularidad_deteccion = 180; %[s]
zoom = 9; %[min]; %Acercamiento para la detección de la anomalía
%Duración del ataque: %[s]
AttackDuration = 142;
%Potencia de ataque: [packets/s]
AttackPower = 12000;
%---------------------------------------------------------------------------------------------------------------------------------------

%Lectura de los polinomios y datos de salida del predictor:
Predicted_pol = load(strcat(strcat('polPredicted', num2str(tiempofinal)), '.txt'));
Real_pol = load(strcat(strcat('polTruth', num2str(tiempofinal)), '.txt'));
pol = load(strcat(strcat('polParteConocida', num2str(tiempofinal)), '.txt'));
polD = load(strcat(strcat('polParteDesconocida', num2str(tiempofinal)), '.txt'));
W = load(strcat(strcat('WindowTest', num2str(tiempofinal)), '.txt'));
pol_completo = [pol; polD];

%Dominio de tiempo:
tref = 1465775999; %Valor constante (es el tiempo de referencia del Lunes [00:00:00])
final = tiempofinal+tref+granularidad_deteccion;
inicio = final-Tsventana;
tiempoventana = datetime([inicio+1:final], 'convertfrom', 'posixtime')';

%Prueba de ataque:
WAtaque = zeros(1, length(W));
%Configurar distintas distribuciones de ataque:
%Bi-modal Dirac:
%RandomW = AttackPower*ones(1, AttackDuration);
%Beta distribution:
% std = 2000;
% RandomW = abs(AttackPower + (random('Beta', 2, 2, 1, AttackDuration)-0.5)*2*std);

%Uniforme:
% std = 2000;
% RandomW = abs(AttackPower + (rand(1, AttackDuration)-0.5)*2*std);

%Alpha-estable:
%RandomW = abs(random('Stable', 1.9, 1, 2000, AttackPower, 1, AttackDuration));

t = [0:AttackDuration-1];
RandomW = abs(cos(2*pi*0.05*t)*10000 + (1 - exp(-0.02*t))*AttackPower);

WAtaque(end-(AttackDuration-1):end) = RandomW;
WAtaque = WAtaque';

%Simulación de ataque:
WSuma = W + WAtaque;

figure; plot(tiempoventana, WSuma, 'r');
hold on; plot(tiempoventana, W, 'k');
xline(tiempoventana(end-granularidad_deteccion), '--', 'lineWidth', 2);
grid on; plot(tiempoventana, Real_pol, 'b', 'lineWidth', 1.7); %'Color', [0.6350 0.0780 0.1840]
plot(tiempoventana, Predicted_pol, 'r', 'lineWidth', 1.7); ylim([0.8*min(W) 1.2*max(W)]);
xlim([tiempoventana(1) tiempoventana(end)]); xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo')

%Idea: usar como polinomio completo una segunda regresión:
W_2 = [W(1:end-granularidad_deteccion); polD];
domainFIT = [[-(Tsventana-1):0] + ceil(granularidad_deteccion/2)]';
stepdomainFIT = 1/(2*domainFIT(end));
domainFIT = domainFIT*stepdomainFIT;
regressionType = strcat('poly', num2str(n));
h_final_2 = fit(domainFIT, W_2, regressionType);
h_final = fit([1:length(W_2)]', W_2, regressionType);
figure; plot([1:length(W_2)], W_2); hold on; plot([1:length(W_2)], h_final(1:Tsventana), 'r', 'lineWidth', 1.7); ylim([0.8*min(W_2) 1.2*max(W_2)]); plot([1:length(W_2)], Real_pol, 'b', 'lineWidth', 1.7);

figure; plot(tiempoventana, WSuma, 'r'); hold on; plot(tiempoventana, h_final(1:Tsventana), 'lineWidth', 2);
plot(tiempoventana, W, 'k'); xline(tiempoventana(end-granularidad_deteccion), '--', 'lineWidth', 2);
grid on; axis tight; xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo')
hold on; plot(tiempoventana, Real_pol, 'b', 'lineWidth', 1.7);

figure; plot(tiempoventana, WSuma, 'r'); hold on; plot(tiempoventana, pol_completo, 'lineWidth', 2);
plot(tiempoventana, W, 'k'); xline(tiempoventana(end-granularidad_deteccion), '--', 'lineWidth', 2);
grid on; axis tight; xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo');
plot(tiempoventana, Real_pol, 'b', 'lineWidth', 1.7);

error = sqrt(mean((Predicted_pol(end-granularidad_deteccion:end)-Real_pol(end-granularidad_deteccion:end)).^2))
%Idea nueva:
% pol_completo = h_final(1:Tsventana);
%Idea 3: Sin embargo, en algunas ventanas es mejor coger la parte conocida de pol_completo solo y
%concatenarla con la predicción. En otras ventanas es peor, así que
%mantenemos la idea 2.
% pol_completo = [pol_completo(1:end-granularidad_deteccion); polD];
% figure; plot(tiempoventana, WSuma, 'r'); hold on; plot(tiempoventana, pol_completo, 'lineWidth', 2);
% plot(tiempoventana, W, 'k'); xline(tiempoventana(end-granularidad_deteccion), '--', 'lineWidth', 2);
% grid on; axis tight; xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo');
% plot(tiempoventana, Real_pol, 'b', 'lineWidth', 1.7);

%Zoom sobre los datos:
W_zoom = W(end-zoom*60+1:end);
tiempoventana_zoom = tiempoventana(end-zoom*60+1:end);
WAtaque_zoom = WAtaque(end-zoom*60+1:end);
pol_completo_zoom = pol_completo(end-zoom*60+1:end);
WSuma_zoom = W_zoom + WAtaque_zoom;
% figure; plot(tiempoventana_zoom, WSuma_zoom, 'r'); hold on; plot(tiempoventana_zoom, pol_completo_zoom, 'lineWidth', 2);
% plot(tiempoventana_zoom, W_zoom, 'k'); xline(tiempoventana(Tsventana-granularidad_deteccion), '--', 'lineWidth', 2);
% xlim([tiempoventana_zoom(1) tiempoventana_zoom(end)]); grid on; xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo')

%Estas dos variables solo se usan para el plot siguiente (de cara a la
%memoria):
pol_real_zoom = Real_pol(end-zoom*60+1:end);
pol_predicted_zoom = Predicted_pol(end-zoom*60+1:end);
figure; plot(tiempoventana_zoom, WSuma_zoom, 'r'); hold on; 
plot(tiempoventana_zoom, W_zoom, 'k'); plot(tiempoventana_zoom, pol_predicted_zoom, 'r', 'lineWidth', 2); plot(tiempoventana_zoom, pol_real_zoom, 'b', 'lineWidth', 2);
xline(tiempoventana(Tsventana-granularidad_deteccion), '--', 'lineWidth', 2);
xlim([tiempoventana_zoom(1) tiempoventana_zoom(end)]); grid on; xlabel('Time'); ylabel('Bandwidth [packets/s]'); title('Ventana objetivo')
ylim([0.8*min(WSuma) 1.2*max(WSuma)]);

%Algoritmo alpha-stable:
[w, w_inv, stepw, dominioFDP, points_under_zero, stepFDP] = getFrequencyAndFDPDomainForCharacteristicFunctions(zeros(1,1), 3);

Wconocida = W_zoom(1:end-granularidad_deteccion);
pol_zona_conocida = fit([1:length(Wconocida)]', Wconocida, 'poly9');
Xt_background_zoom = Wconocida - pol_zona_conocida(1:length(Wconocida));
pol_zona_conocida = fit([1:length(Xt_background_zoom)]', Xt_background_zoom, 'poly9');
Xt_background_zoom = Xt_background_zoom - pol_zona_conocida(1:length(Wconocida));
%Opción A: Asumir que los alpha-stable de toda la ventana son los
%alpha-stables de la parte conocida estacionarizada:
Parametros_AlphaStable = fitdist(Xt_background_zoom, 'Stable');
alpha_modelo = Parametros_AlphaStable.alpha;
beta_modelo = Parametros_AlphaStable.beta;
gamma_modelo = Parametros_AlphaStable.gam;
delta_modelo = Parametros_AlphaStable.delta;
pdN = makedist('Stable','alpha',alpha_modelo,'beta',beta_modelo,'gam',gamma_modelo,'delta',delta_modelo);
pdf_alphaNormal = pdf(pdN, dominioFDP);
figure;
plot(dominioFDP, pdf_alphaNormal, 'LineWidth', 1.5); grid on; hold on; histogram(Xt_background_zoom, 50, 'normalization', 'pdf'); 
hold off; title('Histograma y pdf de tráfico normal');
%Opción B: Usar la predicción de los alpha-stables:
% Parametros_AlphaStable = load('alphaStables.txt');
% alpha = Parametros_AlphaStable(1);
% beta = Parametros_AlphaStable(2);
% gamma = Parametros_AlphaStable(3);
% delta = Parametros_AlphaStable(4);
% delta = 0;

% Xt_At_zoom = WSuma_zoom - pol_completo_zoom;
%Idea 2: Mucho mejor:
Xt_At_zoom_incertidumbre = WSuma_zoom(end-granularidad_deteccion+1:end) - pol_completo_zoom(end-granularidad_deteccion+1:end);
Xt_At_zoom = [Xt_background_zoom; Xt_At_zoom_incertidumbre];
ECFSuma = computeECF(w, Xt_At_zoom);
figure;
plot(w, abs(ECFSuma), w, angle(ECFSuma));
title('ECF de tráfico suma estacionarizado');
ylim([0 1.2])
xlim([-1.5 1.5]*10^(-3))
xlabel('w [rad/s]');

CFNormal = computeCF(w, alpha_modelo, beta_modelo, gamma_modelo, delta_modelo, 0);
figure;
plot(w, abs(CFNormal), w, angle(CFNormal)); axis tight; title('CF de tráfico normal estacionarizada');
ylim([0 1.2])
xlim([-1.5 1.5]*10^(-3))
xlabel('w [rad/s]');

%ECF del ataque:
ECFAtaque = computeECF(w, WAtaque_zoom);
%ECF del cociente:
Cociente = ECFSuma./CFNormal;
figure; plot(w, abs(Cociente), w, angle(Cociente)); hold on;
plot(w, angle(ECFAtaque), 'g'); plot(w, abs(ECFAtaque), 'g'); hold off;
legend('Módulo', 'Fase', 'Módulo y fase teóricos');
ylim([-1 1.2])
xlim([-1.5 1.5]*10^(-3))
grid on; xlabel('w [rad/s]');
%Recortar rango de frecuencias:
[Cocienterec, wrec] = recortarECF(w, stepw, CFNormal, Cociente, tolerancia);
figure;
plot(wrec, abs(Cocienterec), wrec, angle(Cocienterec)); hold on;
plot(w, angle(ECFAtaque), 'g'); plot(w, abs(ECFAtaque), 'g'); hold off; grid on;
legend('Módulo', 'Fase', 'Módulo y fase teóricos');
ylim([-0.2 1.2])
xlim([wrec(1) wrec(end)])

%Comparación con el caso ideal de conocer la tendencia background:
h_ideal = fit([1:length(W_zoom)]', W_zoom, 'poly7');
figure; plot(WSuma_zoom, 'r'); hold on; plot(WSuma_zoom, 'b'); plot(h_ideal(1:length(WSuma_zoom)), 'r', 'lineWidth', 1.2); xline(length(WSuma_zoom)-granularidad_deteccion, '--', 'lineWidth', 2);
grid on;

Xt_At_zoom_incertidumbre_ideal = WSuma_zoom(end-granularidad_deteccion+1:end) - h_ideal(length(WSuma_zoom)-granularidad_deteccion+1:length(WSuma_zoom));
Xt_At_ideal_zoom = [Xt_background_zoom; Xt_At_zoom_incertidumbre_ideal];
ECFSuma_ideal = computeECF(w, Xt_At_ideal_zoom);
figure;
plot(w, abs(ECFSuma_ideal), w, angle(ECFSuma_ideal));
title('ECF caso ideal de tráfico suma estacionarizado');
ylim([0 1.2])
xlim([-1.5 1.5]*10^(-3))
xlabel('w [rad/s]');

pol_ideal_background = fit([1:length(W_zoom)]', W_zoom, 'poly9');
Xt_background_ideal_zoom = W_zoom - pol_ideal_background(1:length(W_zoom));
pol_ideal_background = fit([1:length(Xt_background_ideal_zoom)]', Xt_background_ideal_zoom, 'poly9');
Xt_background_ideal_zoom = Xt_background_ideal_zoom - pol_ideal_background(1:length(W_zoom));
Parametros_AlphaStable = fitdist(Xt_background_ideal_zoom, 'Stable');
alpha_ideal = Parametros_AlphaStable.alpha;
beta_ideal = Parametros_AlphaStable.beta;
gamma_ideal = Parametros_AlphaStable.gam;
delta_ideal = Parametros_AlphaStable.delta;
pdN_ideal = makedist('Stable','alpha',alpha_ideal,'beta',beta_ideal,'gam',gamma_ideal,'delta',delta_ideal);
pdf_alphaNormal_ideal = pdf(pdN_ideal, dominioFDP);
figure;
plot(dominioFDP, pdf_alphaNormal_ideal, 'LineWidth', 1.5); grid on; hold on; histogram(Xt_background_ideal_zoom, 50, 'normalization', 'pdf'); 
plot(dominioFDP, pdf_alphaNormal, 'LineWidth', 1.5); grid on; histogram(Xt_background_zoom, 50, 'normalization', 'pdf'); 
hold off; title('Histograma y pdf de tráfico normal');

CFNormal_ideal = computeCF(w, alpha_ideal, beta_ideal, gamma_ideal, delta_ideal, 0);
figure;
plot(w, abs(CFNormal_ideal), w, angle(CFNormal_ideal)); axis tight; title('CF caso ideal de tráfico normal estacionarizada');
ylim([0 1.2])
xlim([-1.5 1.5]*10^(-3))
xlabel('w [rad/s]');

Cociente_ideal = ECFSuma_ideal./CFNormal_ideal;
figure; plot(w, abs(Cociente_ideal), w, angle(Cociente_ideal)); hold on; title('Cociente caso ideal')
plot(w, angle(ECFAtaque), 'g'); plot(w, abs(ECFAtaque), 'g'); hold off;
legend('Módulo', 'Fase', 'Módulo y fase teóricos');
ylim([-1 1.2])
xlim([-1.5 1.5]*10^(-3))
grid on; xlabel('w [rad/s]');

%Filtrado paso bajo de la señal y rescate de la fdp de ataque:
filtro = zeros(1, length(w));
filtro(find(w == wrec(1)):find(w == wrec(end))) = 1;
%drop Inf values of Cociente:
Cociente(find(abs(Cociente) == Inf)) = 1000;
ECFAtaque_rescatada = filtro.*Cociente;
FT_fdp_Ataque_rescatada = flip(ECFAtaque_rescatada);
%Inversa de la FFT:
fdpAtaque = ifft(fftshift(FT_fdp_Ataque_rescatada));
%Adecuar al dominio:
fdpAtaque = [fdpAtaque(end-points_under_zero+1:end), fdpAtaque(1:end-points_under_zero)];
%Si no hubiera ataque: la fdp sería la ifft del filtro directamente, porque
%la ECF ataque del cociente sería módulo 1 y fase 0 que, al multiplicarse
%por el filtro, es igual al filtro.
fdpSinAtaque = ifft(fftshift(filtro));
fdpSinAtaque = [fdpSinAtaque(end-points_under_zero+1:end), fdpSinAtaque(1:end-points_under_zero)];
figure; plot(dominioFDP, abs(fdpAtaque)); hold on; plot(dominioFDP, abs(fdpSinAtaque)); legend('Ataque reconstruido', 'Sin ataque'); grid on;
xlim([0 1.2*AttackPower + 6000]); title('fdp Ataque rescatada'); xlabel('Bandwidth [packets/s]'); xline(AttackPower, '--'); ylabel('Probability');
figure; plot(w, abs(ECFAtaque_rescatada), w, angle(ECFAtaque_rescatada)); 
%Comparación caso ideal:
Cociente_ideal(find(abs(Cociente_ideal) == Inf)) = 1000;
ECFAtaque_rescatada = filtro.*Cociente_ideal;
FT_fdp_Ataque_rescatada = flip(ECFAtaque_rescatada);
%Inversa de la FFT:
fdpAtaque_ideal = ifft(fftshift(FT_fdp_Ataque_rescatada));
%Adecuar al dominio:
fdpAtaque_ideal = [fdpAtaque_ideal(end-points_under_zero+1:end), fdpAtaque_ideal(1:end-points_under_zero)];
figure; plot(dominioFDP, abs(fdpAtaque_ideal)); xlim([0 1.2*AttackPower + 6000]); title('fdp Ataque caso ideal'); xlabel('Bandwidth [packets/s]'); xline(AttackPower, '--'); ylabel('Probability');
figure; plot(w, abs(ECFAtaque_rescatada), w, angle(ECFAtaque_rescatada))

%Idea: en lugar de realizar el cociente, deconvolucionar fdps:
%FDP Tráfico background:
CFNormal_inv = computeCF(w_inv, alpha_modelo, beta_modelo, gamma_modelo, delta_modelo, 0);
FT_Normal = flip(CFNormal_inv);
%Inversa de la FFT:
fdpNormal = ifft(fftshift(FT_Normal));
%Adecuar valores al dominio:
fdpNormal = [fdpNormal(end-points_under_zero+1:end), fdpNormal(1:end-points_under_zero)];
figure; plot(dominioFDP, fdpNormal./stepFDP); hold on; plot(dominioFDP, pdf_alphaNormal); legend('1', '2');

%FDP Tráfico suma:
ECFSuma_inv = computeECF(w_inv, Xt_At_zoom);
FT_Suma = flip(ECFSuma_inv);
%Inversa de la FFT:
fdpSuma = abs(ifft(fftshift(FT_Suma)));
%Adecuar valores al dominio:
fdpSuma = [fdpSuma(end-points_under_zero+1:end), fdpSuma(1:end-points_under_zero)];