% Ejemplo del modelo SVM de margen duro sobre datos sintéticos en dos
% dimensiones; datos linealmente separables.

clear all 
clc

addpath(genpath('funciones'))

% Generamos los datos
num = 50; % Número de puntos
% dim = 2; % Dimensión de los puntos
rng(9); % Semilla

data = rand(num, 2);
labels = ones(num, 1);
labels(1:num/2) = -1;

% Desplazamos los puntos de la segunda clase para que sean linealmente
% separables
desplazamiento = [1,1];
data(num/2+1:end,:) = data(num/2+1:end,:) + desplazamiento;


% Implementación del SVM de margen duro de Li (2015)
[w, b] = svm_prim_sep(data, labels);
[w_dual, b_dual, alpha] = svm_dual_sep(data, labels);

% Hiperplano separador (una recta en este caso)
x_recta = min(data(:, 1)) - 0.5:0.1:max(data(:, 1)) + 0.5;
y_recta = (-w(1)*x_recta - b)/w(2); % Despejando y de w(1)x + w(2)y + b = 0

% Hiperplanos que pasan por los vectores soporte
data_sv1 = data(alpha > 0 & alpha < 1, :); % Vectores soporte de la clase -1
data_sv2 = data(alpha > 0 & alpha < 1, :); % Vectores soporte de la clase 1
alpha_sv1 = alpha(alpha > 0 & alpha < 1); % Multiplicadores de Lagrange de la clase -1
alpha_sv2 = alpha(alpha > 0 & alpha < 1); % Multiplicadores de Lagrange de la clase 1

x_sv1 = min(data(:, 1)) - 0.5:0.1:max(data(:, 1)) + 0.5;
y_sv1 = (-w(1)*x_sv1 - b + 1)/w(2); % Despejando y de w(1)x + w(2)y + b = 1

x_sv2 = min(data(:, 1)) - 0.5:0.1:max(data(:, 1)) + 0.5;
y_sv2 = (-w(1)*x_sv2 - b - 1)/w(2); % Despejando y de w(1)x + w(2)y + b = -1


% Graficamos los datos y el hiperplano separador
figure(1);
gscatter(data(:, 1), data(:, 2), labels, 'k', 'o*', 5);
title("Datos linealmente separables")

figure(2);
gscatter(data(:, 1), data(:, 2), labels, 'k', 'o*', 5);
hold on;
plot(x_recta, y_recta, 'k-', 'LineWidth', 2);
plot(x_recta, y_recta + 0.1, '--', 'LineWidth', 2, 'Color', "r");
plot(x_sv1, y_sv1, '--', 'LineWidth', 0.5, 'Color', "b");
plot(x_sv2, y_sv2, '--', 'LineWidth', 0.5, 'Color', "g");
% Agregar las ecuaciones de los hiperplanos encima de las líneas
text(0.61, 1.61, sprintf('w''x + b = 1'), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'Color', 'b');
text(0.2, 1.2, sprintf('w''x + b = -1'), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'Color', 'g');

legend('Clase -1', ...
    'Clase 1', ...
    'Hiperplano separador que maximiza las distancias', ...
    'Hiperplano separador que no maximiza las distancias', ...
    Location='northwest');
hold off;
title("Hiperplano separador obtenido con SVM de margen duro")