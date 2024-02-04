% Ejemplo del modelo SVM de margen suave sobre datos sintéticos en dos
% dimensiones; datos no linealmente separables.

clear all 
clc

addpath(genpath('funciones'))

% Generamos los datos no linealmente separables
num = 50; % Número de puntos
% dim = 2; % Dimensión de los puntos
rng(9);  % Semilla

data = rand(num, 2);
labels = ones(num, 1);
labels(1:num/2) = -1;

data(num/2+1:end,:) = data(num/2+1:end,:) + [0.4,0.5];

% Implementación del SVM de margen suave de Li (2015)
C = 2;
[w, b] = svm_prim_nonsep2(data, labels, C);
[w_dual, b_dual, alpha] = svm_dual_nonsep2(data, labels, C);

% Hiperplano separador (una recta en este caso)
x_recta = min(data(:, 1)) - 0.5:0.1:max(data(:, 1)) + 0.5;
y_recta = (-w(1)*x_recta - b)/w(2); % Despejando y de w(1)x + w(2)y + b = 0

% Graficamos los datos y el hiperplano separador
figure(1);
gscatter(data(:, 1), data(:, 2), labels, 'k', 'o*', 5);
title("Datos no linealmente separables");

figure(2);
gscatter(data(:, 1), data(:, 2), labels, 'k', 'o*', 5);
hold on;
plot(x_recta, y_recta, 'k-', 'LineWidth', 2);
legend('Clase -1', ...
    'Clase 1', ...
    'Hiperplano separador', ...
    Location='northwest');
hold off;
title("Hiperplano separador obtenido con SVM de margen suave")