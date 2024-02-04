% Ejemplo del modelo SVM de margen suave sobre dataset Iris.

clear all 
clc

addpath(genpath('funciones'))

% Cargamos la base de datos iris
load fisheriris;

% Para llevar a cabo una clasificación binaria, nos quedamos con 2 de las 3
% especies recogidas en el dataset
indices = strcmp(species, 'setosa') | strcmp(species, 'versicolor');
X = meas(indices, :);  % Tomar solo las clases setosa y versicolor
Y = [ones(1, 50) * -1, ones(1, 50)]'; % Etiquetas correspondientes -> -1 para 'setosa' (primeras 50 posiciones) y 1 para 'versicolor' (el resto)

% Porcentaje del conjunto de entrenamiento
porcentaje_train = 70;

% Número de observaciones del conjunto de entrenamiento
n_observaciones = size(X,1);
n_train = round((porcentaje_train / 100) * n_observaciones);

% Separamos en conjuntos de train y test
indices_aleatorios = randperm(n_observaciones); % Creamos una lista de índices aleatorios

indices_train = indices_aleatorios(1:n_train); 
indices_test = indices_aleatorios(n_train+1:end); 

X_train = X(indices_train, :);
Y_train = Y(indices_train);

X_test = X(indices_test, :);
Y_test = Y(indices_test);

% Usamos la función del libro
[w, b] = svm_prim_sep(X_train, Y_train); 

% Calculamos las predicciones
Y_pred = sign(X_test*w + b);

% Porcentaje de aciertos sobre el conjunto de test
porcentaje_aciertos_test = mean(Y_pred == Y_test) * 100;
porcentaje_aciertos_test