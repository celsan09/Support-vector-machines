% CV for CPSVM model with non-linear kernel
% Paper: Twin SVM for conditional probability estimation in binary and multiclass classification
% Shao et al. 2022
% 
% We solve the dual formulation (see eq. (65) of pdf file formulaciones_binarias.pdf) of CPSVM model given in eq. (37). 


clc
clear all
addpath(genpath('dataset_bin'))
addpath(genpath('dataset_Imb'))
addpath(genpath('Programas_Miguel'));

%load 'exa_bmpm.mat'
%load 'sonar.mat'
%load('heart_statlogN.mat');
%load('bupa_liverN.mat');
%load('ionosphereN.mat');
%load('breastcancer')
load('australian.mat');
%load('diabetes.mat');
%load('german_credit.mat');
%load('splice.mat');
%load x18data.mat; % Flare-M
%load x23data.mat; % Yeast4
%load yeast3.mat
%load('titanic.mat');
%load segment0_n.mat
%load('image_n.mat');
%load('waveformBin.mat');
%load 'phoneme.mat'
%load('ring_n.mat')


[m, n]=size(X);
fInd=[1:n]; %<--- return all features   
CV=10; % cantidad de folds para la CV

% C1, C2 in {2^{-7},2^{-6},...,2^6,2^7}
Cl=-7; 
Ch=7;  

AUCMATRIX=zeros(Ch-Cl+1, Ch-Cl+1);
ACCUMATRIX=zeros(Ch-Cl+1, Ch-Cl+1);

FunPara.kerfPara.type = 'rbf';

FunPara.epsi = 2^(-3); % eps=2^j with j belong to {-7,...,0}
%FunPara.kerfPara.pars=2^7; % Sigma=2^j with j belong to {-7,-6,...,6,7}

t0=cputime;
for i=Cl:Ch
    if
    FunPara.C1 =2^i;
    FunPara.C2=2^i;
    for j=Cl:Ch
     %   FunPara.C2=2^j;
       FunPara.kerfPara.pars=2^j;
        for k=1:CV
            tst=perm(k:10:m); % se debe agregar el perm, para que siempre se mantenga en conjunto
            trn=setdiff(1:m,tst);
            Ytr=Y(trn,:);    % definimos las etiquetas de entrenamiento
            Xtr=X(trn,fInd); % definimos el conjunto de entrenamiento
            Yt=Y(tst',:);    % definimos las etiquetas de test
            Xt=X(tst',fInd); % definimos el conjunto test
            [Prediction,Tf,S] = cpsvm_dual_V1(Xtr, Ytr, Xt, FunPara);
            [AUC(k),Accu(k)]=medi_auc_accu(Prediction,Yt);
        end
        AUCMATRIX(i-Cl+1,j-Cl+1)=mean(AUC);
        ACCUMATRIX(i-Cl+1,j-Cl+1)=mean(Accu);
    end
end

tf=cputime-t0

% Especifica el nombre del archivo de Excel
filename = 'resultados.xlsx';
% 
% % Escribe la matriz AUCMATRIX en la hoja de cálculo 'AUC'
xlswrite(filename, AUCMATRIX', 'AUC');
% 
% % Escribe la matriz ACCUMATRIX en la hoja de cálculo 'Accuracy'
xlswrite(filename, ACCUMATRIX', 'Accuracy'); 
