%% Clears
clc;
clear all;
load('FisherIris.mat');
%% Parametros
mtData = [FisherIris.SepalLength FisherIris.SepalWidth FisherIris.PetalLength FisherIris.PetalWidth double(FisherIris.Species)]; %conjunto de amostras
dK = 3 ; %numero de classes 
dC= 0.4; %C
dMaxIter=10; %max de iterações sem mudar o alpha
dTol=10^-5;
[nSamples,nCols]=size(mtData); %numero de amostras total e numero de colunas da matriz
[TrainInd,~,TestInd]=dividerand(nSamples,0.8,0,0.2);
mtTrain = mtData(TrainInd,:); %matriz de treinamento
mtTest = mtData(TestInd,:); %matriz de teste
mtX=mtTrain(:,1:end-1); %dados de entrada
mtD=mtTrain(:,end); %dados de saida 
%% Criando as 3 SVMs

mtY1=mtD; mtY2=mtD; mtY3=mtD;
mtY1(mtY1~=1)=-1; %Mudança dos dados para SVM1
mtY2(mtY2~=2)=-1; %Mudança dos dados para SVM2
mtY2(mtY2==2)=1;
mtY3(mtY3~=3)=-1; %Mudança dos dados para SVM3
mtY3(mtY3==3)=1;
[W1,b1]=createSVM(dC,dTol,dMaxIter,mtX,mtY1,'l'); %cria a SVM1
[W2,b2]=createSVM(dC,dTol,dMaxIter,mtX,mtY2,'l'); %cria a SVM2
[W3,b3]=createSVM(dC,dTol,dMaxIter,mtX,mtY3,'l'); %cria a SVM3

%% Um-contra-todos e teste
%treinamento
mtY=[mtX*W1'+b1 , mtX*W2'+b2, mtX*W3'+b3];
[~,I]=max(mtY,[],2);
I(I~=mtD)=0;
fprintf('Porcentagem do treinamento: %6.4f',nnz(I)/length(I))
%teste
mtX=mtTest(:,1:end-1); %matriz de entrada para teste
mtDTest=mtTest(:,end); %vetor de saida desejado
mtY=[mtX*W1'+b1 , mtX*W2'+b2, mtX*W3'+b3];
[~,I]=max(mtY,[],2);
I(I~=mtDTest)=0;
pCT= nnz(I)/length(I);
fprintf('\nPorcentagem de teste: %6.4f \n',pCT)



