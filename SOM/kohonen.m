%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mateus Pontes Mota
% Redes Neurais Artificiais
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clears e Imports
clear all;
clc;
load('dados.mat');
%% Declaração das Variavéis
dNeurons=6; %n de neuronios da rede
dAlpha0=0.6; %alfa inicial
dSigma0=10; %sigma inicial
dMaxIter=5000; %max de iterações
dT=0; %iteração
mtInput=pontos'; %matriz de entrada, transpor para que cada linha seja uma amostra
[dL, dC] = size(mtInput); %n de linhas e de colunas
tsW=rand(dNeurons,dNeurons,dC); %tensor dos pesos, duas matriz 6x6 concatenadas
dThal=dMaxIter/log10(dSigma0); %Thal  utilizado no calculo de sigma 
%% Plot Inicial
figure(1);
plot(mtInput(:,1),mtInput(:,2),'.b')
hold on;
plot(tsW(:,:,1),tsW(:,:,2),'or')
plot(tsW(:,:,1),tsW(:,:,2),'k','linewidth',2)
plot(tsW(:,:,1)',tsW(:,:,2)','k','linewidth',2)
hold off;
title('t=0');
drawnow;
%% Treinamento
while(dT<=dMaxIter)
    dAlpha=dAlpha0*(1-dT/dMaxIter); %diminuir a taxa de aprendizagem
    dSigma=round(dSigma0*(1-dT/dMaxIter)); %diminuir o tamanho da vizinhança
    for ii= randperm(dL) % seleção aleatoria das amostras
        mtDist=(mtInput(ii,1)-tsW(:,:,1)).^2+(mtInput(ii,2)-tsW(:,:,2)).^2; %calculo da distancia euclidiana
        [vtWinV,vtWinId]=min(mtDist); %busca do vencedor por coluna com o indice da linha 
        [~,dMinCol]=min(vtWinV); %coluna do menor valor
        dMinLin=vtWinId(dMinCol); %linha do menor valor
        %atualização do neuronio vencedor
        tsW(dMinLin,dMinCol,:) = [tsW(dMinLin,dMinCol,1) tsW(dMinLin,dMinCol,2)] + dAlpha*(mtInput(ii,:)-[tsW(dMinLin,dMinCol,1) tsW(dMinLin,dMinCol,2)]);
        %atualização dos vizinhos
        for jj=1:1:dSigma
            dLin=dMinLin-jj;
            dCol=dMinCol;
            if(dLin>=1) %limita linha minima
                tsW(dLin,dCol,:) = [tsW(dLin,dCol,1) tsW(dLin,dCol,2)] + dAlpha*(mtInput(ii,:)-[tsW(dLin,dCol,1) tsW(dLin,dCol,2)]);
            end
            dLin=dMinLin+jj;
            dCol=dMinCol;
            if(dLin<=6) %limita linha maxima
                tsW(dLin,dCol,:) = [tsW(dLin,dCol,1) tsW(dLin,dCol,2)] + dAlpha*(mtInput(ii,:)-[tsW(dLin,dCol,1) tsW(dLin,dCol,2)]);
            end
            dLin=dMinLin;
            dCol=dMinCol-jj;
            if(dCol>=1) %limita coluna minima
                tsW(dLin,dCol,:) = [tsW(dLin,dCol,1) tsW(dLin,dCol,2)] + dAlpha*(mtInput(ii,:)-[tsW(dLin,dCol,1) tsW(dLin,dCol,2)]);
            end
            dLin=dMinLin;
            dCol=dMinCol+jj;
            if(dCol<=6) %limita coluna maxima
                tsW(dLin,dCol,:) = [tsW(dLin,dCol,1) tsW(dLin,dCol,2)] + dAlpha*(mtInput(ii,:)-[tsW(dLin,dCol,1) tsW(dLin,dCol,2)]);
            end
        end
        
    end
    dT=dT+1;
    if(  (dT==1) || (dT==100) || (dT==1000) || (dT==2000) || (dT==5000) )
        figure(1);
        plot(mtInput(:,1),mtInput(:,2),'.b')
        hold on;
        plot(tsW(:,:,1),tsW(:,:,2),'or')
        plot(tsW(:,:,1),tsW(:,:,2),'k','linewidth',2)
        plot(tsW(:,:,1)',tsW(:,:,2)','k','linewidth',2)
        hold off;
        title(['t=' num2str(dT)]);
        drawnow
        name=sprintf('T%deAlfa%d',dT,dAlpha0*10);
        print(name,'-dpng');
    end
end