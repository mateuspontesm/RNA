function [ W,b ] = createSVM(C,tol,max_iter,X,Y,k  )
%%Esta função cria uma SVM baseada no algoritmo SMO
%C: parametro de regularização
%tol: tolerancia
%max_iter: max # de vezes que iteramos sobre alpha sem mudar
%X,Y : Entradas e saídas do conj de treinamento.
%k : funcao de kernel
N=length(Y);
alpha=zeros(N,1);
b=0;
counter=0;

while counter<max_iter
    N=size(Y,1);
    changedAlphas=0;
    for ii=1:N
        Ei=sum(alpha.*Y.*kernel(X,X(ii,:),k)) - Y(ii); %erro da amostra i: Ei = f(xi)-yi
        %onde f(xi)=sum(alpha.y.k(x))
        %eq. 2
        if ((Ei*Y(ii) < -tol) && alpha(ii) < C) || (Ei*Y(ii) > tol && (alpha(ii) > 0))
            j=ii;
            %olha somente os ii diferentes de j
            while(j==ii)
                j=randi(N);
            end
            Ej = sum(alpha.*Y.*kernel(X,X(j,:),k)) - Y(j); %% computa erro da amostra j, eq. 2
            alphaiOld = alpha(ii); %guarda o alpha indice i antigo
            alphajOld = alpha(j); %guarda o alpha indice j antigo
            
            %Encontra as margens L e H de modo que L <= alpha(j) <= H, para
            %satisfazer a restrição 0 <= alpha(j) <= C :
            %eq. 10 ou eq. 11
            if Y(ii)~=Y(j)
                L = max(0,alpha(j)-alpha(ii));
                H = min(C,C+alpha(j)-alpha(ii));
            else
                L = max(0,alpha(ii)+alpha(j)-C);
                H = min(C,alpha(ii)+alpha(j));
            end
            %continua para o prox i, pois n podemos balancear ai aj
            if (L==H)
                continue
            end
            %Calcular Eta= 2*K(xi,xj) - K(xi,xi) - K(xj,xj)
            %eq. 14
            eta = 2*kernel(X(j,:),X(ii,:),k) - kernel(X(ii,:),X(ii,:),k) - kernel(X(j,:),X(j,:),k);
            %continua para o prox i caso nao seja um maximo
            if eta>=0
                continue
            end
            %Calcula aj atraves da Eq. 12
            alpha(j) = alpha(j) - (Y(j)*(Ei-Ej))/eta;
            %Clip aj de modo que L <= aj <= H
            if alpha(j) > H
                alpha(j) = H;
            elseif alpha(j) < L
                alpha(j) = L;
            end
            %verifica se aj mudou consideravelmente
            %caso contrario, nao precisamos ajustar ai
            %e passamos para o prox i
            if norm(alpha(j)-alphajOld) < tol
                continue
            end
            %Calcula ai pela Eq. 16
            alpha(ii) = alpha(ii) + Y(ii)*Y(j)*(alphajOld-alpha(j));
            %Calcula b1 e b2 usando eq. 17 e eq. 18
            b1 = b - Ei - Y(ii)*(alpha(ii)-alphaiOld)*kernel(X(ii,:),X(ii,:),k)...
                -Y(j)*(alpha(j)-alphajOld)*kernel(X(ii,:),X(j,:),k);
            b2 = b - Ej - Y(ii)*(alpha(ii)-alphaiOld)*kernel(X(ii,:),X(j,:),k)...
                -Y(j)*(alpha(j)-alphajOld)*kernel(X(j,:),X(j,:),k);
            %Calcula b de acordo com Eq. 19
            if 0<alpha(ii)<C
                b=b1;
            elseif 0<alpha(j)<C
                b=b2;
            else
                b=(b1+b2)/2;
            end
            changedAlphas=changedAlphas+1;
        end
    end
    if(changedAlphas==0)
        break
    end
    counter=counter+1;
    %%Utiliza somente as amostras uteis
    X=X((find(alpha~=0)),:);
    Y=Y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end

% Calculo do W
totalSum = 0;
N=length(alpha);
for jj=1:N
    totalSum = totalSum + alpha(jj)*Y(jj)*X(jj,:);
end

W = totalSum;
b = Y(1) - X(1,:)*W';
end

