function ker=kernel(X,xj,k)
%Computa o kernel de X e xj de acordo com o parametro k que identifica a
%função de kernel utilizada
N=size(X,1);
ker=zeros(N,1);
if k=='g'
    for i=1:N
        ker(i,1)=exp(-norm(X(i,:)-xj)); %gaussiana
    end
elseif k=='l'
    for i=1:N
        ker(i,1)=X(i,:)*xj'; %linear
    end
elseif k=='p'
    for i=1:N
        ker(i,1)=(X(i,:)*xj').^4; %polinomial de 4 ordem
    end
end

end