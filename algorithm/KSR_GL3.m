function [results,results_iter,A] = KSR_GL3(trainX,trainY,testX,testY,options)
%% Implementation of KSR_GLLR
%% intput:
%%%     trainX:                     The traing samples, m*n1
%%%     trainY:                     The labels of training samples, n1*1
%%%     testX:                      The test samples, m*n2
%%%     testY:                      The labels of test samples, n2*1
%% output:
%%%     results                      The results (list) [acc,NMI,purity]
%%%     results_iter                 The iteration information of 'results'
%%%     A                            The learned projection matrix
%% Version
%%%     V1                          2022-05-24
%%%     V2                          2022-10-07
    options=defaultOptions(options,'T',10,...    %% The iterations
                            'dim',30,...         %% The dimension reduced
                            't',10,...           %% The maximum iteration of GPI
                           'lambda1',0.01,...    %% The weight of L2,1 norm w.r.t Z
                           'lambda2',1e-4,...    %% The weight of kernel norm w.r.t GLLR
                           'ker',1,...           %% The kernel, 1 = linear, 2 = rbf
                           'gamma',1,...         %% The parameter of rbf kernel
                           'rho',0.1,...         %% The weight of Lagrange
                           'p',1.01);            %% The increase rate of Lagrange
    %% Parameters Setting
    Ypseudo=classifyKNN(trainX,trainY,testX,1);
    fprintf('[init] acc:%.4f\n',getAcc(Ypseudo,testY));
    T=options.T;
    t=options.t;
    dim=options.dim;
    lambda1=options.lambda1;
    lambda2=options.lambda2;
    rho=0.1;
    p=1.01;
    %% Data process
    [m,n]=size(trainX);
    C=length(unique(trainY));
    %% initialization
    KXY=kernelProject(options.ker,trainX, testX, options.gamma);
    KXX=kernelProject(options.ker,trainX, trainX, options.gamma);
    KXXK=KXX'*KXX;
    KXYK=KXX'*KXY;
    D=eye(n);
    %% Solve KXXK^(-0.5)
    [Uxx,Sigma_xx,Vxx]=mySVD(KXXK);
    Sigma_xx(Sigma_xx<0)=0;
    squreSigma=Sigma_xx.^0.5;
    inverseSqrtSigma=diag(1./(diag(squreSigma)));
    inverseSqrtSigma(isinf(inverseSqrtSigma))=0;
    KXXinverse=Uxx*(inverseSqrtSigma)*Vxx';
    %% Solve KXX*hotY
    hotY=hotmatrix(trainY,C,1);
    KXXhotY=KXX*hotY;
    L2=KXXinverse'*KXXhotY;
    opt2.ReducedDim=dim;
    Q=PCA(trainX,opt2);
    dim=min(dim,size(Q,2));
    Y1=zeros(dim,C); 
    for i=1:T
     %% Update B
      B=SVT(Q'*L2+Y1/rho,lambda2/rho);
      %% Update Z
      left=KXXK+lambda1*D;
      right=KXYK;
      Z=left\right;
      %% Update D
      D=updateL21(Z);
      %% Update A by GPI
      %%% Let Q=(KXX)^(-0.5) A, solve Q
      L1=KXXinverse*(KXY-KXX*Z);
      left=L1*L1'+rho*(L2*L2');
      right=rho*L2*(B-Y1/rho)';
      try 
          opt.t=t;
          Q=GPI(left,right,opt);
          Q=real(Q);
      catch ME
          break;
      end
      %% Update Lagrange
      Y1=Y1+rho*(Q'*L2-B);
      rho=min(rho*p,1e2);
      A=KXXinverse*Q;
      %% Classification
      Ztrain=A'*KXX;
      Ztest=A'*KXY;
      Ypseudo=classifyKNN(Ztrain,trainY,Ztest,1);
      results=MyClusteringMeasure(testY,Ypseudo,1);%[ACC ACC2 MIhat Purity]';
       for index=1:3
           results_iter(index,i)=results(index);
       end
       fprintf('[%d]-th acc:%.4f, MIhat: %.4f, Purity:%.4f\n',i,...
            results(1),results(2),results(3));
    end
end

