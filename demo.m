%% Add path
addpath('./utils/');
addpath(genpath('./algorithm/'));
rng(1);
%% Load COIL20
path='./COIL20.mat';
load(path,'X','Y');
X=X';
X=L2Norm(X')';
%% Split dataset
%%% select 'number' samples from each class as a training set, and use
%%%     as a training set
number=25;
[X1,Y1,X2,Y2] = splitData(X,Y,number);
%% Select a algorithm
algorithm=@KSR_GL3;
%% Set the hyper-parameters
%%% Notice: you should modify `options`, so as to tune the hyper-parameters
options=defaultOptions([],...
                    'T',10,...    %% The iterations
                    'dim',100,...         %% The dimension reduced
                    't',10,...           %% The maximum iteration of GPI
                   'lambda1',2,...       %% The weight of L2,1 norm w.r.t Z
                   'lambda2',1e-4,...    %% The weight of kernel norm w.r.t GL3
                   'ker',4,...           %% The kernel, 1 = linear, 2 = rbf
                   'gamma',1,...         %% The parameter of rbf kernel
                   'rho',0.01,...         %% The weight of Lagrange
                   'p',1.01);            %% The increase rate of Lagrange
%% Run the algorithm
res=[];
for i=1:20
    [res(:,i),~,~]=algorithm(X1,Y1,X2,Y2,options);
end
res=mean(res,2)*100;
fprintf('Mean acc:%.4f, NMI:%.4f, Purity:%.4f\n',res(1),res(2),res(3));
