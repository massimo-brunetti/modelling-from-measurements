%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NN training on multiple solutions, then prediction for different IC %%%

%% DATA INITIALIZATION

clear all; close all; clc

N = 2049; % original state dimension 
N = 64; % rescaled state dimension
mr = [1 10]; % number of time-histories (1st & 2nd batch)

for j = 1:length(mr)
    
    % ODE solution (series of) & training set
    
    input = []; 
    output  = []; 

    for i = 1:mr(j) 
        ic = randn(N,1); 
        [t,x,u] = KS_solver(ic);
        [m,n] = size(u); 
        input = [input; u(1:end-1,:)];
        output = [output; u(2:end,:)];
    end

    figure(1);
    subplot(2,3,1+3*(j-1));
    waterfall(x,t,abs(u)); colormap(hot);
    title(['KS sol. for training, excerpt from a series of ' num2str(mr(j))]);
    xlabel('Space');
    ylabel('Time');
    zlabel('Solution');
    grid on

%% NN TRAINING 

    net = feedforwardnet([5 5 5]);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'radbas';
    net.layers{3}.transferFcn = 'purelin';
    [net,tr] = train(net,input',output'); 

%% ODE SOLUTION Vs NN PREDICTION

    ic2 = randn(N,1); 
    [t2,x2,u2] = KS_solver(ic2);
    [m2,n2] = size(u2);
    
    u_nn2(1,:) = ic2; 
    for i = 2:m2
        y2 = net(ic2);
        u_nn2(i,:) = y2'; 
        ic2 = y2; 
    end
 
 %% RESULTS PLOTTING

    figure(1);
    
    subplot(2,3,2+3*(j-1));
    pcolor(x2,t2,u2), shading interp, colormap(hot);
    title('KS sol.for testing');
    xlabel('Space');
    ylabel('Time');

    subplot(2,3,3+3*(j-1)); 
    pcolor(x2,t2,u_nn2), shading interp, colormap(hot);
    title('NN forecast');
    xlabel('Space');
    ylabel('Time');

end