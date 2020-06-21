%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% NN training on single solution, then prediction for different IC %%%%%

%% DATA INITIALIZATION

clear all; close all; clc

N = 2049; % original state dimension 
N = 64; % rescaled state dimension
ic = randn(N,1); % Initial condition
[t,x,u] = KS_solver(ic); % associated numerical solution
[m,n] = size(u); % state dimension (n), number of snapshots (m)


%% NN TRAINING 

% Creating a training set from the numerical solution
input = u(1:m-1,:);
output = u(2:m,:);

% Feed Forward (non-linear) NN creation and training 
HL = [5 5 5];
net = feedforwardnet(HL);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
[net,tr] = train(net,input',output');

% NN reconstruction  
u_nn(1,:) = ic; 
for i = 2:m
    y = net(ic);
    u_nn(i,:) = y'; 
    ic = y; 
end

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

subplot(2,3,1)
plot(x,u(1,:),'k');
title('Kuramoto-Sivashinsky initial conditions');
xlabel('Space');
ylabel('Conditon');
grid ON;

subplot(2,3,2)
pcolor(x,t,u),shading interp, colormap(hot);
title('Kuramoto-Sivashinsky sol. for training');
xlabel('Space');
ylabel('Time');

subplot(2,3,3)
pcolor(x,t,u_nn),shading interp, colormap(hot);
title(['Neural Network reconstruction, FF [' num2str(HL) '] layers']);
xlabel('Space');
ylabel('Time');

subplot(2,3,4)
plot(x2,u2(1,:),'k');
title('Kuramoto-Sivashinsky initial conditions');
xlabel('Space');
ylabel('Conditon');
grid ON;

subplot(2,3,5)
pcolor(x2,t2,u2),shading interp, colormap(hot);
title('Kuramoto-Sivashinsky sol. for testing');
xlabel('Space');
ylabel('Time');

subplot(2,3,6)
pcolor(x2,t2,u_nn2),shading interp, colormap(hot);
title(['Neural Network forecast FF [' num2str(HL) '] layers']);
xlabel('Space');
ylabel('Time');

figure(2);
plotperform(tr);
title('Feed Forward Neural Network training performance');


