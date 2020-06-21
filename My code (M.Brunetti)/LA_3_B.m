%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Lorenz trajectory Vs. transitioning forecast by triggering force %%%%%

%% DATA INITIALIZATION

clear all, close all
dt = 0.01; T = 10; t = 0:dt:T; 
b = 8/3; sig = 10; r = 28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);  

%% ODE SOLUTION AND TRIGGERING FORCE COMPUTATION

x0 = 30*(rand(3,1)-0.5); 
[t,y] = ode45(Lorenz,t,x0);
yx = y(:,1);
yxT = yx';
        
d = 15; % max delay [0..m-1] for Hankel matrix
m = length(yxT); % nummber of snapshots
H=[];
for jj = 1:(d+1)
    H = [H; yxT(:,jj:m-(d+1)+jj)]; 
end              

[U,Sigma,V] = svd(H,'econ'); % singular values decomposition
VT = V';
trigger = VT(d,:);
flag = trigger';

%% RESULTS PLOTTING

figure(1); 

subplot(2,2,1);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro'), hold on
title(['Lorenz 3D trajectory, rho = ' num2str(r)]);
xlabel('X');
ylabel('Y');
zlabel('Z');
view(-23,18)
grid on;

subplot(2,2,2);   
plot(t,yx), hold on
title('Trajectory projection');
xlabel('Time');
ylabel('X');
grid on;

subplot(2,2,3);   
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title('Singular values');
xlabel('Identifier')
ylabel('Value (normalized)')
grid on;

subplot(2,2,4);
plot(V(:,1)), hold on;
plot(V(:,d).^2), hold on; 
title('Triggering force');
legend({'Mode 1',['Mode ' num2str(d)]},'Location','southeast');
xlabel('Time');
ylabel('Amplitude')
grid on;   



