%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NN training on Lorenz trajectory set (multiple rho), then forecast %%%&

%% DATA INITIALIZATION

clear all, close all
mr = 10; % number of trajectories for training 
dt = 0.01; T = 10; t = 0:dt:T; % trajectories time base
b = 8/3; sig = 10; % Lorenz attractor parameters
R1 = [10 28 40]; % rho values for training (the higher, the more chaotic)
R2 = [17 35]; % rho values for testing 
input = []; % input to train the NN
output = []; % output to train the NN 

for i = 1:length(R1)
    
    r = R1(i); 
    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                      r * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);     
    
    figure(1); 
    subplot(3,3,1+3*(i-1));  
    for ii = 1:mr  
        x0 = 30*(rand(3,1)-0.5);
        [~,y] = ode45(Lorenz,t,x0);
        input = [input; y(1:end-1,:)];
        output = [output; y(2:end,:)];               
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')               
    end   
    title(['Lorenz 3D trajectories for training, rho = ' num2str(r)]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(-23,18);
    grid on
    
end

%% NN TRAINING

net = feedforwardnet([5 5 5]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%% ODE SOLUTION Vs NN FORECAST

for i =1:length(R2)  
    
    r = R2(i);
    for ii = 1:2
        
        r = R2(ii); 
        x0 = 20*(rand(3,1)-0.5); % new IC (not in the training)
        [~,y] = ode45(Lorenz,t,x0); % new trajectory (based on physics)
        x0_NN = x0; 
        y_NN(1,:) = x0_NN; 
        
        for j = 2:length(t)
            y0 = net(x0_NN);
            y_NN(j,:) = y0.';
            x0_NN = y0; 
        end

        %% RESULTS PLOTTING         

        figure(1); 
        
        subplot(3,3,1+ii);
        plot(t,y(:,3),'r',t,y_NN(:,3),'b-.')
        title(['ODE solution vs NN forecast, rho = ' num2str(r)])
        xlabel('Time');
        ylabel('Z');
        legend({'ODE','NN'});
        grid on;
    
        subplot(3,3,4+ii)
        plot(t,y(:,2),'r',t,y_NN(:,2),'b-.')
        title(['ODE solution vs NN forecast, rho = ' num2str(r)])
        xlabel('Time');
        ylabel('Y');
        legend({'ODE','NN'});
        grid on;
    
        subplot(3,3,7+ii)
        plot(t,y(:,1),'r',t,y_NN(:,1),'b-.')
        title(['ODE solution vs NN forecast, rho = ' num2str(r)])
        xlabel('Time');
        ylabel('X');
        legend({'ODE','NN'}); 
        grid on;
        
    end   
end
