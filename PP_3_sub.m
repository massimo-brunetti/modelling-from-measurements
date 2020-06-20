%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Cost function: return the error from raw data, on input parameters %%%

function error = PP_3_sub(P)

% xdot = (b-p*y)*x;
% ydot = (r*x-d)*y;

%% DATA INITIALIZATION

% time base
t1 = 1845;
t2 = 1903;
dt = 2;
t = (t1:dt:t2); % historical time base 
te = (t1:dt:t2+dt*length(t)); % extended time base 

% hare status
xa = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70];
xb = [100 92 70 10 11 137 137 18 22 52 83 18 10 9 65]; 
x = [xa xb];

% lynx status
ya = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20];
yb = [34 45 40 15 15 60 80 26 18 37 50 35 12 12 25]; 
y = [ya yb];

% system status
X = [x; y];
[n, m] = size(X); % state dimension, number of snapshots

%% ERROR FUNCTION

LotkaVolterra = @(t,x)([P(1)*x(1)-P(2)*x(1)*x(2); ...
                        P(3)*x(1)*x(2)-P(4)*x(2)]);
[~,Xn] = ode23(LotkaVolterra,t,X(:,1));

errx = Xn(:,1)-x';
erry = Xn(:,2)-y';
error = errx'*errx+erry'*erry;

end