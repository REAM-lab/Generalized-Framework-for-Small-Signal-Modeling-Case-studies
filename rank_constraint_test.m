A = [0.2 0 1 0;0 0.2 0 1;-1 1 0.2 0;1 -1 0 0.2];
B = [0 0 1 0]';
C = [0 1 0 0];

X = sdpvar(4,4);
Y = sdpvar(4,4);

Bp = null(B')';
Cp = null(C)';
W  = eye(3)*1e-6;
F = [Bp*(A*X+X*A')*Bp' <= -W, Cp*(Y*A+A'*Y)*Cp' <= -W];

F = [F, [X eye(4);eye(4) Y] >= 0];   % not needed, see below
F = [F, rank([X eye(4);eye(4) Y]) <= 4+2];


optimize(F,[],sdpsettings('lmirank.solver','mosek'))
eig(value([X eye(4);eye(4) Y]))

rank([X eye(4);eye(4) Y])