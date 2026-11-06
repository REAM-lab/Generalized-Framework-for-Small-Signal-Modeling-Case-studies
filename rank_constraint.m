phi = [1.       0.       0.       0.       0.       0. ;
       0.       1.       0.       0.       0.       0. ;
       0.       0.       1.       0.      -1.       0. ;
       0.       0.       0.       1.       0.      -1  ;
       0.       0.      -1       -0.       1.       0. ;
       0.       0.      -0.      -1        0.       1. ];

B_stack = [376.9911184e1 0        -376.9911184e1 0         0 0;
           0        376.9911184e1 0         -376.9911184e1 0 0;
           0        0        0         0         10 0;
           0        0        0         0         0 10];


m = 6;
n = 4;
T = sdpvar(n,m, 'full');
E = sdpvar(n,n);
v = sdpvar(n, 1);

%F = [rank([E])<=2, T * phi == E * B_stack];
constraints = [T * phi == E * B_stack, E>=0, trace(E)<=100];

optimize(constraints,-trace(E), sdpsettings('solver', 'mosek'))

%optimize(F,1,sdpsettings('lmirank.solver','mosek'))
E= value(E);
rank(E)
E
T = value(T)

A_stack = [  -376.9911184  376.99111843    0.            0.        ;
            -376.99111843  -376.9911184    0.            0.        ;
              0.            0.            0.          376.99111843;
              0.            0.         -376.99111843    0.        ];
C_stack = eye(4);

F = [zeros(2,2) zeros(2,2);
     zeros(2,2) eye(2);
     eye(2)     zeros(2,2)];
G = [eye(2);
     zeros(2,2);
     zeros(2,2)];

A_sys = E * A_stack + T * F * C_stack;
B_sys = T * G;
lambda = eig(A_sys, round(E,10))

% provide eigenvalues less than 1e5
lambda_finite = lambda((lambda > -1e5) & (lambda <1e9))

sys = dss(A_sys, B_sys, eye(4), zeros(4,2), round(E,1));

t = 0:0.01:2;
v_ref_d = zeros(size(t));
v_ref_d(t >= 1) = 1;

v_ref_q = zeros(size(t));
u = [v_ref_d; v_ref_q].';

% Simulate the system response
[y, t, x] = lsim(sys, u, t);


subplot(2,1,2);
plot(t, y(3), 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Output y(t)');
title('Descriptor System Response');