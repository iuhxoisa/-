dataset = 'a9a.test';
[b,A] = libsvmread(dataset);%'入libsvm数据集a9a上的实验，libsvmread为另外运行的读入程序'
[m,n] = size(A);
mu = 1e-2/m;
%'设置参数'
opts = struct();
opts.xtol = 1e-8;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.record  = 1;
opts.verbose = 1;
opts.Delta = sqrt(n);
fun = @(x) lr_loss(A,b,m,x,mu);%'函数句柄,这里fun是逻辑回归函数'
x0 = zeros(n,1);
hess = @(x,u) hess_lr(A,b,m,mu,x,u);%'函数句柄，这里hess得到的是海瑟矩阵与方向u的乘积'

[x1,out1] = fminTR(x0,fun,hess,opts);%'调用信赖域法求解'
%'在CINA数据集上的实验'
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
fun = @(x) lr_loss(A,b,m,x,mu);
x0 = zeros(n,1);
hess = @(x,u) hess_lr(A,b,m,mu,x,u);
[x2,out2] = fminTR(x0,fun,hess,opts);
%'在ijcnn1数据集上的实验'
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(A,b,m,x,mu);
x0 = zeros(n,1);
hess = @(x,u) hess_lr(A,b,m,mu,x,u);
[x3,out3] = fminTR(x0,fun,hess,opts);
%'将目标函数梯度范数随迭代步的变化可视化'
fig = figure;
semilogy(0:out1.iter, out1.nrmG, '-o', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:out2.iter, out2.nrmG, '-.*', 'Color',[0.99 0.1 0.2], 'LineWidth',1.8);
hold on
semilogy(0:out3.iter, out3.nrmG, '--d', 'Color',[0.99 0.1 0.99], 'LineWidth',1.5);
legend('a9a','CINA','ijcnn1');
ylabel('$\|\nabla \ell_(x^k)\|_2$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','lr_tr.eps');
%'逻辑回归的损失函数，作为优化问题的目标函数'
function [f,g] = lr_loss(A,b,m,x,mu)
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,2)^2;
%'nargout在函数内，用于获取实际输出变量个数。（nargout(fun):获取fun指定函数所定义的输出变量个数。）'
if nargout > 1
   g = Atran*(b./(1+expba) - b)/m + 2*mu*x;
end
end
%'目标函数的海瑟矩阵，要求提供当前优化变量x和方向，返回海瑟矩阵在x处作用在方向u上的值'
function H = hess_lr(A,b,m,mu,x,u)
    Ax = A*x;
    Atran = A';
    expba = exp(- b.*Ax);
    p = 1./(1 + expba);
    w = p.*(1-p);
    H = Atran*(w.*(A*u))/m + 2*mu*u;
end