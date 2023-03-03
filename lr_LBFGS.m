%'设置随机种子'
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

dataset = 'a9a.test';
[b,A] = libsvmread(dataset);%'入libsvm数据集a9a上的实验，libsvmread为另外运行的读入程序'
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(A,b,m,x,mu);%'函数句柄,这里fun是逻辑回归函数'
%'设置参数'
opts = struct();
opts.xtol = 1e-6;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.maxit = 2000;
opts.record  = 0;
opts.m = 5;

x0 = zeros(n,1);
[x1,~,~,out1] = fminLBFGS_Loop(x0,fun,opts);%'调用LBFGS法（双循环递归算法）求解'
%'在CINA数据集上的实验'
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
fun = @(x) lr_loss(x,mu);
x0 = zeros(n,1);
fun = @(x) lr_loss(A,b,m,x,mu);
[x2,~,~,out2] = fminLBFGS_Loop(x0,fun,opts);
%'在ijcnn1数据集上的实验'
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(A,b,m,x,mu);
x0 = zeros(n,1);
[x3,~,~,out3] = fminLBFGS_Loop(x0,fun,opts);
%'将目标函数梯度范数随迭代步的变化可视化'
fig = figure;
k1 = 1:10:out1.iter;
semilogy(k1-1, out1.nrmG(k1), '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
k2 = 1:10:out2.iter;
semilogy(k2-1, out2.nrmG(k2), '-.', 'Color',[0.99 0.1 0.2], 'LineWidth',1.8);
hold on
k3 = 1:10:out3.iter;
semilogy(k3-1, out3.nrmG(k3), '--', 'Color',[0.99 0.1 0.99], 'LineWidth',1.5);
legend('a9a','CINA','ijcnn1');
ylabel('$\|\nabla \ell (x^k)\|_2$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('迭代步');
print(fig, '-depsc','lr_lbfgs.eps');
%'逻辑回归的损失函数，作为优化问题的目标函数'
function [f,g] = lr_loss(A,b,m,x,mu)
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,2)^2;

if nargout > 1
   g = Atran*(b./(1+expba) - b)/m + 2*mu*x;
end
end
