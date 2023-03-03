dataset = 'a9a.test';
[b,A] = libsvmread(dataset);%'��libsvm���ݼ�a9a�ϵ�ʵ�飬libsvmreadΪ�������еĶ������'
[m,n] = size(A);
mu = 1e-2/m;
%'���ò���'
opts = struct();
opts.xtol = 1e-8;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.verbose = 0;
fun = @(x) lr_loss(x,mu,A,b);%'�������,����fun���߼��ع麯��'
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);%'�������������hess�õ����Ǻ�ɪ�����뷽��u�ĳ˻�'

[x1,out1] = fminNewton(x0,fun,hess,opts);%'����ţ�ٷ����'
%'��CINA���ݼ��ϵ�ʵ��'
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
[m,n] = size(A);
fun = @(x) lr_loss(x,mu,A,b);
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);
[x2,out2] = fminNewton(x0,fun,hess,opts);
%'��ijcnn1���ݼ��ϵ�ʵ��'
dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);
Atran = A';
[m,n] = size(A);
mu = 1e-2/m;
fun = @(x) lr_loss(x,mu,A,b);
x0 = zeros(n,1);
hess = @(x,u) lr_hess(x,u,A,b,mu);
[x3,out3] = fminNewton(x0,fun,hess,opts);
%'��Ŀ�꺯���ݶȷ�����������ı仯���ӻ�'
fig = figure;
semilogy(0:out1.iter, out1.nrmG, '-o', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
hold on
semilogy(0:out2.iter, out2.nrmG, '-.*', 'Color',[0.99 0.1 0.2], 'LineWidth',1.8);
hold on
semilogy(0:out3.iter, out3.nrmG, '--d', 'Color',[0.99 0.1 0.99], 'LineWidth',1.5);
legend('a9a','CINA','ijcnn1');
ylabel('$\|\nabla \ell_(x^k)\|_2$', 'fontsize', 14, 'interpreter', 'latex');
xlabel('������');
print(fig, '-depsc','lr_newton.eps');
%'�߼��ع����ʧ��������Ϊ�Ż������Ŀ�꺯��'
function [f,g] = lr_loss(x,mu,A,b)
[m,n] = size(A);
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,2)^2;
%'nargout�ں����ڣ����ڻ�ȡʵ�����������������nargout(fun):��ȡfunָ����������������������������'
if nargout > 1
    g = Atran*(b./(1+expba) - b)/m + 2*mu*x;
end
end
%'Ŀ�꺯���ĺ�ɪ����Ҫ���ṩ��ǰ�Ż�����x�ͷ��򣬷��غ�ɪ������x�������ڷ���u�ϵ�ֵ'
function H = lr_hess(x,u,A,b,mu)
[m,n] = size(A);
Ax = A*x;
Atran = A';
expba = exp(- b.*Ax);
p = 1./(1 + expba);
w = p.*(1-p);
H = Atran*(w.*(A*u))/m + 2*mu*u;
end