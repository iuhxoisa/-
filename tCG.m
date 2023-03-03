%'������뾶Ϊ������ʱ��������������͵�ͬ�����ţ�ٷ���'
%'������Ϣ��������$x$���ݶ�grad����ɪ����hess��������뾶 $\Delta$���㷨�����ṹ��opts'
%'�����Ϣ��������������Ľ�$\eta$���������㴦�Ĺ���½�������'
%'HetaΪ��ɪ�����ڵ�$x$�����ڷ���$\eta$�ϵĽ��'
%'�� $\nabla^2f(x^k)\eta^k$��������Ϣ�ṹ��out���˳�ԭ��stop_tCG'
function [eta, Heta, out, stop_tCG] ...
    = tCG(x, grad, hess, Delta, opts)
if ~isfield(opts,'kappa');     opts.kappa = 0.1;   end%'ţ�ٷ�����⾫�ȵĲ���'
if ~isfield(opts,'theta');     opts.theta = 1;   end%'ţ�ٷ����󾫾��ȵĲ���'
if ~isfield(opts,'maxit');     opts.maxit = length(x);   end%'���������������ڹ����ݶȷ�����Ĭ�ϵ��ڱ�����ά��'
if ~isfield(opts,'minit');     opts.minit = 5;   end%'��С��������'
%'��������'
theta = opts.theta;
kappa = opts.kappa;
%'��ʼ����eta�������Ϊ�Ż���������ʼΪȫ0����$\mathbf{0}$�� $r_0$ '
%'��ʼ��ΪĿ�꺯�����ݶ� $\nablaf(x^k)$'
eta = zeros(size(x));
Heta = eta;
r = grad;%'Ŀ�꺯�����ݶ�'
e_Pe = 0;
r_r = r'*r;
norm_r = sqrt(r_r);
norm_r0 = norm_r;
%'�����ݶȷ���ʼʱ�̵Ĺ���� $p^0=-g$ ���ڴ�������mdelta��ʾ $-p^k$ (minus delta)'
mdelta = r;
d_Pd = r_r;
e_Pd = 0;
%'�����ݶȷ��Ż���Ŀ�꺯��'
model_fun = @(eta, Heta) eta'*grad + .5*(eta'*Heta);
model_value = 0;
%'������������������ֹͣʱ��Ϊstop_tCG = 5'
stop_tCG = 5;
%'������ѭ��'
for j = 1 : opts.maxit
    Hmdelta = hess(mdelta);  %'$g$ ��ʾ�ݶȣ�$H$ ��ʾ��ɪ���󡣴˴�����$-Hp^k$'  
    d_Hd = mdelta'*Hmdelta;  %'��������$(p^k)^\top Hp^k$'
    alpha = r_r/d_Hd;  %'�����ݶȷ��Ĳ���$\displaystyle\alpha_{k}=\frac{\|r^k\|_2^2}{(p^k)^\topHp^k}$'
    %'�µĹ����$\eta^{k+1}=\eta^{k}+\alpha_{k} p^k$'
    %'����'$(\eta^{k+1})^\top \eta^{k+1}=\alpha_{k}^2(p^k)^\top p^k+2\alpha_{k}p^k\eta^k+(\eta^k)^\top \eta^k$
    e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
   %'���ʡ�0$(p^{k})^\top Hp^{k}\le0$ ˵����ɪ�����������������ȥ�õ��ķ���һ�����½����򣩣���ֹͣ�����ݶ��㷨'
   %'$\|\eta^{k+1}\|_2^2\ge\Delta^2$ ʱ�������㳬��������߽磬ֹͣ�㷨'
   %'����$\tau$ ʹ�� $\|\eta^k+\tau p^k\|_2^2=\Delta^2$'
   %'��$\eta=\eta^k+\tau p^k$ Ϊ���յĵ������������$H\eta^{k+1}=H\eta^k -\tauH\eta^k$'
   if d_Hd <= 0 || e_Pe_new >= Delta^2
        tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;
        eta  = eta - tau*mdelta;
        Heta = Heta -tau*Hmdelta;
   %'�Ը������˳���Ϊ1���Գ����߽��˳���Ϊ2.�˳�����'
        if d_Hd <= 0
            stop_tCG = 1;
        else
            stop_tCG = 2;
        end
        break;
   end
  %'����$\eta^{k+1}$'   
    e_Pe = e_Pe_new;
    new_eta  = eta-alpha*mdelta;
    new_Heta = Heta -alpha*Hmdelta;
%'����$\eta^{k+1}$���ĺ���ֵ'
%'���º��Ŀ�꺯��ֵû���½����˳��������ܾ��ò����¡�stop_tCG==6��ʾĿ�꺯��ֵ�ǽ���ֹͣ'
    new_model_value = model_fun(new_eta, new_Heta);
    if new_model_value >= model_value
        stop_tCG = 6;
        break;
    end
% '���û�дﵽ����������������ܸ��µ� $\eta^{k+1}$�������µ�$\eta^{k+1}$ ���±���' 
    eta = new_eta;
    Heta = new_Heta;
    model_value = new_model_value;
% '���²в� $r^{k+1}=r^{k}+\alpha_{k}H p^k$�����䷶�����ر�ļ�¼��һ����$\|r^k\|_2$Ϊr_rold ' 
    r = r -alpha*Hmdelta;
    r_rold = r_r;
    r_r = r'*r;
    norm_r = sqrt(r_r);
%��׼�Ĺ����ݶȷ����������������ﵽ��С������������$\|r\|_2\le\|r_0\|_2\min(\kappa,\|r_0\|_2^\theta)$ ʱ����Ϊ�㷨������ֹͣ����
%��� $\kappa < \|r_0\|^\theta$����˵������ $\|r^k\|_2\le\kappa\|r^0\|_2$
%Ϊ���ϸ��������˵����ʱ���ţ�ٷ����������򷽷��������������׶�
%��֮������$\|r^k\|_2\le\|r_0\|^{1+\theta}$ ���ϸ�˵����ʱ $\|r^0\|$
%�Ѿ���С����ʱ��Ӧ���ţ�ٷ��������򷽷����ڳ����������׶�
if j >= opts.minit && norm_r <= norm_r0*min(norm_r0^theta, kappa)
        if kappa < norm_r0^theta
            stop_tCG = 3;
        else
            stop_tCG = 4;
        end
        break;
    end
    % �����µ���������$\beta_k=\frac{\|r^{k+1}\|^2}{\|r^k\|^2}$,
    % $p^{k+1}=-r^{k+1}+\beta_k p^k$?     
    beta = r_r/r_rold;
    mdelta = r + beta*mdelta;
    % ���� $\eta^{k+1}p^{k+1}=(\eta^k+\alpha_k p^k)^\top(-r^{k+1}+\beta_{k+1}p^k)
    % =\beta_{k+1}(p^k)^\top(\eta^k+\alpha_k p^k)$����������% $\eta^0=\mathbf{0}$ ��$\eta^k$ Ϊ$p^0,\dots,p^k$ ��������ϣ�
    %���� $(r^{k+1})^\top p^j=0,\ j=0,1,\dots,k$ ��֪ $(r^{k+1})^\top (\eta^k+\alpha_k p^k)=0$
    % �Լ� $(p^{k+1})^\top p^{k+1}=(-r^{k+1}+\beta_k p^k)^\top(-r^{k+1}+\beta_k p^k)$��
    %ע�⵽ $(r^{k+1})^\top p^k=0$���� $(p^{k+1})^\top p^{k+1}=(r^{k+1})^\top r^{k+1}+\beta_k^2(p^k)^\top p^k$  
    e_Pd = beta*(e_Pd + alpha*d_Pd);
    d_Pd = r_r + beta*beta*d_Pd;
end
% �˳�ѭ������¼�˳���Ϣ?
out.iter = j;
out.stop_tCG = stop_tCG;
end