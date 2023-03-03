function [x, out] = fminTR(x,fun, hess, opts, varargin)
if (nargin < 3); error('[x, out] = fminTR(fun, x, opts)'); end
if (nargin < 4); opts = []; end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts,'eta1');     opts.eta1 = 1e-2;   end%'$\rho^k$ ���½磨�������˽�ʱ��ζ��������뾶��Ҫ��С���ܾ����£�'
if ~isfield(opts,'eta2');     opts.eta2 = 0.9;    end%'$\rho^k$ ���Ͻ磨�������˽�ʱ��ζ��������뾶��Ҫ���󲢽��ܸ��£�'
if ~isfield(opts,'gamma1');   opts.gamma1 = .25;  end%'ÿ�ε���������뾶��С�ı���'
if ~isfield(opts,'gamma2');   opts.gamma2 = 10;   end%'ÿ�ε���������뾶����ı���'
if ~isfield(opts,'maxit');    opts.maxit = 200;   end
if ~isfield(opts,'record');   opts.record = 0;    end
if ~isfield(opts,'itPrint');  opts.itPrint = 1;   end
if ~isfield(opts,'verbose');  opts.verbose = 0;    end
% '�ӽṹ���и��Ʋ���'
maxit   = opts.maxit;   record  = opts.record;  itPrint = opts.itPrint;
gtol    = opts.gtol;    ftol    = opts.ftol;
eta1    = opts.eta1;    eta2    = opts.eta2;
gamma1  = opts.gamma1;  gamma2  = opts.gamma2;
% '����׼��,�����ʼ��x���ĺ���ֵ���ݶ�'?
out = struct();
[f,g] = fun(x);
out.nfe = 1;
nrmg = norm(g,2);
out.nrmG = nrmg;
fp = f; gp = g;
% '���������������ýضϹ����ݶȷ���⣬���ýṹ��opts_tCGΪ���ṩ����'?
opts_tCG = struct();
% '��ʼ��������뾶����ʼ�趨Ϊ�ṩ�Ĳ���ֵ��Ĭ��ֵ$\sqrt{\mathrm{len}(x)}/8$ '
% '���� $\Delta$ ���Ͻ�$\bar{\Delta}$ Ϊ$\sqrt{\mathrm{len}(x)}$ '
Delta_bar = sqrt(length(x));
Delta0 = Delta_bar/8;
if isfield(opts,'Delta')
    Delta = opts.Delta;
else
    Delta = Delta0;
end
% '���������ֱ��¼������뾶����������С�Ĵ������Է����ʼֵ����'
consecutive_TRplus = 0;
consecutive_TRminus = 0;
% '����Ҫ��ϸ���ʱ���趨�����ʽ'?
if record >= 1
    if ispc; str1 = '  %10s'; str2 = '  %8s';
    else     str1 = '  %10s'; str2 = '  %8s'; end
    stra = ['%5s', str1, str2, str2, str2, str2, str2, str2, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'F', 'fdiff', 'mdiff', 'redf', 'ratio', 'Delta', 'nrmg');
    fprintf('%s', str_head);
    str_num = ['%4d  %+8.7e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2.1e  %2.1e'];
end
% '������ѭ������maxitΪ����������'
for iter = 1:maxit
% '�ضϹ����ݶȷ��������ж������Ĳ���'?
    opts_tCG.kappa = 0.1;
    opts_tCG.theta = 1;
 % '����tCG���������ýضϹ����ݶȷ���������������⣬�õ���������d�� stop_tCG��ʾ�ضϹ����ݶȷ����˳�ԭ��'?
     hess_tCG = @(d) hess(x,d);
    [d, Hd, out_tCG] = tCG(x, gp, hess_tCG, Delta, opts_tCG);
    stop_tCG = out_tCG.stop_tCG;
%'�����ֵ $$\displaystyle\rho_k=\frac{f(x^k)-f(x^k+d^k)}{m_k(0)-m_k(d^k)},$$'
%'��ȷ���Ƿ���µ���������������뾶'
%'���ȼ��� $m_k(0)-m_k(d^k)=-\nabla f(x^k)^\top d^k + \frac{1}{2}(d^k)^\top B^k d^k$'
%'Ϊ�˱�֤��ֵ�ȶ���,����һ��С����rreg'     
    mdiff = g'*d + .5*d'*Hd;
    rreg = 10*max(1, abs(fp)) * eps;
    mdiff = -mdiff + rreg;
    model_decreased = mdiff > 0;
%'����ʵ���\hat{x}^{k+1}=x^k+d^k$��������õ㴦���㺯��ֵ���ݶ�'
    xnew = x + d;
    [f,g] = fun(xnew);
    nrmg = norm(g,2);
    out.nfe = out.nfe + 1;
%'���� $f(x^k)-f(x^k+d^k)$��ͬ��������һ��С����rreg��Ȼ����� $\rho^k$��Ϊ����������뾶���ж��Ƿ���µ�����'
     redf = fp - f + rreg;
    ratio = redf/mdiff;
%' ��$\rho^k>\eta_1$ ʱ�����ܴ˴θ��£�����¼��һ���ĺ���ֵ���ݶ�?' ?    
    if ratio >= eta1
        x = xnew;   fp = f;  gp = g;
        out.nrmG = [out.nrmG; nrmg];
end 
% '���㺯��ֵ��Ա仯'
    fdiff = abs(redf/(abs(fp)+1));
 % 'ͣ��׼�򣺵������ݶȷ���С����ֵ����ֵ����Ա仯С����ֵ�� $\rho^k>0$ ʱ��ֹͣ����'   
	 cstop = nrmg <= gtol || (abs(fdiff) <= ftol && ratio > 0);
  %'�ڴﵽ����ʱ����ʼ����ʱ����������������ʱ��ÿ���ɲ�ʱ����ӡ��ϸ���?
    if record>=1 && (cstop || ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if mod(iter,20*itPrint) == 0 && iter ~= maxit && ~cstop
            fprintf('\n%s', str_head);
        end
%'stop_tCG| ��¼�ڲ�ĽضϹ����ݶȷ��˳���ԭ�򣬷ֱ��Ӧ������� '         
   switch stop_tCG
            case{1}
                str_tCG = ' [negative curvature]\n';
            case{2}
                str_tCG = ' [exceeded trust region]\n';
            case{3}
                str_tCG = ' [linear convergence]\n';
            case{4}
                str_tCG = ' [superlinear convergence]\n';
            case{5}
                str_tCG = ' [maximal iteration number reached]\n';
            case{6}
                str_tCG = ' [model did not decrease]\n';
        end
        
        fprintf(strcat(str_num,str_tCG), ...
            iter, f, fdiff, mdiff, redf, ratio, Delta, nrmg);
    end
% '������ͣ��׼��ʱ����¼�ﵽ����ֵ���˳�ѭ��?'
    if cstop
        out.msg = 'optimal';
        break;
    end
 % '������뾶�ĵ���'
 % '��� $\rho^k<\eta_1$ ���㷨�ǽ���${m_k(0)-m_k(d^k)}\approx 0$ԼΪ0 ʱ�������ܵ�ǰ������'
 %'���������,��������뾶��������������ԣ��� $\Delta \leftarrow \gamma_1\Delta$��'
 %'��������뾶���������������Ϊ�㣬��С������һ '   
 if ratio < eta1 || ~model_decreased || isnan(ratio)
        Delta = Delta*gamma1;
        consecutive_TRplus = 0;
        consecutive_TRminus = consecutive_TRminus + 1;
% '��������뾶���� 5 �μ�Сʱ����Ϊ��ǰ��������뾶���󣬲������Ӧ����ʾ��Ϣ'?   ?       
 if consecutive_TRminus >= 5 && opts.verbose >= 1
            consecutive_TRminus = -inf;
            fprintf(' +++ Detected many consecutive TR- (radius decreases).\n');
            fprintf(' +++ Consider decreasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
% '��$\rho_k>\eta_2$ ��$\|d_k\|=\Delta$����ӦΪ�ضϹ����ݶȷ������������ʻ��߳���������뾶����ֹ��,����������뾶'
%'$\Delta \leftarrow \min\{\gamma_2\Delta, \bar{\Delta}\}$,��������뾶��������С�������㣬���������һ'? ?   
 elseif ratio > eta2 && (stop_tCG == 1 || stop_tCG == 2)
        Delta = min(gamma2*Delta, Delta_bar);
        consecutive_TRminus = 0;
        consecutive_TRplus = consecutive_TRplus + 1;
% '���ǵ�������뾶���� 5 ������ʱ����Ϊ��ǰ��������뾶��С�������Ӧ����ʾ��Ϣ'��?
        if consecutive_TRplus >= 5 && opts.verbose >= 1
            consecutive_TRplus = -inf;
            fprintf(' +++ Detected many consecutive TR+ (radius increases).\n');
            fprintf(' +++ Consider increasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', Delta_bar, Delta0);
        end
% '���������������������������뾶���е�����������������ͼ�С����������'?
    else
        consecutive_TRplus = 0;
        consecutive_TRminus = 0;
    end
end
% '�����������˳�ʱ����¼�˳���Ϣ'??
out.iter = iter;
out.f    = f;
out.nrmg = nrmg;
end