function [x, out] = fminNewton(x, fun, hess, opts, varargin)
%'nargin��ʾ��������Ĳ�������������3ʱ��������3ʱ��Ϊ�ṹ��optsΪ�սṹ�壬������Ĭ�ϲ���'
if (nargin < 3); error('[x, out] = fminNewton(x, fun, hess, opts)');end
if (nargin < 4); opts = []; end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts,'maxit');    opts.maxit = 200;   end
if ~isfield(opts,'verbose');   opts.verbose = 0;    end%'verbose��11ʱ���������Ϣ'
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end%'���������һ�ε�����Ϣ'
%'�ӽṹ��opts�и��Ʋ���'
maxit   = opts.maxit;   verbose  = opts.verbose;  itPrint = opts.itPrint;
xtol    = opts.xtol;    gtol    = opts.gtol;    ftol    = opts.ftol;
%'parslsΪ�������������ɵĽṹ��'
parsls.ftol = opts.rho1;  parsls.gtol = opts.rho2;
%'�����ʼ��x�ĺ���ֵ���ݶ�ֵ'
[f,g] = fun(x);
nrmg = norm(g,2);
nrmx = norm(x,2);
%'����ṹ��'
out = struct();
out.msg = 'MaxIter';%'out.msg�Ƿ�ﵽ������MaxIter����û�дﵽ��������Ϊ���������������˳�'
out.nfe = 1;%'���ú�������'
out.nrmG = nrmg;%'�����˳�ʱ�ݶȷ���'
%'����Ҫ��ϸ���ʱ���趨�����ʽ'
if verbose >= 1
    if ispc; str1 = '  %10s'; str2 = '  %8s';
    else     str1 = '  %10s'; str2 = '  %8s';
    end
    stra = ['%5s', str1, str2, str2, str2, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'F', 'nrmg', 'fdiff', 'xdiff');
    fprintf('%s', str_head);
    str_num = ['%4d  %+8.7e  %+2.1e  %+2.1e  %+2.1e\n'];
end
%'�Ǿ�ȷţ�ٷ�ʹ�ù����ݶȷ����ţ�ٷ��̣��ṹ��opts_tCGΪ�����ݶȷ��ṩ����'
opts_tCG = struct();
%'������ѭ��'
for iter = 1:maxit
    %'��¼��һ��������Ϣ�������㣬����ֵ���ݶȣ�������ķ�����'
    fp = f;
    gp = g;
    xp = x;
    nrmxp = nrmx;
    %'���ýضϹ����ݶȷ�tCG(����ȷ�����ţ�ٷ��̣��õ�ţ�ٷ���d'
    opts_tCG.kappa = 0.1;
    opts_tCG.theta = 1;
    hess_tCG = @(d) hess(x,d);%'�������'
    [d, ~, ~] = tCG(x, gp, hess_tCG, inf, opts_tCG);
    %'�ط���d��������'
    workls.task = 1;%'��ʼ����������workls.task��Ϊ1����workls.task��ָʾ��һ��ִ�еĲ�����'
    deriv = d'*g;%'��ǰ����������ĵ���'
    normd = norm(d);
    
    stp = 1;%'����'
    while 1
        %ls_csrchÿ�ε���ִֻ����������һ��
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
        %'workls.taskΪ2ʱ��Ҫ���¼��㵱ǰ��ĺ���ֵ�ݶ�'
        if (workls.task == 2)
            x = xp + stp*d;
            [f,  g] = feval(fun, x, varargin{:});
            out.nfe = out.nfe + 1;
            deriv = g'*d;
        else
            break
        end
    end
    nrms = stp*normd;%'nrms��ʾx(k+1)-x(k)�ķ���'
    xdiff = nrms/max(nrmxp,1);%'���������Ա仯��'
    nrmg = norm(g,2);%'x(k)�����ݶȷ���'
    out.nrmG = [out.nrmG; nrmg];
    nrmx = norm(x,2);%'x�ķ���'
    out.nfe = out.nfe + 1;%'���ú�������'
    fdiff = abs(fp-f)/(abs(fp)+1);%'����ֵ��Ա仯��'
    %'cstopͣ���жϣ��ݶ�С����ֵ����ֵ��Ա仯����x����Ա仯��С����ֵʱ������ֹͣ'
    cstop = nrmg <= gtol || (abs(fdiff) <= ftol && abs(xdiff) <= xtol);
    %'�ڴﵽ����ʱ����ʼ����ʱ����������������ʱ��ÿ���ɲ�ʱ����ӡ��ϸ���'
    if verbose>=1 && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if mod(iter,20*itPrint) == 0 && iter ~= maxit && ~cstop
            fprintf('\n%s', str_head);
        end    
        fprintf(str_num, ...
            iter, f, nrmg, fdiff, xdiff);
    end
    %'��������ʱ�����Ϊ�ﵽ����ֵ�˳�'
    if cstop
        out.msg = 'Optimal';
        break;
    end
end
%'�����˳�ʱ����¼�����Ϣ'
out.iter = iter;
out.f    = f;
out.nrmg = nrmg;
end