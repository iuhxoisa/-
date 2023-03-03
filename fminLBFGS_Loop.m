function [x, f, g, Out]= fminLBFGS_Loop(x, fun, opts, varargin)
if ~isfield(opts, 'xtol');      opts.xtol = 1e-6; end
if ~isfield(opts, 'gtol');      opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol');      opts.ftol = 1e-16; end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts, 'm');         opts.m  = 5; end%'L-BFGS���ڴ�Դ洢��'
if ~isfield(opts, 'maxit');     opts.maxit  = 1000; end
if ~isfield(opts, 'storeitr');  opts.storeitr = 0; end%'����Ƿ��¼ÿһ�ε�����x'
if ~isfield(opts, 'record');    opts.record = 0; end%'����Ƿ���Ҫ��������Ϣ'
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end%'ÿ���������һ�ε�����Ϣ'
%'��������'
xtol = opts.xtol;
ftol = opts.ftol;
gtol = opts.gtol;
maxit = opts.maxit;
storeitr = opts.storeitr;
parsls.ftol = opts.rho1;
parsls.gtol = opts.rho2;
m = opts.m;
record = opts.record;
itPrint = opts.itPrint;
%'��ʼ���͵���׼���������ʼ�����Ϣ����ʼ��������Ϣ'?
[f,  g] = feval(fun, x , varargin{:});%'ִ����Ϊ���������fun����'
nrmx = norm(x);
Out.f = f;  Out.nfe = 1; Out.nrmG = [];
%'storeitr�ڲ�Ϊ0ʱ����¼ÿһ��x'
if storeitr
    Out.xitr = x;
end
%'SK��YK���ڴ���L-BFGS�㷨�����m����x�ı仯��s�����ݶ�g�ı仯��y��'
n = length(x);
SK = zeros(n,m);
YK = zeros(n,m);
istore = 0; pos = 0;  status = 0;  perm = [];
%'���ô�ӡ��ʽ'?
if record == 1
    if ispc; str1 = '  %10s'; str2 = '  %6s';
    else     str1 = '  %10s'; str2 = '  %6s'; end
    stra = ['%5s',str2,str2,str1, str2, str2,'\n'];
    str_head = sprintf(stra, ...
        'iter', 'stp', 'obj', 'diffx', 'nrmG', 'task');
    str_num = ['%4d  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2d\n'];
end
%'������ѭ��'
Out.msg = 'MaxIter';
for iter = 1:maxit
%'��¼��һ��������Ϣ'?    
    xp = x;   nrmxp = nrmx;
    fp = f;   gp = g;
%'L-BFGS˫ѭ���������½����򡣵�һ�ε������ø��ݶȷ���֮����L-BFGS����������d=-Hg'    
    if istore == 0
        d = -g;
    else
        d = LBFGS_Hg_Loop(-g);
    end
%'��L-BFGS�����õ����½������������������ú���ls_csrch����������'    
    workls.task =1;% '���ȳ�ʼ�����������workls.taskΪ1'
    deriv = d'*g;%'derivΪĿ�꺯���ص�ǰ�½�����ķ�����'
    normd = norm(d);
    stp = 1;
    while 1
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
   % 'ls_csrchÿ�ε���ִֻ����������һ��������workls.taskָʾ��һ��Ӧ��ִ�еĲ���'?     
        if (workls.task == 2)
            x = xp + stp*d;%'�˴�workls.task==2��ζ��Ҫ���¼��㵱ǰ�㺯��ֵ���ݶȵ�'
            [f,  g] = feval(fun, x, varargin{:});
            Out.nfe = Out.nfe + 1;
            deriv = g'*d;
        else
            break%'ֱ����������������ʱ�˳�������ѭ�����õ�����֮���x'?
        end
    end
%'�����������õ��Ĳ���$\alpha_k$���� $s^k=x^{k+1}-x^k=\alpha_kd^k$ '
%'��$\|s^k\|_2=\alpha_k\|d^k\|_2$������$\|s^k\|_2/\max(1,\|x^k\|_2)$��������Ϊ�ж������ı�׼'?    
    nrms = stp*normd;
    diffX = nrms/max(nrmxp,1);
% ���� $\|x^{k+1}\|_2$, $\|g^{k+1}\|_2$����¼һ��������Ϣ?
    nrmG =  norm(g);
    Out.nrmg =  nrmG;
    Out.f = [Out.f; f];
    Out.nrmG = [Out.nrmG; nrmG];
    if storeitr
        Out.xitr = [Out.xitr, x];
    end
    nrmx = norm(x);
% 'ͣ��׼��diffX��ʾ���ڵ�����$x$ ����Ա仯��nrmG��ʾ��ǰ $x$ �����ݶȷ���'%
%'$\displaystyle\frac{|f(x^{k+1})-f(x^k)|}{1+|f(x^k)|}$���Ա�ʾ����ֵ����Ա仯'
%'��ǰ���߾�С����ֵʱ�����ߵ�����С����ֵʱ����Ϊ�ﵽͣ����׼���˳���ǰѭ��'
    cstop = ((diffX < xtol) && (nrmG < gtol) )|| (abs(fp-f)/(abs(fp)+1)) < ftol;
%'����Ҫ��ϸ���ʱ���ڿ�ʼ����ʱ,�ﵽ����ʱ,�ﵽ�������������˳�����ʱ,ÿ���ɲ�����ӡ��ϸ���'
if (record == 1) && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, ...
            iter, stp, f, diffX, nrmG, workls.task);
    end
%'���ﵽ��������ʱ��ֹͣ��������Ϊ�ﵽ����'?     
    if cstop
        Out.msg = 'Converge';
        break;
    end
    % '���� $s^k=x^{k+1}-x^{k}$, $y^k=g^{k+1}-g^{k}$�����õ��� $\|y^k\|$ ��С����ֵʱ�����浱ǰ�� $s^k, y^k$��������ȥ��
    % '�� pos��¼��ǰ�洢λ�ã�Ȼ�󸲸Ǹ�λ����ԭ������Ϣ'    
    ygk = g-gp;		s = x-xp;
    if ygk'*ygk>1e-20
        istore = istore + 1;
        pos = mod(istore, m); if pos == 0; pos = m; end
        YK(:,pos) = ygk;  SK(:,pos) = s;   rho(pos) = 1/(ygk'*s);
    %'�����ṩ��L-BFGS ˫ѭ���ݹ��㷨����ָ��˫ѭ����ѭ�������������еļ�¼����mʱ'��
    %'��ѭ��m�Σ�����ѭ���������ڵ�ǰ�ļ�¼������perm����˳���¼�洢λ��'
        if istore <= m; status = istore; perm = [perm, pos];
        else status = m; perm = [perm(2:m), perm(1)]; end
    end
end
%'��������ѭ�����˳�ʱ����¼���'
Out.iter = iter;
Out.nge = Out.nfe;
%'L-BFGS ˫ѭ���ݹ��㷨'
%'����˫ѭ���ݹ��㷨��������һ������������-Hg����ʼ��qΪ��ʼ������L-BFGS���㷨�У���һ��Ϊ���ݶȷ���'
   function y = LBFGS_Hg_Loop(dv)
        q = dv;   alpha = zeros(status,1);
%'��һ��ѭ����status��������status�ļ�����ϣ����������㹻��ʱΪm��'
%'perm����˳���¼�˴洢λ�ã�������ȡ��λ��k�ĸ�ʽΪ $\alpha_i=\rho_i(s^i)^\top q^{i+1}$ '
%'$q^{i}=q^{i+1}-\alpha_i y^i$������i��k-1��С��k-m'?        
        for di = status:-1:1
            k = perm(di);
            alpha(di) = (q'*SK(:,k)) * rho(k);
            q = q - alpha(di)*YK(:,k);
        end
% $r^{k-m}=\hat{H}^{k-m}q^{k-m}.$
        y = q/(rho(pos)* (ygk'*ygk));
%'�ڶ���ѭ����������ʽ $\beta_{i}=\rho_i(y^i)^\top r^i$'
%'$r^{i+1}=r^i+(\alpha_i-\beta_i)s^i$�������е�y��Ӧ�ڵ�����ʽ�е�r��������ѭ������ʱ���Է��ص�y��ֵΪ�½�����'?        
		for di = 1:status
            k = perm(di);
            beta = rho(k)* (y'* YK(:,k));
            y = y + SK(:,k)*(alpha(di)-beta);
        end
   end
end