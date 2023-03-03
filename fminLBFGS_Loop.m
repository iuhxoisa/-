function [x, f, g, Out]= fminLBFGS_Loop(x, fun, opts, varargin)
if ~isfield(opts, 'xtol');      opts.xtol = 1e-6; end
if ~isfield(opts, 'gtol');      opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol');      opts.ftol = 1e-16; end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts, 'm');         opts.m  = 5; end%'L-BFGS的内存对存储数'
if ~isfield(opts, 'maxit');     opts.maxit  = 1000; end
if ~isfield(opts, 'storeitr');  opts.storeitr = 0; end%'标记是否记录每一次迭代的x'
if ~isfield(opts, 'record');    opts.record = 0; end%'标记是否需要迭代的信息'
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end%'每隔几步输出一次迭代信息'
%'参数复制'
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
%'初始化和迭代准备，计算初始点的信息，初始化迭代信息'?
[f,  g] = feval(fun, x , varargin{:});%'执行作为变量输入的fun函数'
nrmx = norm(x);
Out.f = f;  Out.nfe = 1; Out.nrmG = [];
%'storeitr在不为0时，记录每一步x'
if storeitr
    Out.xitr = x;
end
%'SK，YK用于储存L-BFGS算法中最近m步的x的变化（s）和梯度g的变化（y）'
n = length(x);
SK = zeros(n,m);
YK = zeros(n,m);
istore = 0; pos = 0;  status = 0;  perm = [];
%'设置打印格式'?
if record == 1
    if ispc; str1 = '  %10s'; str2 = '  %6s';
    else     str1 = '  %10s'; str2 = '  %6s'; end
    stra = ['%5s',str2,str2,str1, str2, str2,'\n'];
    str_head = sprintf(stra, ...
        'iter', 'stp', 'obj', 'diffx', 'nrmG', 'task');
    str_num = ['%4d  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2d\n'];
end
%'迭代主循环'
Out.msg = 'MaxIter';
for iter = 1:maxit
%'记录上一步迭代信息'?    
    xp = x;   nrmxp = nrmx;
    fp = f;   gp = g;
%'L-BFGS双循环方法找下降方向。第一次迭代采用负梯度方向，之后用L-BFGS方法来估计d=-Hg'    
    if istore == 0
        d = -g;
    else
        d = LBFGS_Hg_Loop(-g);
    end
%'沿L-BFGS方法得到的下降方向做线搜索，调用函数ls_csrch进行线搜索'    
    workls.task =1;% '首先初始化线搜索标记workls.task为1'
    deriv = d'*g;%'deriv为目标函数沿当前下降方向的方向导数'
    normd = norm(d);
    stp = 1;
    while 1
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
   % 'ls_csrch每次调用只执行线搜索的一步，并用workls.task指示下一步应当执行的操作'?     
        if (workls.task == 2)
            x = xp + stp*d;%'此处workls.task==2意味着要重新计算当前点函数值和梯度等'
            [f,  g] = feval(fun, x, varargin{:});
            Out.nfe = Out.nfe + 1;
            deriv = g'*d;
        else
            break%'直到满足线搜索条件时退出线搜索循环，得到更新之后的x'?
        end
    end
%'对于线搜索得到的步长$\alpha_k$，令 $s^k=x^{k+1}-x^k=\alpha_kd^k$ '
%'则$\|s^k\|_2=\alpha_k\|d^k\|_2$，计算$\|s^k\|_2/\max(1,\|x^k\|_2)$并将其作为判断收敛的标准'?    
    nrms = stp*normd;
    diffX = nrms/max(nrmxp,1);
% 更新 $\|x^{k+1}\|_2$, $\|g^{k+1}\|_2$，记录一步迭代信息?
    nrmG =  norm(g);
    Out.nrmg =  nrmG;
    Out.f = [Out.f; f];
    Out.nrmG = [Out.nrmG; nrmG];
    if storeitr
        Out.xitr = [Out.xitr, x];
    end
    nrmx = norm(x);
% '停机准则diffX表示相邻迭代点$x$ 的相对变化，nrmG表示当前 $x$ 处的梯度范数'%
%'$\displaystyle\frac{|f(x^{k+1})-f(x^k)|}{1+|f(x^k)|}$用以表示函数值的相对变化'
%'当前两者均小于阈值时，或者第三个小于阈值时，认为达到停机标准，退出当前循环'
    cstop = ((diffX < xtol) && (nrmG < gtol) )|| (abs(fp-f)/(abs(fp)+1)) < ftol;
%'当需要详细输出时，在开始迭代时,达到收敛时,达到最大迭代次数或退出迭代时,每若干步，打印详细结果'
if (record == 1) && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, ...
            iter, stp, f, diffX, nrmG, workls.task);
    end
%'当达到收敛条件时，停止迭代，记为达到收敛'?     
    if cstop
        Out.msg = 'Converge';
        break;
    end
    % '计算 $s^k=x^{k+1}-x^{k}$, $y^k=g^{k+1}-g^{k}$。当得到的 $\|y^k\|$ 不小于阈值时，保存当前的 $s^k, y^k$，否则略去。
    % '用 pos记录当前存储位置，然后覆盖该位置上原来的信息'    
    ygk = g-gp;		s = x-xp;
    if ygk'*ygk>1e-20
        istore = istore + 1;
        pos = mod(istore, m); if pos == 0; pos = m; end
        YK(:,pos) = ygk;  SK(:,pos) = s;   rho(pos) = 1/(ygk'*s);
    %'用于提供给L-BFGS 双循环递归算法，以指明双循环的循环次数，当已有的记录超过m时'，
    %'则循环m次，否则当循环次数等于当前的记录个数。perm按照顺序记录存储位置'
        if istore <= m; status = istore; perm = [perm, pos];
        else status = m; perm = [perm(2:m), perm(1)]; end
    end
end
%'当从上述循环中退出时，记录输出'
Out.iter = iter;
Out.nge = Out.nfe;
%'L-BFGS 双循环递归算法'
%'利用双循环递归算法，返回下一步的搜索方向-Hg，初始化q为初始方向，在L-BFGS主算法中，这一向为负梯度方向'
   function y = LBFGS_Hg_Loop(dv)
        q = dv;   alpha = zeros(status,1);
%'第一个循环，status步迭代（status的计算见上，当迭代步足够大时为m）'
%'perm按照顺序记录了存储位置，从中提取出位置k的格式为 $\alpha_i=\rho_i(s^i)^\top q^{i+1}$ '
%'$q^{i}=q^{i+1}-\alpha_i y^i$，其中i由k-1减小到k-m'?        
        for di = status:-1:1
            k = perm(di);
            alpha(di) = (q'*SK(:,k)) * rho(k);
            q = q - alpha(di)*YK(:,k);
        end
% $r^{k-m}=\hat{H}^{k-m}q^{k-m}.$
        y = q/(rho(pos)* (ygk'*ygk));
%'第二个循环，迭代格式 $\beta_{i}=\rho_i(y^i)^\top r^i$'
%'$r^{i+1}=r^i+(\alpha_i-\beta_i)s^i$，代码中的y对应于迭代格式中的r，当两次循环结束时，以返回的y的值为下降方向'?        
		for di = 1:status
            k = perm(di);
            beta = rho(k)* (y'* YK(:,k));
            y = y + SK(:,k)*(alpha(di)-beta);
        end
   end
end