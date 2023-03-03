function [x, out] = fminNewton(x, fun, hess, opts, varargin)
%'nargin表示函数输入的参数个数。不足3时报错，等于3时认为结构体opts为空结构体，即采用默认参数'
if (nargin < 3); error('[x, out] = fminNewton(x, fun, hess, opts)');end
if (nargin < 4); opts = []; end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts, 'rho1');      opts.rho1  = 1e-4; end
if ~isfield(opts, 'rho2');      opts.rho2  = 0.9; end
if ~isfield(opts,'maxit');    opts.maxit = 200;   end
if ~isfield(opts,'verbose');   opts.verbose = 0;    end%'verbose≥11时输出迭代信息'
if ~isfield(opts,'itPrint');    opts.itPrint = 1;   end%'隔几步输出一次迭代信息'
%'从结构体opts中复制参数'
maxit   = opts.maxit;   verbose  = opts.verbose;  itPrint = opts.itPrint;
xtol    = opts.xtol;    gtol    = opts.gtol;    ftol    = opts.ftol;
%'parsls为线搜索参数构成的结构体'
parsls.ftol = opts.rho1;  parsls.gtol = opts.rho2;
%'计算初始点x的函数值和梯度值'
[f,g] = fun(x);
nrmg = norm(g,2);
nrmx = norm(x,2);
%'输出结构体'
out = struct();
out.msg = 'MaxIter';%'out.msg是否达到收敛，MaxIter：在没有达到收敛，记为超出最大迭代次数退出'
out.nfe = 1;%'调用函数次数'
out.nrmG = nrmg;%'迭代退出时梯度范数'
%'当需要详细输出时，设定输出格式'
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
%'非精确牛顿法使用共轭梯度法求解牛顿方程，结构体opts_tCG为共轭梯度法提供参数'
opts_tCG = struct();
%'迭代主循环'
for iter = 1:maxit
    %'记录上一步迭代信息（迭代点，函数值，梯度，迭代点的范数）'
    fp = f;
    gp = g;
    xp = x;
    nrmxp = nrmx;
    %'调用截断共轭梯度法tCG(步精确地求解牛顿方程）得到牛顿方向d'
    opts_tCG.kappa = 0.1;
    opts_tCG.theta = 1;
    hess_tCG = @(d) hess(x,d);%'函数句柄'
    [d, ~, ~] = tCG(x, gp, hess_tCG, inf, opts_tCG);
    %'沿方向d做线搜索'
    workls.task = 1;%'初始化线搜索标workls.task记为1。（workls.task：指示下一步执行的操作）'
    deriv = d'*g;%'当前线搜索方向的导数'
    normd = norm(d);
    
    stp = 1;%'步长'
    while 1
        %ls_csrch每次调用只执行线搜索的一步
        [stp, f, deriv, parsls, workls] = ....
            ls_csrch(stp, f, deriv , parsls , workls);
        %'workls.task为2时需要重新计算当前点的函数值梯度'
        if (workls.task == 2)
            x = xp + stp*d;
            [f,  g] = feval(fun, x, varargin{:});
            out.nfe = out.nfe + 1;
            deriv = g'*d;
        else
            break
        end
    end
    nrms = stp*normd;%'nrms表示x(k+1)-x(k)的范数'
    xdiff = nrms/max(nrmxp,1);%'跌代点的相对变化量'
    nrmg = norm(g,2);%'x(k)处的梯度范数'
    out.nrmG = [out.nrmG; nrmg];
    nrmx = norm(x,2);%'x的范数'
    out.nfe = out.nfe + 1;%'调用函数次数'
    fdiff = abs(fp-f)/(abs(fp)+1);%'函数值相对变化量'
    %'cstop停机判断：梯度小于阈值或函数值相对变化量和x的相对变化量小于阈值时，迭代停止'
    cstop = nrmg <= gtol || (abs(fdiff) <= ftol && abs(xdiff) <= xtol);
    %'在达到收敛时，开始迭代时，到达最大迭代次数时，每若干步时，打印详细结果'
    if verbose>=1 && (cstop || iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if mod(iter,20*itPrint) == 0 && iter ~= maxit && ~cstop
            fprintf('\n%s', str_head);
        end    
        fprintf(str_num, ...
            iter, f, nrmg, fdiff, xdiff);
    end
    %'满足收敛时，标记为达到最优值退出'
    if cstop
        out.msg = 'Optimal';
        break;
    end
end
%'迭代退出时，记录输出信息'
out.iter = iter;
out.f    = f;
out.nrmg = nrmg;
end