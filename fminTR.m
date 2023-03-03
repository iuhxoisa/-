function [x, out] = fminTR(x,fun, hess, opts, varargin)
if (nargin < 3); error('[x, out] = fminTR(fun, x, opts)'); end
if (nargin < 4); opts = []; end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts,'eta1');     opts.eta1 = 1e-2;   end%'$\rho^k$ 的下界（当超出此界时意味着信赖域半径需要缩小并拒绝更新）'
if ~isfield(opts,'eta2');     opts.eta2 = 0.9;    end%'$\rho^k$ 的上界（当超出此界时意味着信赖域半径需要增大并接受更新）'
if ~isfield(opts,'gamma1');   opts.gamma1 = .25;  end%'每次调整信赖域半径缩小的比例'
if ~isfield(opts,'gamma2');   opts.gamma2 = 10;   end%'每次调整信赖域半径增大的比例'
if ~isfield(opts,'maxit');    opts.maxit = 200;   end
if ~isfield(opts,'record');   opts.record = 0;    end
if ~isfield(opts,'itPrint');  opts.itPrint = 1;   end
if ~isfield(opts,'verbose');  opts.verbose = 0;    end
% '从结构体中复制参数'
maxit   = opts.maxit;   record  = opts.record;  itPrint = opts.itPrint;
gtol    = opts.gtol;    ftol    = opts.ftol;
eta1    = opts.eta1;    eta2    = opts.eta2;
gamma1  = opts.gamma1;  gamma2  = opts.gamma2;
% '迭代准备,计算初始点x处的函数值和梯度'?
out = struct();
[f,g] = fun(x);
out.nfe = 1;
nrmg = norm(g,2);
out.nrmG = nrmg;
fp = f; gp = g;
% '信赖域子问题利用截断共轭梯度法求解，利用结构体opts_tCG为其提供参数'?
opts_tCG = struct();
% '初始化信赖域半径。初始设定为提供的参数值或默认值$\sqrt{\mathrm{len}(x)}/8$ '
% '并令 $\Delta$ 的上界$\bar{\Delta}$ 为$\sqrt{\mathrm{len}(x)}$ '
Delta_bar = sqrt(length(x));
Delta0 = Delta_bar/8;
if isfield(opts,'Delta')
    Delta = opts.Delta;
else
    Delta = Delta0;
end
% '两个参数分别记录信赖域半径连续增大或减小的次数，以方便初始值调整'
consecutive_TRplus = 0;
consecutive_TRminus = 0;
% '当需要详细输出时，设定输出格式'?
if record >= 1
    if ispc; str1 = '  %10s'; str2 = '  %8s';
    else     str1 = '  %10s'; str2 = '  %8s'; end
    stra = ['%5s', str1, str2, str2, str2, str2, str2, str2, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'F', 'fdiff', 'mdiff', 'redf', 'ratio', 'Delta', 'nrmg');
    fprintf('%s', str_head);
    str_num = ['%4d  %+8.7e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %2.1e  %2.1e'];
end
% '迭代主循环，以maxit为最大迭代次数'
for iter = 1:maxit
% '截断共轭梯度法中用于判断收敛的参数'?
    opts_tCG.kappa = 0.1;
    opts_tCG.theta = 1;
 % '调用tCG函数，利用截断共轭梯度法求解信赖域子问题，得到迭代方向d， stop_tCG表示截断共轭梯度法的退出原因'?
     hess_tCG = @(d) hess(x,d);
    [d, Hd, out_tCG] = tCG(x, gp, hess_tCG, Delta, opts_tCG);
    stop_tCG = out_tCG.stop_tCG;
%'计算比值 $$\displaystyle\rho_k=\frac{f(x^k)-f(x^k+d^k)}{m_k(0)-m_k(d^k)},$$'
%'以确定是否更新迭代和修正信赖域半径'
%'首先计算 $m_k(0)-m_k(d^k)=-\nabla f(x^k)^\top d^k + \frac{1}{2}(d^k)^\top B^k d^k$'
%'为了保证数值稳定性,增加一个小常数rreg'     
    mdiff = g'*d + .5*d'*Hd;
    rreg = 10*max(1, abs(fp)) * eps;
    mdiff = -mdiff + rreg;
    model_decreased = mdiff > 0;
%'构造实验点\hat{x}^{k+1}=x^k+d^k$，并计算该点处计算函数值和梯度'
    xnew = x + d;
    [f,g] = fun(xnew);
    nrmg = norm(g,2);
    out.nfe = out.nfe + 1;
%'计算 $f(x^k)-f(x^k+d^k)$，同样地增加一个小常数rreg，然后计算 $\rho^k$作为修正信赖域半径和判断是否更新的依据'
     redf = fp - f + rreg;
    ratio = redf/mdiff;
%' 当$\rho^k>\eta_1$ 时，接受此次更新，并记录上一步的函数值和梯度?' ?    
    if ratio >= eta1
        x = xnew;   fp = f;  gp = g;
        out.nrmG = [out.nrmG; nrmg];
end 
% '计算函数值相对变化'
    fdiff = abs(redf/(abs(fp)+1));
 % '停机准则：当满足梯度范数小于阈值或函数值的相对变化小于阈值且 $\rho^k>0$ 时，停止迭代'   
	 cstop = nrmg <= gtol || (abs(fdiff) <= ftol && ratio > 0);
  %'在达到收敛时，开始迭代时，到达最大迭代次数时，每若干步时，打印详细结果?
    if record>=1 && (cstop || ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if mod(iter,20*itPrint) == 0 && iter ~= maxit && ~cstop
            fprintf('\n%s', str_head);
        end
%'stop_tCG| 记录内层的截断共轭梯度法退出的原因，分别对应输出如下 '         
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
% '当满足停机准则时，记录达到最优值，退出循环?'
    if cstop
        out.msg = 'optimal';
        break;
    end
 % '信赖域半径的调整'
 % '如果 $\rho^k<\eta_1$ 或算法非降或当${m_k(0)-m_k(d^k)}\approx 0$约为0 时，不接受当前步迭代'
 %'这种情况下,对信赖域半径进行缩减具体而言，令 $\Delta \leftarrow \gamma_1\Delta$，'
 %'将信赖域半径的连续增大次数置为零，缩小次数加一 '   
 if ratio < eta1 || ~model_decreased || isnan(ratio)
        Delta = Delta*gamma1;
        consecutive_TRplus = 0;
        consecutive_TRminus = consecutive_TRminus + 1;
% '当信赖域半径连续 5 次减小时，认为当前的信赖域半径过大，并输出相应的提示信息'?   ?       
 if consecutive_TRminus >= 5 && opts.verbose >= 1
            consecutive_TRminus = -inf;
            fprintf(' +++ Detected many consecutive TR- (radius decreases).\n');
            fprintf(' +++ Consider decreasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
% '当$\rho_k>\eta_2$ 且$\|d_k\|=\Delta$（对应为截断共轭梯度法因遇到负曲率或者超出信赖域半径而终止）,增大信赖域半径'
%'$\Delta \leftarrow \min\{\gamma_2\Delta, \bar{\Delta}\}$,将信赖域半径的连续减小次数置零，增大次数加一'? ?   
 elseif ratio > eta2 && (stop_tCG == 1 || stop_tCG == 2)
        Delta = min(gamma2*Delta, Delta_bar);
        consecutive_TRminus = 0;
        consecutive_TRplus = consecutive_TRplus + 1;
% '考虑当信赖域半径连续 5 次增大时，认为当前的信赖域半径过小，输出相应的提示信息'?
        if consecutive_TRplus >= 5 && opts.verbose >= 1
            consecutive_TRplus = -inf;
            fprintf(' +++ Detected many consecutive TR+ (radius increases).\n');
            fprintf(' +++ Consider increasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', Delta_bar, Delta0);
        end
% '除了以上两种情况，不对信赖域半径进行调整，将其连续增大和减小次数都置零'?
    else
        consecutive_TRplus = 0;
        consecutive_TRminus = 0;
    end
end
% '当从外层迭代退出时，记录退出信息'??
out.iter = iter;
out.f    = f;
out.nrmg = nrmg;
end