function [stp, f, g, options, work] = ls_csrch(stp,f, g , options , work)      
%     ****************************************
% 
%      Subroutine dcsrch
% 

%      This subroutine finds a step that satisfies a sufficient
%      decrease condition and a curvature condition.
%   此子例程查找满足充分减小条件和曲率条件的步长
%      Each call of the subroutine updates an interval with
%      endpoints stx and sty. The interval is initially chosen
%      so that it contains a minimizer of the modified function
%   子程序的每次调用都会更新一个具有端点stx和sty的间隔。初始选择区间，使其包含修改函数的极小值
%            psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).
% 
%      If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
%      interval is chosen so that it contains a minimizer of f.
%   选择区间，使其包含f的极小值
%      The algorithm is designed to find a step that satisfies
%      the sufficient decrease condition
%   该算法被设计为找到满足充分减少条件和曲率条件的步长
%            f(stp) <= f(0) + ftol*stp*f'(0),
% 
%      and the curvature condition 曲率条件
% 
%            abs(f'(stp)) <= gtol*abs(f'(0)).
% 
%      If ftol is less than gtol and if, for example, the function
%      is bounded below, then there is always a step which satisfies
%      both conditions.
%  如果ftol小于gtol，例如，如果函数在下有界，那么总是有一个步骤满足
%      If no step can be found that satisfies both conditions, then
%      the algorithm stops with a warning. In this case stp only
%      satisfies the sufficient decrease condition.
%  如果找不到满足这两个条件的步骤，则算法停止并发出警告。在这种情况下，stp仅满足充分减小条件。
%      A typical invocation of dcsrch has the following outline:
%  dcsrch的典型调用有以下概要：
%      Evaluate the function at stp = 0.0d0; store in f.  在stp=0.0d0时函数值；存储在f中。
%      Evaluate the gradient at stp = 0.0d0; store in g.  在stp=0.0d0时梯度值；存储在g中。
%      Choose a starting step stp. 选择开始步长stp。
% 
%      task = 'START'
%   10 continue
%         call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
%     +               isave,dsave)
%         if (task .eq. 'FG') then
%            Evaluate the function and the gradient at stp
%            go to 10
%            end if
% 
%      NOTE: The user must not alter work arrays between calls. 用户不得在调用之间更改工作数组
% 
%      The subroutine statement is
% 
%        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
%                          task,isave,dsave)
%      where
% 
%        stp is a double precision variable.     stp是一个双精度变量
%          On entry stp is the current estimate of a satisfactory   输入的stp是当前对满足条件的步长的估计
%             step. On initial entry, a positive initial estimate 在初始输入时，必须提供一个正的初始估计值。
%             must be provided.
%          On exit stp is the current estimate of a satisfactory step  退出的stp是当前对满足条件的步长的估计
%             if task = 'FG'. If task = 'CONV' then stp satisfies     stp满足充分的减小（task = 'FG'）
%             the sufficient decrease and curvature condition.         stp满足曲率条件（task = 'CONV）
% 
%        f is a double precision variable.       f是双精度变量。
%          On initial entry f is the value of the function at 0.  在初始条件f中，函数的值为f（0）。
%             On subsequent entries f is the value of the        在后续条件中，f是stp处函数的值。
%             function at stp.
%          On exit f is the value of the function at stp.     退出时f是stp处函数的值。
% 
%        g is a double precision variable.    g是双精度变量。
%          On initial entry g is the derivative of the function at 0.  初始输入时，g是函数在0处的导数
%             On subsequent entries g is the derivative of the       在后续条件中，g是stp处函数的导数。
%             function at stp.               
%          On exit g is the derivative of the function at stp.    退出时g是stp处函数的导数。
% 
%        ftol is a double precision variable.      ftol是一个双精度变量。
%          On entry ftol specifies a nonnegative tolerance for the
%             sufficient decrease condition.   输入时，ftol指定了充分减少条件的非负公差。
%          On exit ftol is unchanged.    退出时ftol不变。
% 
%        gtol is a double precision variable.    gtol是一个双精度变量。
%          On entry gtol specifies a nonnegative tolerance for the  
%             curvature condition.            输入时，gtol指定了充分减少条件的非负公差。
%          On exit gtol is unchanged.            退出时gtol不变
%     
%        xtol is a double precision variable.         xtol是一个双精度变量。
%          On entry xtol specifies a nonnegative relative tolerance
%             for an acceptable step. The subroutine exits with a
%             warning if the relative difference between sty and stx
%             is less than xtol.
%    输入xtol指定可接受步骤的非负相对公差。如果sty和stx之间的相对差值小于xtol，则子程序退出并显示警告。
%          On exit xtol is unchanged.     退出时xtol不变
% 
%        task is a character variable of length at least 60.   task是长度至少为60的字符变量
%          On initial entry task must be set to 'START'.        初始输入task必须设置为“START”
%          On exit task indicates the required action:      退出task指示所需的操作：
% 
%           1    START
%           2  If task(1:2) = 'FG' then evaluate the function and
%             derivative at stp and call dcsrch again.   在stp处计算函数和导数，然后再次调用dcsrch。
% 
%           0  If task(1:4) = 'CONV' then the search is successful.   搜索成功。
% 
%          -1  If task(1:4) = 'WARN' then the subroutine is not able
%             to satisfy the convergence conditions. The exit value of
%             stp contains the best point found during the search.
%    子程序不能满足收敛条件。stp处的退出值包含搜索期间找到的最佳点。
%          -5   If task(1:5) = 'ERROR' then there is an error in the
%             input arguments.   输入参数中存在错误
% 
%          On exit with convergence, a warning or an error, the
%             variable task contains additional information.  在退出时出现收敛、警告或错误，变量task包含信息。
% 
%        stpmin is a double precision variable.   stpmin是一个双精度变量。
%          On entry stpmin is a nonnegative lower bound for the step.  输入stpmin是步长的非负下限。
%          On exit stpmin is unchanged.  退出时stpmin是不变的
% 
%        stpmax is a double precision variable.     stpmax是一个双精度变量。
%          On entry stpmax is a nonnegative upper bound for the step.    输入stpmax是步长的非负上限。
%          On exit stpmax is unchanged.            退出时stpmax是不变的
% 
%        isave is an integer work array of dimension 2.   isave是一个维数为2的整数数组。
% 
%        dsave is a double precision work array of dimension 13.   dsave是维数为13的双精度数组。
% 
%      Subprograms called     调用的子程序
% 
%        MINPACK-2 ... dcstep
% 
%      MINPACK-1 Project. June 1983.
%      Argonne National Laboratory.
%      Jorge J. More' and David J. Thuente.
% 
%      MINPACK-2 Project. November 1993.
%      Argonne National Laboratory and University of Minnesota.
%      Brett M. Averick, Richard G. Carter, and Jorge J. More'.
% 
%     ****************************************
zero=0.0;     p5=0.5;    p66=0.66;
xtrapl=1.1;    xtrapu=4.0;
% c     Initialization block.  初始化 读入参数
if  work.task == 1
    % default options  默认参数
    if ~isfield(options,'maxiter')
        options.maxiter = 20;
    end
    if ~isfield(options,'display')
%         options.display = 'iter';
          options.display = 'no';
    end
    % options for wolfe condition   wolf准则的参数
    if ~isfield(options,'ftol')
        options.ftol = 1e-3;
    end
    if ~isfield(options,'gtol')
        options.gtol = 0.2;
    end
    if ~isfield(options,'xtol')
        options.xtol = 1e-30;
    end
    if ~isfield(options,'stpmin')
        options.stpmin = 1e-20;
%         options.stpmin = 1e-10;
    end
    if ~isfield(options,'stpmax')
        options.stpmax = 1e5;
%         options.stpmax = 2;
    end
    % c        Check the input arguments for errors.  检查输入参数是否有错误。
    if (stp < options.stpmin)
        work.task = -5;
        work.msg = 'ERROR: STP .LT. STPMIN';
    end
    if (stp > options.stpmax)
        work.task = -5;
        work.msg = 'ERROR: STP .GT. STPMAX';
    end
    if (g > zero)
        work.task = -5;
        work.msg = 'ERROR: INITIAL G .GE. ZERO';
    end
    if (options.ftol < zero)
        work.task = -5;
        work.msg =  'ERROR: FTOL .LT. ZERO';
    end
    if (options.gtol < zero)
        work.task = -5;
        work.msg = 'ERROR: GTOL .LT. ZERO';
    end
    if (options.xtol < zero)
        work.task = -5;
        work.msg =  'ERROR: XTOL .LT. ZERO';
    end
    if (options.stpmin < zero)
        work.task = -5;
        work.msg =  'ERROR: STPMIN .LT. ZERO';
    end
    if (options.stpmax < options.stpmin)
        work.task = -5;
        work.msg =  'ERROR: STPMAX .LT. STPMIN';
    end
    % c        Exit if there are errors on input.   如果输入有错误，则退出。
    if work.task == -5
        error(work.msg);
        return;
    end
    % c        Initialize local variables.  初始化局部变量
    % c        The variables stx, fx, gx contain the values of the step,   
    % c        function, and derivative at the best step.    变量stx、fx、gx包含最佳步长下的步长、函数和导数的值。
    % c        The variables sty, fy, gy contain the value of the step,
    % c        function, and derivative at sty.         变量sty、fy、gy包含sty处的步长、函数和导数的值。
    % c        The variables stp, f, g contain the values of the step,
    % c        function, and derivative at stp.   变量stp、f、g包含步长、函数和导数的值
    work.brackt    = false;
    work.stage     = 1;
    work.ginit     = g;
    work.gtest     = options.ftol * work.ginit;
    work.gx        = work.ginit;
    work.gy        = work.ginit;
    work.finit     = f;
    work.fx        = work.finit;
    work.fy        = work.finit;
    work.stx       = zero;
    work.sty       = zero;
    work.stmin     = zero;
%     work.stmin     = 1e-10;
    work.stmax     = stp + xtrapu*stp;
    work.width     = options.stpmax - options.stpmin;
    work.width1    = work.width/p5;
    work.bestfx    = f;
    work.bestgx    = g;
    work.beststp   = stp;
    work.iter = 0;
    work.task = 2;   work.msg = 'FG';
    return;
end
%------------------------------------------------------
% main loop  主循环

%  if  work.task == 2, the code come back with new f and g  代码返回新的f和g
if  work.task == 2 
    if work.bestfx > f
        work.bestfx    = f;
        work.bestgx    = g;
        work.beststp   = stp;
    end
end
% exceed max iterations  超过最大迭代次数
if (work.iter >= options.maxiter) 
     % currently, just set work.task = 0
     work.task = 0;
     work.msg =  'EXCEED MAX ITERATIONS';
     
     % set the stp to best in the history  将stp设置为历史最佳
     stp = work.beststp;
     f   = work.bestfx;
     g   = work.bestgx;
     return;
end
 work.iter = work.iter + 1;
 % c     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
 % c     algorithm enters the second stage.  算法进入第二阶段
 ftest = work.finit + stp*work.gtest;
 if (work.stage == 1) && (f <= ftest) && (g >= zero)
     work.stage = 2;
 end
 % c     Test for warnings.   警告测试
 if (work.brackt && ( stp < work.stmin || stp >= work.stmax ) )
     work.task = -1;
     work.msg = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS';
 end
 if (work.brackt && work.stmax-work.stmin <= options.xtol*work.stmax)
     work.task = -1;
     work.msg = 'WARNING: XTOL TEST SATISFIED';
 end
 if (stp >= options.stpmax && f <= ftest && g <= work.gtest)
     work.task = -1;
     work.msg =  'WARNING: STP = STPMAX';
 end
 if (stp <= options.stpmin && (f > ftest || g >= work.gtest))
     work.task = -1;
     work.msg =  'WARNING: STP = STPMIN';
 end
 %打印输出信息
 if strcmp(options.display, 'iter')
    fprintf('stpmin: %4.3e \t stpmax: %4.3e \t stage: %d, brackt: %d \n', ...
            work.stmin, work.stmax, work.stage, work.brackt);
 end
 % c     Test for convergence. 收敛测试
 if (f <= ftest && abs(g) <= options.gtol*(-work.ginit))
     work.task = 0;
     work.msg =  'CONVERGENCE';
 end
 % c     Test for termination. 迭代终止测试
 % if (task(1:4) .eq. 'WARN' .or. task(1:4) .eq. 'CONV') go to 10
 if (work.task == -1 || work.task == 0) 
     % set the best value in the history
%      stp = work.beststp;
%      f   = work.bestfx;
%      g   = work.bestgx;    
     return;
 end
 % c     A modified function is used to predict the step during the
 % c     first stage if a lower function value has been obtained but
 % c     the decrease is not sufficient.
 %如果已经获得了较低的函数值，但该减小不充分，则使用修改的函数来预测第一阶段期间的步长。
 if (work.stage == 1 && f <= work.fx && f >= ftest)
  % c    Define the modified function and derivative values.  定义修改的函数和导数值。
     fm  = f - stp*work.gtest;
     fxm = work.fx - work.stx*work.gtest;
     fym = work.fy - work.sty*work.gtest;
     gm  = g - work.gtest;
     gxm = work.gx - work.gtest;
     gym = work.gy - work.gtest;
     % c   Call dcstep to update stx, sty, and to compute the new step.  调用dcstep更新stx、sty并计算新步长。
     %     call dcstep(stx,fxm,gxm,sty,fym,gym,stp,fm,gm,brackt,stmin,work.stmax)
     [work.stx, fxm,gxm, work.sty, fym,gym, ...
         stp, fm,gm, work.brackt] = ...
         ls_dcstep(work.stx,fxm,gxm, work.sty,fym,gym, ...
         stp, fm,gm, work.brackt,work.stmin, work.stmax);
     % c     Reset the function and derivative values for f.  重置f的函数和导数值
     work.fx = fxm + work.stx*work.gtest;
     work.fy = fym + work.sty*work.gtest;
     work.gx = gxm + work.gtest;
     work.gy = gym + work.gtest;
 else
     % c   Call dcstep to update stx, sty, and to compute the new step.  调用dcstep更新stx、sty并计算新步长。
     %         dcstep(stx,fx,gx,sty,fy,gy,stp,f,g,brackt,stmin,stmax);
     [work.stx, work.fx,work.gx, work.sty,work.fy,work.gy,...
         stp,f,g,work.brackt] = ...
         ls_dcstep(work.stx, work.fx,work.gx, work.sty,work.fy,...
         work.gy,stp,f,g,work.brackt,work.stmin,work.stmax);
 end
 % c     Decide if a bisection step is needed.  确定步长是否需要的
 if (work.brackt)
     if (abs(work.sty-work.stx) >= p66*work.width1)
         stp = work.stx + p5*(work.sty-work.stx);
     end
     work.width1 = work.width;
     work.width = abs(work.sty-work.stx);
 end
 % c     Set the minimum and maximum steps allowed for stp.   设置stp允许的最小和最大步长
 if (work.brackt)
     work.stmin = min(work.stx, work.sty);
     work.stmax = max(work.stx, work.sty);
 else
     work.stmin = stp + xtrapl*(stp-work.stx);
     work.stmax = stp + xtrapu*(stp-work.stx);
 end
 % c     Force the step to be within the bounds stpmax and stpmin.  强制步长在stpmax和stpmin范围内
 stp = max(stp,options.stpmin);
 stp = min(stp,options.stpmax);
 % c     If further progress is not possible, let stp be the best 
 % c     point obtained during the search.     如果无法取得进一步进展，则stp作为搜索过程中获得的最佳点
 if (work.brackt && (stp <= work.stmin || stp >= work.stmax) || ...
         (work.brackt && work.stmax-work.stmin <= options.xtol*work.stmax))
     stp = work.stx;
 end
 % c     Obtain another function and derivative. 获得另一个函数值和导数
 work.task = 2;   work.msg = 'FG';