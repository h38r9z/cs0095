java c
IEOR   E4525   Machine   Learning   for   OR   and   FE 
Due:      November   19   2020 
Final Exam 
1. Feedforward Networks (25 points) 
1.1.    (4   pts)   Suppose   I   have   a   neural   network   with   a   single   hidden   layer   with   weights W1 ∈   Rk×d, no bias terms, and ReLU activation   functions   (the   input   is   xi ∈ Rd   ).    Now suppose   I add a second hidden layer after the ﬁrst one, with no bias terms. Let’s say that   W(1)I,   W(2)I      are the weight   matrices   in this   new   network.    How   can   I   choose   W(1)I,   W(2)I   such   that   the   network   represents   the   same   function   as   the   one-hidden-layer   network?
1.2.    (4   pts)   How   many   parameters   are   there   in   a   fully   connected   feed   forward   network   with   l   hidden   layers,   width   k   ,   d   input   features,   and   we   use   a   linear   model   to   combine   the   output   at   the   last   hidden   layer   into   a   single   prediction?
1.3.    (9   pts)   In   class   we   saw   that   a   neural   network   can   learn   the   XOR   function.    Prove   that   a   feedforward   neural   network   with   a   single   layer   can   learn any Boolean function.       A   Boolean   function   is   a   function   f   :   {0, 1}n      →   {0, 1}.    Given   such   a   function   f,   construct   a   neural   network   with   a   single   layer   that   correctly   outputs   f.
1.4.    Consider   a   feedforward   neural   network with   linear   activation   functions:    σ(z)   = a ·   z +   b,   for   a, b   ∈ R,   with   a   ≠   0.
1.4.1.    (4 pts)   Consider   a network with   a   single   hidden   layer   with   weight   matrix   W   ∈ Rk×d and   ofsets   b   ∈ Rk   .    Derive   an   expression   that   shows   that   the   output   of   the   neural   network      is      linear      in      the      input    x    ∈   Rd.       This    expression should not include the   intermediate   variables   h or   z   in   the   hidden   layer.
1.4.2.    (4   pts)   Suppose   that   the   width   k   of the   single   hidden   layer   in   the   network   is   much   smaller thand, the number of   features.   Now consider some linear regression βTx+β0   on   the   original   features   x.      Can   this   linear   regression   be   expressed   using   this   neural   network?   If yes,   how?   If no,   why   not?
2. SGD (10 points) 
2.1.    Consider minibatch   SGD with a batch   size   of m.    In minibatch   SGD   we   normally   sample   without   replacement.    Suppose   we   run   minibatch   SGD with replacement.      Derive   the   mean   and   variance   of this   estimator.
3. Support Vector Machines (20 points) 
3.1.    (10   pts)   In   class   we   saw   that   a   deep   net   can   implement   the   XOR   function.    But   so   can   SVM!      Give   an      SVM   that   computes   the   XOR   function.         For   this   exercise,   you   should   assume   that      x      ∈   {-1, 1}   and   the      output      is      in      {-1, 1}.       Written   this   way,      the      XOR   dataset   is
([-1, -1], -1), ([1,   -1],   1), ([-1,   1],   1),   ([1,   1],   -1)3.2.    (10   pts)  代 写IEOR E4525 Machine Learning for OR and FE 2020 Final ExamPython
代做程序编程语言 In   class   we   saw   that   the   SVM   problem   for   the   separable   case   can   be   written   as
min   β0   ,βⅡβⅡ2(2)   s.t.   yi(β0   + βT   xi)   ≥ 1,   ∀i =   1, . . . ,   n
In   the   soft-margin   SVM   problem,   we   instead   solve   the   following   problem:
minβ0,βλ1 2kβk22 +nXi=1max(0, 1 − yi(β0 + βT xi))Either prove or give a complete counterexample for the following statement:    There   exists   a    single    value    λ    such    that   for    every    set    of   n    data    points    x1   , . . . ,   xn       that    are    separable,   hard   SVM   and   soft   SVM   return   the   same   solution   β,   β0
4. PCA and clustering (25 points) 
Suppose   that   we   have   a   clustering   problem   with   each   data   point      xi         ∈   Rd.       The   K-means   optimization   problem   is:
C1,...,CK 1 |Ck|X i,i0 ∈Ck kxi − xi0 k2                        (1)Suppose   we   perform   PCA   to   get   k   < d principal components. Let zi ∈ Rk be the represen- tation of xi in terms of the k principal components. We will compare clustering on xi and zi.
4.1.    (4   pts)   We   use   the   K-means   clustering   algorithm   covered   in   class,   on   the   original   data   points xi.   Give an example showing that the K-means   algorithm   may   converge   to   a   local   minimum   which   is   not   a   global   minimum   (hint:   give   a   one-dimensional   example).
4.2.    (4 pts) We use   the   K-means   clustering   algorithm   covered   in   class,   on   the   PCA   represen-   tation   zi,   with   K   > k.    Does   the   resulting   clustering   represent   a local minimum   of the K-means   clustering   optimization   problem   given   in   (1)?    Here,   you   may   take   local   mini-   mum to mean that the K-means algorithm would not make   any   changes   to   the   clustering   if allowed   to   run   starting   from   the   computed   clustering,   but   using   the   xi.
4.3.    (12   pts)   If your   answer   to   the   previous   question   was   yes,   argue   why.    If your   answer   was   no,   give   a   counterexample.
4.4.    (5   pts)   Suppose   that   the   data   matrix   X   ∈ Rn×d   ,   where   each   xi    is   a   row,   is   rank   r   =   k.   Does   this   change   your   answer   to   question 4.2.?   Why/why   not? 
5. Matrix Completion (20 points) 
5.1.    (10    pts)      Every      matrix      M    ∈   Rn×m    of   rank    exactly    r   can   be      factorized      into      matrices   B    ∈   Rn×r   ,   Y    ∈   Rr×m such    that    M      =    BY.       Under   the    assumption   that      B      must      be   orthonormal,   characterize   the   set   of solutions   B   and   Y   to   the   optimization   problem.
Hint:      the   solution   is   not   unique.
5.2.    (10    pts)    Consider    the    alternating      minimization      algorithm      for      the      matrix      completion   problem.   At   iteration   t,   we   saw   in   class   that   the   update   for   Y   is   as   follows
Yt = arg min Y X (i,j) (xij − yi>zjt−1)2 + λk Y k2F.
Derive   an   exact   expression   for   Yt.





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
