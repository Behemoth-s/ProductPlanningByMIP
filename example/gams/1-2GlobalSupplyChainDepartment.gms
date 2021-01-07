Set i 'number of products' /p1*p12/;
Set k 'number of machine' /m1*m3/;
Set t 'number of time periods' /t1*t15/;
alias (t, tt);


Parameters fam(i) 'subset of products'
 /       p1  2
         p2  2
         p3  2
         p4  2
         p5  2
         p6  2
         p7  3
         p8  3
         p9  3
         p10 3
         p11 3
         p12 3/;
Set mapi2fam(i, k);
mapi2fam(i, k) = fam(i)= ord(k);

Parameters ss(i) 'end period safety stock'
 /       p1  10.0
         p2  10.0
         p3  10.0
         p4  10.0
         p5  10.0
         p6  10.0
         p7  20.0
         p8  20.0
         p9  20.0
         p10 20.0
         p11 20.0
         p12 20.0 /;

Parameters ssinit(i) 'initial stock '
 /       p1  83.0
         p2  31.0
         p3  11.0
         p4  93.0
         p5  82.0
         p6  72.0
         p7  23.0
         p8  91.0
         p9  83.0
         p10 34.0
         p11 61.0
         p12 82.0 /;
Parameters alpha(i, k) 'constant production rate';
alpha(i, k) =1;
Parameters beta(i) 'Cleaning time after mixing '
 /       p1  30.0
         p2  20.0
         p3  30.0
         p4  40.0
         p5  40.0
         p6  10.0
         p7  30.0
         p8  20.0
         p9  10.0
         p10 50.0
         p11 30.0
         p12 20.0 /;

Table demand(i, t) 'demand forecast'
         t1    t2    t3    t4    t5    t6    t7    t8    t9    t10    t11    t12    t13    t14    t15
  p1      0    95   110    96    86   124    83   108   114    121    110    124    104     86     87
  p2     98    96    96    98   103   104   122   101    89    108    101    109    106    108     76
  p3    106     0    89   123    96   105    83    82   112    109    119     85     99     80    123
  p4     98   121     0   105    98    96   101    81   117     76    103     81     95    105    102
  p5      0   124   113   123   123    79   111    98    97     80     98    124     78    108    109
  p6    103   102     0    95   107   105   107   105    75     93    115    113    111    105     85
  p7    110    93     0   112    84   124    98   101    83     87    105    118    115    106     78
  p8     85    92   101   110    93    96   120   109   121     87     92     85     91     93    109
  p9    122   116   109     0   105   108    88    98    77     90    110    102    107     99     96
  p10   120   124    94   105    92    86   101   106    75    109     83     95     79    108    100
  p11   117    96    78     0   108    87   114   107   110     94    104    101    108    110     80
  p12   125   112    75     0   116   103   122    88    85     84     76    102     84     88     82;

Parameters L(k) 'machine capacity'
         / m1 1400
           m2 700
           m3 700/;

Positive Variables x(i, t) 'mixing batch size';
Binary Variables y(i, t) 'productiong setup';
Positive Variables s(i, t) 'end period inventory levle';

Equation
    dem_sat(i, t) 'demand satisfaction'
    dem_sat1(i) 'demand satisfaction 1'
    vub(i, t) 'set-up enforcement'
    safety_stock_limit(i, t) 'safety stock limit'
    mix_cap(t) 'mixing capacity restriction'
    pack_cap(k, t) 'packaging capacity restriction without cleaning times'
    obj_def 'objective definition';

Variable inventory 'total inventory, minimize objective';

dem_sat(i, t)$(ord(t)>=2) .. s(i, t-1) + x(i, t) =e= demand(i, t) + s(i, t);
dem_sat1(i) .. ssinit(i)  + x(i, 't1') =e= demand(i, 't1') + s(i, 't1');
safety_stock_limit(i, t) .. s(i, t) =g= ss(i);
vub(i, t) .. x(i, t) =l= (sum(tt$(ord(tt) >= ord(t)), demand(i, tt)) + ss(i)) * y(i, t) ;
mix_cap(t) .. sum(i, alpha(i, 'm1')*x(i, t)) + sum(i, beta(i) *y(i, t)) =l= L('m1');
pack_cap(k, t)$(ord(k)>=2) .. sum(mapi2fam(i,k), alpha(i, k)*x(i, t)) =l= L(k);
obj_def .. inventory =e= sum((i, t), s(i, t));

Model ex2 /all/;
Option MIP = SCIP;
Solve ex2 minimizing inventory using mip;
Display inventory.l, s.l;
Display x.l, y.l;

