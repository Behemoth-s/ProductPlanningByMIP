Set t 'month '        /1*8/;
Set i 'product object' /bike/;
Parameters q(i) 'set-up cost' /bike 5000 /;
Parameters p(i) 'cost per product' /bike 100/;
Table d(i, t) 'demand '
         1       2       3       4       5       6       7       8
bike   400     400     800     800    1200    1200    1200    1200;

Parameters s_init 'initial stock' /bike 200/;
Parameters h(i) 'holding cost per bike month' /bike 5/;

Positive Variable x(i, t);
Positive Variable s(i, t);
Alias (k, t);
Binary Variable y(i, t);
x.lo(i, t)=0;
s.lo(i, t)=0;
Variable cost ;
Equation acost
dem_sat(i, t)
dem_sat1(i)
vub(i, t);

acost .. cost =e=  sum((i, t), p(i) * x(i, t) + q(i) * y(i, t))
                 + sum((i, t)$( ord(t) <=7), h(i)*s(i, t))
                 + sum(i,h(i)/2*s(i,'8'));

dem_sat(i, t)$(ord(t) >= 2) .. s(i, t-1) + x(i, t) =e= d(i, t) + s(i, t);

dem_sat1(i) .. s_init(i) + x(i, '1') =e= d(i, '1') + s(i, '1');
vub(i, t) .. x(i, t) =l= sum(k$(ord(k)>=ord(t)), d(i, k))*y(i, t);


Model ex1 /all/;
Option MIP = CBC;
solve ex1 minimizing cost using mip;
