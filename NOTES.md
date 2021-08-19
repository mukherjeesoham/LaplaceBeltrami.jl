# Regularize quantities at the poles for the 
# grad and divergence in the round sphere metric.
grad: dt_(s),  (1 / st) dp_(s)
div : (1 / st) ( dt(st At)  + dp (Ap) )

define: 
Ap_ = st * Ap 

then: 
div : (1 / st) (dt (st At) + dp (Ap_ / st))
      (1 / st) (dt (st At) + (1 /st) dp Ap_ + (1 / st) ct Ap_) 
      (1 / st^2) ( st^2 dt (st At) + dp Ap_ + ct Ap_) 
      (1 / st^2) ( st^2 ct At + st^3 dt (At)  + dp Ap_ + ct Ap_) 
Regular ---------^

(1/st) ( dt (st A^t) + dp (st (st / st) A^p) )
                       (-ct / st^2) [st^2 A^p]  + (1 / st) [dp (st^2 A^p)]

(1/st) ( dt (st A^t) + dp (st A^p) )

# A^t at the pole is zero.
# Coordinate basis
et_a : {1, 0}
ep_a : {0, 1}

et^a : {1, 0}
ep^a : {0, 1 / st^2}

# Dyad
qt_a : {1,  0}
qp_a : {0, st}

qt^a : {1, 0}
qp^a : {0, (1/st)}

A^p = A cdot ep
A^t = A cdot et

# v-----Regular
Atilde^p = A cdot qp = A cdot (ep * st)  = st A^p
Atilde^t = A cdot et
