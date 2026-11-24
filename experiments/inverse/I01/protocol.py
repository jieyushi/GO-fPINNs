"Define stage grids, parameter settings, observation counts, and learning rates."
TRUE={'alpha':.9,'beta':1.,'gamma':1.}
XL,XR,T=0.,1.,1.
METHODS=('fl1','go_single','go_multi')
RUN_ROOT='runs/I01_gradient_v2'
GRADIENT_POLICY='I01_local_GE_xt_default_v2'
LABELS={'fl1':'FL1-fPINN','go_single':'GO-fPINNs (single-stage)','go_multi':'GO-fPINNs (multi-stage)'}
CASES={
    'alpha_0p1':dict(unknown='alpha',initial=[.1]),
    'alpha_0p3':dict(unknown='alpha',initial=[.3]),
    'alpha_0p5':dict(unknown='alpha',initial=[.5]),
    'beta_0p1':dict(unknown='beta',initial=[.1]),
    'beta_0p5':dict(unknown='beta',initial=[.5]),
    'beta_2p0':dict(unknown='beta',initial=[2.]),
    'gamma_0p1':dict(unknown='gamma',initial=[.1]),
    'gamma_0p5':dict(unknown='gamma',initial=[.5]),
    'gamma_2p0':dict(unknown='gamma',initial=[2.]),
    'joint_0p5':dict(unknown='joint',initial=[.5,.5,.5]),
}
FIGURE_CASES=('alpha_0p1','alpha_0p5','beta_0p5','beta_2p0','gamma_0p5','gamma_2p0')


def plan(method,smoke=False):
    if method not in METHODS:raise ValueError('Unknown method.')
    sizes=[10,20,30,40,50] if method=='go_multi' else [50]
    if smoke:sizes=[3,4,5,6,7] if method=='go_multi' else [7]
    return [dict(stage=sid,M=n,N=n,observations=n,
                 lr=1e-3/2**sid if method=='go_multi' else 5e-4,
                 scheme='fast_l1' if method=='fl1' else 'fast_l21') for sid,n in enumerate(sizes)]


def unknown_names(unknown):
    return tuple(TRUE) if unknown=='joint' else (unknown,)


def gradient_axes(unknown,mode='script'):
    """Coordinate order is (t,x); the default includes both for every task.

    ``script`` remains a CLI-compatible name, now equivalent to ``xt``.
    The historical alpha spatial-only comparison is explicit ``x``.
    """
    if mode=='script':return (0,1)
    return {'x':(1,),'xt':(0,1),'none':()}[mode]


def gradient_contract(unknown,method,mode='script',w_grad=.001):
    """Record actual GE axes and the local manufactured-derivative contract."""
    selected=gradient_axes(unknown,mode)
    active=selected if method!='fl1' and w_grad else ()
    return dict(policy=GRADIENT_POLICY,requested_mode=mode,coordinate_order=['t','x'],
                selected_axes=[['t','x'][axis] for axis in selected],
                effective_axes=[['t','x'][axis] for axis in active],enabled=bool(active),
                definition='sum over selected axes and real/imaginary components of mean((d(local)/d(axis)-d(f_local_TRUE)/d(axis))^2)',
                local='beta*u_xx + gamma*|u|^2*u',
                target='derivatives of the fixed TRUE local manufactured source, at current sigma coordinates',
                full_caputo_residual_gradient=False,w0=w_grad,
                effective_weight='w0*exp(-3*loss_f.detach()) for GO; zero for FL1')
