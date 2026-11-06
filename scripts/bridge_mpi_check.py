"""Check of the native overlap stepper across MPI ranks (aegis/bridge.py, src/aegis_core.cpp).

    mpiexec -n 2 python scripts/bridge_mpi_check.py

An Orszag-Tang state on an x-decomposed periodic domain runs N steps through OverlapStepper, with the exchange done by
MPI inside native code and the GIL released, and the gathered result is compared with the serial MHDSolver at the same
dt. Prints one JSON line on rank 0. Ranks sharing one GPU show correctness only, not scaling.
"""
import os
import sys
import json
import contextlib
import io
import dataclasses

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mpi4py import MPI
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, EN, iRHO, iPR, iVX, iVY
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
    from aegis.parallel import local_config
    from aegis import bridge


def setup(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros_like(s.U)
    W[iRHO] = 25 / (36 * np.pi)
    W[iPR] = 5 / (12 * np.pi)
    W[iVX] = -xp.sin(2 * np.pi * s.Y)
    W[iVY] = xp.sin(2 * np.pi * s.X)
    s.U[...] = prim_to_cons(W, cfg.gamma)
    B0 = 1 / np.sqrt(4 * np.pi)
    s.set_vector_potential(lambda x, y: B0 * (np.cos(4 * np.pi * x) / (4 * np.pi) + np.cos(2 * np.pi * y) / (2 * np.pi)))
    return s


def main(n_steps=30, nx=64, ny=48):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    cfg = Config(nx=nx, ny=ny, x_max=1.0, y_max=1.0, cfl=0.3, gamma=5 / 3, bc_x_type="periodic", bc_y_type="periodic",
                 diag_interval=10**9, snapshot_times=[], use_cuda_graphs=False)
    lc, off = local_config(cfg, rank, size)
    s = setup(lc)
    s._ct_sync()
    ng = s.ng
    I = (slice(None), slice(ng, ng + lc.nx), slice(ng, ng + lc.ny))
    dt = comm.allreduce(0.5 * s.compute_dt(cons_to_prim(s.U, s.eos)), op=MPI.MIN)
    # Az gives valid faces everywhere, including the x ghosts; the cell field and y ghosts follow
    E0 = comm.allreduce(float(xp.sum(s.U[(EN,) + I[1:]])), op=MPI.SUM)
    st = bridge.OverlapStepper(s, comm)
    timing = st.advance(n_steps, dt)
    Un = to_numpy(s.U[I])
    parts = comm.gather(Un, root=0)
    E1 = comm.allreduce(float(xp.sum(s.U[(EN,) + I[1:]])), op=MPI.SUM)
    from aegis import ct
    d = float(xp.max(xp.abs(ct.div_faces(s.Bf, (lc.dx, lc.dy), ng, (lc.nx, lc.ny)))))
    divb = comm.allreduce(d, op=MPI.MAX)
    shared = None
    if size > 1:                                # the shared x-face must be identical on both sides of a cut
        lo = to_numpy(s.Bf[0][ng, ng:ng + lc.ny])
        hi = to_numpy(s.Bf[0][ng + lc.nx, ng:ng + lc.ny])
        got = comm.sendrecv(hi, dest=(rank + 1) % size, source=(rank - 1) % size)
        shared = comm.allreduce(float(np.max(np.abs(got - lo))), op=MPI.MAX)
    if rank == 0:
        U = np.concatenate(parts, axis=1)
        a = setup(cfg)
        for _ in range(n_steps):
            a.dt = dt
            a.step_ssprk3()
        Ua = to_numpy(a.U[(slice(None), slice(ng, ng + nx), slice(ng, ng + ny))])
        err = float(np.max(np.abs(U - Ua) / (np.abs(Ua).max(axis=(1, 2), keepdims=True) + 1e-300)))
        print("RESULT " + json.dumps(dict(ranks=size, steps=n_steps, parity=err, energy=abs(E1 - E0) / abs(E0), div_b=divb,
                                          shared_face=shared, timing=timing)), flush=True)


if __name__ == "__main__":
    main()
