# bridge.py - run N solver steps in native code, GIL released, from CUDA graphs of the solver's own kernels.
#
# The Python solver stays the description of the scheme. Its step (MHDSolver._rk_core, the same HLLD / MUSCL / CT kernels)
# is captured once as CUDA graphs; src/aegis_core.cpp then replays them from raw handles. Nothing in the numerics changes,
# so conservation and div B are those of the solver; tools/build_core.py builds the extension.
#
#   NativeStepper(solver).advance(n, dt)    n steps of a fixed dt through one graph.
#   OverlapStepper(solver, comm).advance(n, dt)   n steps on a slab subdomain (or one periodic rank), with each RK stage
#       split so that communication overlaps computation:
#         stream 1: rates and update of the strips next to the x cuts, pack their layers, stage them to the host;
#         non-blocking MPI of the layers (loopback on one rank);
#         stream 2: rates and update of the interior, which reads no ghost cell, while the exchange is in flight;
#         rejoin, copy the received layers to the device, unpack them into the ghost layers of the next stage state.
#
# Scope: ideal gas, ideal MHD, constrained transport, Cartesian, fixed dt during a call (the caller chooses dt, e.g. the
# global CFL value at the start of a batch; check it against the CFL limit between batches). OverlapStepper needs periodic
# y and periodic (or MPI-cut) x. Split-step physics is refused: it is not in the graphs. The shared x-face field on a cut
# is taken from the right rank, so both ranks hold the same value. Exchange is host-staged MPI (MPICH ABI: Intel MPI,
# MPICH); NCCL and GPU-aware MPI are not used, and nothing here has run on more than one GPU or node.
import os

import numpy as np
import cupy as cp

from . import fused, physics
from .config import NVAR, RHO, MX, BX, BY, EN, RHOC, FLOOR_RHO, iBX, iBY
from .physics import xp, cons_to_prim, muscl_x, muscl_y, hlld_flux_x, hlld_flux_y

try:
    from . import aegis_core as _core
except ImportError:
    _core = None

_CUDA_DLL = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"


def available():
    return _core is not None


def _require(cfg, solver):
    if _core is None:
        raise ImportError("aegis_core is not built: pip install pybind11 ziglang; python tools/build_core.py")
    extra = [k for k in ("rad_coeff", "cond_kappa0", "cond_kappa_i0", "visc_coeff", "nernst_coeff", "hall_di", "gyro_di")
             if getattr(cfg, k, 0)]
    if (not solver.ct or solver.cyl or not isinstance(solver.eos, (int, float)) or cfg.is_resistive() or cfg.fld
            or cfg.two_temp or solver.strength or cfg.pic or solver.walls is not None or extra):
        raise ValueError("the native bridge covers ideal-gas ideal MHD with div_b='ct', Cartesian, no split-step physics")


class NativeStepper:
    """whole-step graph (RK stages and the boundary refresh) replayed natively"""

    def __init__(self, solver):
        s = solver
        _require(s.cfg, s)
        self.s = s
        s._ct_sync()
        self._dtd = cp.zeros(())
        self._pool = cp.cuda.MemoryPool()
        self._stream = cp.cuda.Stream(non_blocking=True)
        # one ordinary step on saved state, so that every kernel and buffer exists before the capture
        saved = s.U.copy(), [f.copy() for f in s.Bf], s._acc_boundary_energy.copy()
        s._glm_ch_frozen = 0.0
        s._rk_core(1e-12)
        s.U[...] = saved[0]
        for f, g in zip(s.Bf, saved[1]):
            f[...] = g
        s._acc_boundary_energy = saved[2]
        cp.cuda.Device().synchronize()
        with cp.cuda.using_allocator(self._pool.malloc), self._stream:
            self._stream.begin_capture()
            s._rk_core(self._dtd)
            s.apply_bc(s.U)
            s._ghost_faces(s.Bf, s.U)
            self.graph = self._stream.end_capture()
        self.graph.upload(self._stream)

    def advance(self, n, dt):
        """n steps of dt, without the GIL; returns the seconds spent in native code"""
        self._dtd.fill(dt)
        cp.cuda.Device().synchronize()
        sec = _core.run_graph(self.graph.graphExec, self._stream.ptr, int(n), _CUDA_DLL)
        self.s.t += n * dt
        self.s.step += n
        return sec

    def advance_python(self, n, dt):
        """the same replay driven from Python, for comparison"""
        self._dtd.fill(dt)
        cp.cuda.Device().synchronize()
        for _ in range(n):
            self.graph.launch(stream=self._stream)
            self._stream.synchronize()
        self.s.t += n * dt
        self.s.step += n


class OverlapStepper:
    """see the module notes. solver: a slab of the x-periodic global problem (or the whole of it on one rank), its U and
    faces set including their ghost layers. comm: mpi4py communicator (None: one rank, periodic loopback)."""

    def __init__(self, solver, comm=None, strip=3, overlap=True, mpi_dll=None):
        s = solver
        _require(s.cfg, s)
        cfg = s.cfg
        if strip < 2 or cfg.nx < 2 * strip + 1:
            raise ValueError("OverlapStepper needs strips of at least 2 cells and interior cells between them")
        if cfg.bc_y_type != "periodic":
            raise ValueError("OverlapStepper needs periodic y")
        self.s, self.w, self.ng, self.nx, self.ny = s, strip, s.ng, cfg.nx, cfg.ny
        self.N, self.Ny = s.nx_tot, s.ny_tot
        s._ct_sync()
        self.g = float(cfg.gamma)
        # stage states; S[0] is the solver's U. Stage k reads S[k] and S[0] and writes S[k + 1]; S[3] goes back into S[0]
        # at the end of the step (a face on a region edge is written by both regions, so the last stage cannot write
        # into the state the other region still reads)
        self.S = [s.U] + [xp.empty_like(s.U) for _ in range(3)]
        self.F = [list(s.Bf)] + [[f.copy() for f in s.Bf] for _ in range(3)]
        self._dtd = cp.zeros(())
        ng, Ny = self.ng, self.Ny
        self.n_values = NVAR * ng * Ny + ng * Ny + ng * (Ny + 1)
        self.send = [xp.zeros(self.n_values), xp.zeros(self.n_values)]
        self.recv = [xp.zeros(self.n_values), xp.zeros(self.n_values)]
        self.s1, self.s2 = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
        self.pool1, self.pool2 = cp.cuda.MemoryPool(), cp.cuda.MemoryPool()
        self._warm()
        self.gb, self.gi, self.gu = [], [], []
        for k in range(3):
            self.gb.append(self._capture(self.s1, self.pool1, lambda k=k: self._boundary(k)))
            self.gi.append(self._capture(self.s2, self.pool2, lambda k=k: self._interior(k)))
            self.gu.append(self._capture(self.s1, self.pool1, lambda k=k: self._unpack(k)))
        o = _core.Overlap()
        o.boundary = [g.graphExec for g in self.gb]
        o.interior = [g.graphExec for g in self.gi]
        o.unpack = [g.graphExec for g in self.gu]
        o.stream1, o.stream2 = self.s1.ptr, self.s2.ptr
        o.send_lo, o.send_hi = self.send[0].data.ptr, self.send[1].data.ptr
        o.recv_lo, o.recv_hi = self.recv[0].data.ptr, self.recv[1].data.ptr
        o.n_values, o.overlap = self.n_values, overlap
        self.mpi_dll = ""
        if comm is not None and comm.Get_size() > 1:
            from mpi4py import MPI
            if not any(v in MPI.Get_library_version() for v in ("Intel", "MPICH")):
                raise RuntimeError("the native exchange assumes the MPICH ABI (Intel MPI, MPICH)")
            r, p = comm.Get_rank(), comm.Get_size()
            o.comm, o.left, o.right = comm.py2f(), (r - 1) % p, (r + 1) % p
            self.mpi_dll = mpi_dll or ("impi.dll" if os.name == "nt" else "libmpi.so.12")
        self.o = o

    def _capture(self, stream, pool, fn):
        with cp.cuda.using_allocator(pool.malloc), stream:
            stream.begin_capture()
            fn()
            g = stream.end_capture()
        g.upload(stream)
        return g

    def _warm(self):
        """every kernel once outside a capture (compiled and registered), on saved state"""
        saved = [a.copy() for a in self.S], [[f.copy() for f in F] for F in self.F]
        d = self._dtd
        self._dtd = cp.zeros(())
        for k in range(3):
            self._boundary(k)
            self._interior(k)
            self._unpack(k)
        cp.cuda.Device().synchronize()
        self._dtd = d
        for a, b in zip(self.S, saved[0]):
            a[...] = b
        for F, G in zip(self.F, saved[1]):
            for f, h in zip(F, G):
                f[...] = h

    # ---------------- one region ----------------
    def _rates(self, Uk, Fk, a, b):
        """cell rates (a..b-1), x-face rates (faces a..b) and y-face rates (cells a..b-1): the block a-2..b+1 is a small
        solver domain with 2 ghost cells each side, so this is the arithmetic of MHDSolver.compute_rhs on it"""
        ng, ny, dx, dy = self.ng, self.ny, self.s.cfg.dx, self.s.cfg.dy
        W = cons_to_prim(xp.ascontiguousarray(Uk[:, a - 2:b + 2, :]), self.g)
        WLx, WRx = muscl_x(W, self.g)
        sh = WLx.shape
        WLx[iBX] = WRx[iBX] = Fk[0][a:b + 1, :]
        Fx = hlld_flux_x(xp.ascontiguousarray(WLx.reshape(NVAR, -1)), xp.ascontiguousarray(WRx.reshape(NVAR, -1)),
                         self.g, 0.0)[0].reshape(sh)
        WLy, WRy = muscl_y(W, self.g)
        sh2 = WLy.shape
        WLy[iBY] = WRy[iBY] = Fk[1][a - 2:b + 2, 2:self.Ny - 1]
        Fy = hlld_flux_y(xp.ascontiguousarray(WLy.reshape(NVAR, -1)), xp.ascontiguousarray(WRy.reshape(NVAR, -1)),
                         self.g, 0.0)[0].reshape(sh2)
        m = b - a
        dFx = Fx[:, 1:1 + m, ng:ng + ny] - Fx[:, 0:m, ng:ng + ny]
        dFy = Fy[:, ng:ng + m, 1:1 + ny] - Fy[:, ng:ng + m, 0:ny]
        R = (0.0 - dFx / dx) - dFy / dy
        R[BX] = 0.0
        R[BY] = 0.0
        E = fused.emf2d(Fx, Fy, W, m, ny, dx, dy)
        dB0 = -(E[:, 1:] - E[:, :-1]) / dy                               # x-faces a..b
        dB1 = (E[1:, :] - E[:-1, :]) / dx                                # y-faces of cells a..b-1
        return R, dB0, dB1

    def _advance_region(self, k, a, b):
        """write the next stage state on cells a..b-1, faces a..b, with y wrap and the cell field from the faces"""
        ng, ny, dt = self.ng, self.ny, self._dtd
        Uk, Fk = self.S[k], self.F[k]
        Un, Fn = self.S[k + 1], self.F[k + 1]
        U0, F0 = self.S[0], self.F[0]
        R, dB0, dB1 = self._rates(Uk, Fk, a, b)
        r = (slice(None), slice(a, b), slice(ng, ng + ny))
        A = Uk[r] + dt * R
        if k == 0:
            new = A
        elif k == 1:
            new = 0.75 * U0[r] + 0.25 * A
        else:
            new = U0[r] * (1.0 / 3.0) + A * (2.0 / 3.0)
        rho = xp.maximum(new[RHO], FLOOR_RHO)
        new[RHOC] = rho * xp.clip(new[RHOC] / rho, 0.0, 1.0)
        fx, fy = (slice(a, b + 1), slice(ng, ng + ny)), (slice(a, b), slice(ng, ng + ny + 1))
        c = ((1.0, 0.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0))[k]
        for q, (sl, d) in enumerate(((fx, dB0), (fy, dB1))):
            A_ = Fk[q][sl] + dt * d
            Fn[q][sl] = A_ if k == 0 else c[0] * F0[q][sl] + c[1] * A_
        new[BX] = 0.5 * (Fn[0][a:b, ng:ng + ny] + Fn[0][a + 1:b + 1, ng:ng + ny])
        new[BY] = 0.5 * (Fn[1][a:b, ng:ng + ny] + Fn[1][a:b, ng + 1:ng + ny + 1])
        Un[r] = new
        # periodic y on this region (ct.ghost_faces and apply_bc)
        Un[:, a:b, :ng] = Un[:, a:b, ny:ny + ng]
        Un[:, a:b, ng + ny:] = Un[:, a:b, ng:2 * ng]
        Fn[0][a:b + 1, :ng] = Fn[0][a:b + 1, ny:ny + ng]
        Fn[0][a:b + 1, ng + ny:] = Fn[0][a:b + 1, ng:2 * ng]
        Fn[1][a:b, :ng] = Fn[1][a:b, ny:ny + ng]
        Fn[1][a:b, ng + ny + 1:] = Fn[1][a:b, ng + 1:2 * ng + 1]

    def _pack(self, k, lo, buf):
        Un, Fn = self.S[k + 1], self.F[k + 1]
        ng = self.ng
        parts = [Un[:, lo:lo + ng, :].ravel(), Fn[0][lo:lo + ng, :].ravel(), Fn[1][lo:lo + ng, :].ravel()]
        buf[...] = xp.concatenate(parts)

    def _boundary(self, k):
        ng, nx, w = self.ng, self.nx, self.w
        self._advance_region(k, ng, ng + w)
        self._advance_region(k, ng + nx - w, ng + nx)
        self._pack(k, ng, self.send[0])                 # first layers: the left neighbour's upper ghosts
        self._pack(k, ng + nx - ng, self.send[1])       # last layers: the right neighbour's lower ghosts

    def _interior(self, k):
        ng, nx, w = self.ng, self.nx, self.w
        self._advance_region(k, ng + w, ng + nx - w)

    def _unpack(self, k):
        Un, Fn = self.S[k + 1], self.F[k + 1]
        ng, nx, Ny = self.ng, self.nx, self.Ny
        for buf, lo in ((self.recv[0], 0), (self.recv[1], ng + nx)):
            n1 = NVAR * ng * Ny
            n2 = n1 + ng * Ny
            Un[:, lo:lo + ng, :] = buf[:n1].reshape(NVAR, ng, Ny)
            Fn[0][lo:lo + ng, :] = buf[n1:n2].reshape(ng, Ny)
            Fn[1][lo:lo + ng, :] = buf[n2:].reshape(ng, Ny + 1)
        if k == 2:                                       # the step is complete
            self.S[0][...] = Un
            for f0, fn in zip(self.F[0], Fn):
                f0[...] = fn

    def advance(self, n, dt):
        """n steps of dt on this rank (all ranks call it together); returns the native timing breakdown"""
        self._dtd.fill(dt)
        cp.cuda.Device().synchronize()
        out = _core.run_overlap(self.o, int(n), _CUDA_DLL, self.mpi_dll)
        self.s.t += n * dt
        self.s.step += n
        return out
