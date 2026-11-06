// aegis_core.cpp - native timestep driver for AEGIS-MHD: replays CUDA graphs with the Python GIL released.
//
// The numerical kernels are not rewritten here. aegis/bridge.py captures the solver's own HLLD, MUSCL and CT kernels as
// CUDA graphs, once, through CuPy; this extension receives raw handles and device pointers (cupy_ndarray.data.ptr), so
// nothing is copied through Python, and loops over them in native code. So the conservation and CT properties are
// those of the same kernels.
//
//   run_graph      N replays of one graph on one stream
//   run_overlap    N steps of three-stage SSP-RK on a slab subdomain with communication overlapped by computation:
//                    stream 1: boundary-strip rates and update, then pack and stage the strips to the host;
//                    non-blocking MPI exchange (or a loopback on one rank, for periodic self-exchange);
//                    stream 2: interior rates and update, running while the exchange is in flight;
//                    rejoin, copy the received layers to the device, unpack them into the ghost cells.
//
// CUDA and MPI are loaded at run time from the libraries the Python process already uses, so the build needs neither a
// CUDA toolkit nor MPI headers. CUDA goes through the driver API (nvcuda / libcuda): CuPy may link its runtime statically,
// and runtime stream and graph handles are interchangeable with driver handles. The MPI handles
// assume the MPICH ABI (Intel MPI, MPICH: integer handles); Open MPI is refused by bridge.py. Exchange goes through host
// staging: no GPU-aware MPI or NCCL.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_t;
static lib_t find_lib(const char* name) { return GetModuleHandleA(name); }
static void* sym(lib_t l, const char* s) { return reinterpret_cast<void*>(GetProcAddress(l, s)); }
#else
#include <dlfcn.h>
typedef void* lib_t;
static lib_t find_lib(const char* name) { return dlopen(name, RTLD_NOW | RTLD_NOLOAD); }
static void* sym(lib_t l, const char* s) { return dlsym(l, s); }
#endif

namespace py = pybind11;
using clk = std::chrono::steady_clock;

// CUDA driver entry points (CUresult is an int; handles are pointers, device addresses are 64-bit)
typedef int (*graph_launch_t)(void* graph_exec, void* stream);
typedef int (*stream_sync_t)(void* stream);
typedef int (*d2h_t)(void* dst, std::uint64_t src, size_t n, void* stream);
typedef int (*h2d_t)(std::uint64_t dst, const void* src, size_t n, void* stream);
typedef int (*err_str_t)(int, const char**);
static graph_launch_t cuGraphLaunch_ = nullptr;
static stream_sync_t cuStreamSync = nullptr;
static d2h_t cuD2H = nullptr;
static h2d_t cuH2D = nullptr;
static err_str_t cuErr = nullptr;

// MPI entry points, MPICH ABI
typedef int (*isend_t)(const void*, int, int, int, int, int, int*);
typedef int (*irecv_t)(void*, int, int, int, int, int, int*);
typedef int (*waitall_t)(int, int*, void*);
static const int MPICH_DOUBLE = 0x4c00080b;
static isend_t mpiIsend = nullptr;
static irecv_t mpiIrecv = nullptr;
static waitall_t mpiWaitall = nullptr;

static void init_cuda(const std::string& dll) {
    if (cuGraphLaunch_) return;
    lib_t l = find_lib(dll.c_str());
    if (!l) throw std::runtime_error("CUDA driver " + dll + " is not loaded in this process");
    cuGraphLaunch_ = reinterpret_cast<graph_launch_t>(sym(l, "cuGraphLaunch"));
    cuStreamSync = reinterpret_cast<stream_sync_t>(sym(l, "cuStreamSynchronize"));
    cuD2H = reinterpret_cast<d2h_t>(sym(l, "cuMemcpyDtoHAsync_v2"));
    cuH2D = reinterpret_cast<h2d_t>(sym(l, "cuMemcpyHtoDAsync_v2"));
    cuErr = reinterpret_cast<err_str_t>(sym(l, "cuGetErrorString"));
    if (!cuGraphLaunch_ || !cuStreamSync || !cuD2H || !cuH2D)
        throw std::runtime_error("missing CUDA driver symbols in " + dll);
}

static void init_mpi(const std::string& dll) {
    if (mpiIsend) return;
    lib_t l = find_lib(dll.c_str());
    if (!l) throw std::runtime_error("MPI library " + dll + " is not loaded in this process");
    mpiIsend = reinterpret_cast<isend_t>(sym(l, "MPI_Isend"));
    mpiIrecv = reinterpret_cast<irecv_t>(sym(l, "MPI_Irecv"));
    mpiWaitall = reinterpret_cast<waitall_t>(sym(l, "MPI_Waitall"));
    if (!mpiIsend || !mpiIrecv || !mpiWaitall) throw std::runtime_error("missing MPI symbols in " + dll);
}

// errors found while the GIL is released are carried out as a string and raised afterwards
#define CK(call)                                                                  \
    do {                                                                          \
        int _e = (call);                                                          \
        if (_e) {                                                                 \
            const char* _m = "cuda error";                                        \
            if (cuErr) cuErr(_e, &_m);                                            \
            err = std::string(#call) + ": " + _m;                                 \
            return;                                                               \
        }                                                                         \
    } while (0)

static double run_graph(std::uintptr_t graph_exec, std::uintptr_t stream, long n, const std::string& cuda_dll) {
    init_cuda(cuda_dll);
    void* g = reinterpret_cast<void*>(graph_exec);
    void* s = reinterpret_cast<void*>(stream);
    std::string err;
    double seconds = 0.0;
    {
        py::gil_scoped_release nogil;
        [&]() {
            auto t0 = clk::now();
            for (long i = 0; i < n; ++i) {
                CK(cuGraphLaunch_(g, s));
                CK(cuStreamSync(s));  // one step at a time: the graph resets nothing between replays
            }
            seconds = std::chrono::duration<double>(clk::now() - t0).count();
        }();
    }
    if (!err.empty()) throw std::runtime_error(err);
    return seconds;
}

struct Overlap {
    std::vector<std::uintptr_t> boundary, interior, unpack;  // one graph per RK stage
    std::uintptr_t stream1 = 0, stream2 = 0;
    std::uintptr_t send_lo = 0, send_hi = 0, recv_lo = 0, recv_hi = 0;  // device buffers (.data.ptr)
    long n_values = 0;                                                  // doubles per exchange direction
    int comm = -1, left = -1, right = -1;                               // comm < 0: loopback on one rank
    bool overlap = true;
};

static py::dict run_overlap(const Overlap& o, long steps, const std::string& cuda_dll, const std::string& mpi_dll) {
    init_cuda(cuda_dll);
    if (o.comm >= 0) init_mpi(mpi_dll);
    const size_t nb = static_cast<size_t>(o.n_values) * sizeof(double);
    std::vector<double> h_slo(o.n_values), h_shi(o.n_values), h_rlo(o.n_values), h_rhi(o.n_values);
    void* s1 = reinterpret_cast<void*>(o.stream1);
    void* s2 = reinterpret_cast<void*>(o.stream2);
    auto dev = [](std::uintptr_t p) { return reinterpret_cast<void*>(p); };
    double t_boundary = 0, t_wait = 0, t_total = 0;
    std::string err;
    auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    {
        py::gil_scoped_release nogil;
        [&]() {
            auto t_all = clk::now();
            for (long n = 0; n < steps; ++n) {
                for (size_t k = 0; k < o.boundary.size(); ++k) {
                    auto t0 = clk::now();
                    // stream 1: boundary strips and their pack, then stage them to the host
                    CK(cuGraphLaunch_(dev(o.boundary[k]), s1));
                    CK(cuD2H(h_slo.data(), o.send_lo, nb, s1));
                    CK(cuD2H(h_shi.data(), o.send_hi, nb, s1));
                    CK(cuStreamSync(s1));
                    t_boundary += since(t0);
                    int req[4];
                    if (o.comm >= 0) {  // non-blocking exchange fired before the interior starts
                        int tag = static_cast<int>(100 + k);
                        mpiIrecv(h_rlo.data(), (int)o.n_values, MPICH_DOUBLE, o.left, tag, o.comm, &req[0]);
                        mpiIrecv(h_rhi.data(), (int)o.n_values, MPICH_DOUBLE, o.right, tag + 10, o.comm, &req[1]);
                        mpiIsend(h_shi.data(), (int)o.n_values, MPICH_DOUBLE, o.right, tag, o.comm, &req[2]);
                        mpiIsend(h_slo.data(), (int)o.n_values, MPICH_DOUBLE, o.left, tag + 10, o.comm, &req[3]);
                    } else {            // one rank, periodic: the strips are its own ghosts
                        h_rlo = h_shi;
                        h_rhi = h_slo;
                    }
                    t0 = clk::now();
                    if (o.overlap) {  // stream 2: interior, while the exchange is in flight
                        CK(cuGraphLaunch_(dev(o.interior[k]), s2));
                        if (o.comm >= 0) mpiWaitall(4, req, reinterpret_cast<void*>(1));  // MPI_STATUSES_IGNORE
                    } else {          // reference: wait for the exchange, then compute
                        if (o.comm >= 0) mpiWaitall(4, req, reinterpret_cast<void*>(1));
                        CK(cuGraphLaunch_(dev(o.interior[k]), s2));
                    }
                    t_wait += since(t0);
                    CK(cuStreamSync(s2));  // rejoin
                    CK(cuH2D(o.recv_lo, h_rlo.data(), nb, s1));
                    CK(cuH2D(o.recv_hi, h_rhi.data(), nb, s1));
                    CK(cuGraphLaunch_(dev(o.unpack[k]), s1));
                    CK(cuStreamSync(s1));
                }
            }
            t_total = since(t_all);
        }();
    }
    if (!err.empty()) throw std::runtime_error(err);
    py::dict out;
    out["seconds"] = t_total;
    out["boundary_seconds"] = t_boundary;
    out["comm_wait_seconds"] = t_wait;
    return out;
}

PYBIND11_MODULE(aegis_core, m) {
    m.doc() = "native GIL-free timestep driver: replays CUDA graphs captured by aegis/bridge.py";
    m.def("run_graph", &run_graph, py::arg("graph_exec"), py::arg("stream"), py::arg("n"), py::arg("cuda_dll"));
    py::class_<Overlap>(m, "Overlap")
        .def(py::init<>())
        .def_readwrite("boundary", &Overlap::boundary)
        .def_readwrite("interior", &Overlap::interior)
        .def_readwrite("unpack", &Overlap::unpack)
        .def_readwrite("stream1", &Overlap::stream1)
        .def_readwrite("stream2", &Overlap::stream2)
        .def_readwrite("send_lo", &Overlap::send_lo)
        .def_readwrite("send_hi", &Overlap::send_hi)
        .def_readwrite("recv_lo", &Overlap::recv_lo)
        .def_readwrite("recv_hi", &Overlap::recv_hi)
        .def_readwrite("n_values", &Overlap::n_values)
        .def_readwrite("comm", &Overlap::comm)
        .def_readwrite("left", &Overlap::left)
        .def_readwrite("right", &Overlap::right)
        .def_readwrite("overlap", &Overlap::overlap);
    m.def("run_overlap", &run_overlap, py::arg("config"), py::arg("steps"), py::arg("cuda_dll"), py::arg("mpi_dll") = "");
}
