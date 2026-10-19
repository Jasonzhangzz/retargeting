"""Local acados discovery and locked code-generation cache."""
import ctypes
import os
import sys
from pathlib import Path

def acados_root_candidates(repo_root=None):
    root=Path(__file__).resolve().parents[2]
    return [str(p) for p in (Path(os.environ['ACADOS_SOURCE_DIR']) if os.environ.get('ACADOS_SOURCE_DIR') else root/'third_party/acados',
        root/'third_party/acados',Path.home()/'acados',Path.home()/'zz_ws/acados')]


def _is_acados_tree(acados_root):
    interface = os.path.join(
        acados_root, "interfaces", "acados_template", "acados_template", "__init__.py"
    )
    if not os.path.isfile(interface):
        return False
    lib_dir = os.path.join(acados_root, "lib")
    has_lib = any(
        os.path.isfile(os.path.join(lib_dir, name))
        for name in ("libacados.so", "libacados.so.1", "libacados.dylib", "acados.dll")
    )
    has_link_libs = os.path.isfile(os.path.join(lib_dir, "link_libs.json"))
    return has_lib and has_link_libs


def _preload_acados_shared_libraries(acados_root):
    if os.name == "nt":
        return
    lib_dir = os.path.join(acados_root, "lib")
    if not os.path.isdir(lib_dir):
        return
    load_mode = getattr(ctypes, "RTLD_GLOBAL", None)
    for lib_name in (
        "libblasfeo.so.0",
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libdaqp.so",
        "libosqp.so",
        "libacados.so",
    ):
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.isfile(lib_path):
            continue
        try:
            if load_mode is None:
                ctypes.CDLL(lib_path)
            else:
                ctypes.CDLL(lib_path, mode=load_mode)
        except OSError:
            continue


def ensure_acados_env():
    """Set ``ACADOS_SOURCE_DIR`` and Python/library paths in this process."""
    chosen = None
    for acados_root in acados_root_candidates():
        if _is_acados_tree(acados_root):
            chosen = acados_root
            break
    if chosen is None:
        raise RuntimeError("Built acados not found; set ACADOS_SOURCE_DIR (see README.md)")

    os.environ["ACADOS_SOURCE_DIR"] = chosen
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    acados_lib_dir = os.path.join(chosen, "lib")
    if os.path.isdir(acados_lib_dir):
        ld_entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
        if acados_lib_dir not in ld_entries:
            os.environ["LD_LIBRARY_PATH"] = ":".join([acados_lib_dir] + ld_entries)
        _preload_acados_shared_libraries(chosen)

    interface_root = os.path.join(chosen, "interfaces", "acados_template")
    if os.path.isdir(interface_root) and interface_root not in sys.path:
        sys.path.insert(0, interface_root)
    return chosen


class quiet_acados_stderr:
    """Suppress native solve diagnostics; retain the returned status in Python.

    acados prints QP failures to C stdout, not just stderr. Flush C stdio
    before restoring descriptors so buffered output cannot leak afterwards.
    """

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._libc = ctypes.CDLL(None)
        self._libc.fflush(None)
        self._fds = {fd: os.dup(fd) for fd in (1, 2)}
        devnull = os.open(os.devnull, os.O_WRONLY)
        for fd in self._fds:
            os.dup2(devnull, fd)
        os.close(devnull)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._libc.fflush(None)
        for fd, saved in self._fds.items():
            os.dup2(saved, fd)
            os.close(saved)
        return False


def _canonical_ocp_signature(payload):
    """Compare model algebra without CasADi 3.6 serializer alias IDs.

    StringSerializer preserves incidental sharing between separately packed
    SX expressions. Code generation changes that sharing; equivalent models
    therefore get different acados hashes. A single CSE Function gives a
    stable representation of every symbolic expression and its free inputs.
    All non-symbolic OCP settings remain part of the signature.
    """
    import hashlib
    import json
    import casadi as cs
    from acados_template.utils import make_object_json_dumpable, OCP_COMPARE_IGNORED_FIELD_PATHS
    payload = json.loads(json.dumps(payload, default=make_object_json_dumpable))
    payload.pop('hash', None)
    for field_path in OCP_COMPARE_IGNORED_FIELD_PATHS:
        parent = payload
        for key in field_path[:-1]:
            parent = parent.get(key, {})
        parent.pop(field_path[-1], None)
    model = payload['model']
    decoder = cs.StringDeserializer(model['serialized_expressions'])
    outputs = [decoder.unpack() for _ in model['expression_names']]
    inputs = cs.symvar(cs.vertcat(*[cs.vec(value) for value in outputs]))
    # MX already retains the compact, shared subroutines. Running CSE over
    # those calls costs ~1.5 s per warm load without improving their identity.
    # SX needs CSE to remove the incidental serializer alias differences.
    function = cs.Function('cache_model', inputs, outputs,
                           {'cse': not any(isinstance(value, cs.MX) for value in outputs)})
    model['serialized_expressions'] = function.serialize()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def acados_cache_directory(model_name):
    """Persistent, overridable cache outside the disposable /tmp tree."""
    root = os.environ.get('RETARGET_ACADOS_CACHE_DIR')
    if root is None:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        root = base / 'retarget' / 'acados'
    return Path(root).expanduser().resolve() / (model_name + '_codegen')


def prune_acados_intermediates(directory):
    """Remove generated build inputs, retaining libraries, JSON and logs."""
    directory = Path(directory)
    suffixes = {'.c', '.h', '.o', '.obj', '.a', '.s', '.pxd', '.pyx'}
    names = {'Makefile', 'CMakeLists.txt', 'CMakeCache.txt', 'cmake_install.cmake'}
    removed_bytes = 0
    for path in directory.rglob('*'):
        if path.is_file() and (path.suffix in suffixes or path.name in names):
            removed_bytes += path.stat().st_size
            path.unlink()
    for path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            try:
                path.rmdir()
            except OSError:
                pass  # Retain nonempty runtime/diagnostic directories.
    return removed_bytes


def build_cached_solver(ocp, json_file, shared_file, *, prune_generated=False, optimize_maps=()):
    """Serialize code generation shared by the simulator and planner workers.

    Current acados-template assigns a hash to an unnamed OCP. Use the model's
    existing version/configuration identity so both workers load the same
    library, then publish/load it while holding an inter-process cache lock.
    """
    import fcntl
    import inspect
    import json
    import subprocess
    import time
    from pathlib import Path
    from acados_template import AcadosOcpSolver
    legacy_json_argument = 'json_file' in inspect.signature(AcadosOcpSolver.generate).parameters
    if not legacy_json_argument:
        ocp.code_gen_options.json_file = str(json_file)
    if hasattr(ocp, 'name'):
        ocp.name = ocp.model.name
    directory = Path(json_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    optimization_file = directory / 'optimized_maps.json'
    optimize_maps = tuple(optimize_maps)
    with (directory/'.codegen.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('[acados] waiting for solver cache: %s' % ocp.model.name, flush=True, file=sys.__stdout__)
            fcntl.flock(lock, fcntl.LOCK_EX)
        current = False
        try:
            data = json.loads(Path(json_file).read_text())
            options = data.get('code_gen_options', data)
            current = (Path(shared_file).is_file()
                       and Path(shared_file).stat().st_mtime_ns >= Path(json_file).stat().st_mtime_ns
                       and all(k in options for k in ('acados_lib_path', 'code_export_directory'))
                       and data.get('name', ocp.model.name) == ocp.model.name)
        except (OSError, ValueError, TypeError):
            pass
        try:
            cached_optimization = tuple(json.loads(optimization_file.read_text()))
        except (OSError, ValueError, TypeError):
            cached_optimization = ()
        current = current and cached_optimization == optimize_maps
        reuse_check = getattr(AcadosOcpSolver, 'is_code_reuse_possible', None)
        if current and reuse_check is not None:
            # New constructors silently regenerate on a hash mismatch even
            # with generate=False/build=False. Detect it here so rebuilding
            # keeps our lock and large-map compiler flags.
            ocp.make_consistent(verbose=False)
            if hasattr(ocp, 'to_dict') and 'serialized_expressions' in data.get('model', {}):
                # A different CasADi release can leave a readable JSON with
                # an incompatible serialized graph. Rebuild that cache;
                # do not turn a cache miss into an unrelated solver fallback.
                try:
                    current = _canonical_ocp_signature(ocp.to_dict()) == _canonical_ocp_signature(data)
                except (RuntimeError, ValueError, KeyError, TypeError):
                    current = False
            else:
                current = reuse_check(ocp, verbose=False, tol_code_reuse=0.)
        if not current:
            started = time.perf_counter()
            print('[acados] generating solver: %s' % ocp.model.name, flush=True, file=sys.__stdout__)
            # Since acados 0.5.6 generate() reads the filename from the OCP;
            # passing the removed keyword made both CSO and MPC fall back.
            if legacy_json_argument:
                AcadosOcpSolver.generate(ocp, json_file=str(json_file))
            else:
                AcadosOcpSolver.generate(ocp)
            makefile = directory / 'Makefile'
            if os.name != 'nt' and makefile.is_file():
                # Unrolled SCM Hessians can contain megabytes of scalar C.
                # GCC's optimizer spends minutes on these files at -O2.
                # Keep the normal flags for small functions and the solver;
                # only skip optimization on the oversized generated maps.
                # Controllers may explicitly optimize selected hot maps;
                # the selection is part of the persistent build identity.
                large_sources = sorted(p for p in directory.rglob('*.c')
                                       if p.stat().st_size >= 256*1024)
                if large_sources:
                    with makefile.open('a') as handle:
                        handle.write('\n# Bound cold-start time for large CasADi maps.\n')
                        for source in large_sources:
                            target = source.relative_to(directory).with_suffix('.o')
                            level = 2 if optimize_maps and source.name.endswith(optimize_maps) else 0
                            handle.write('%s: CFLAGS += -O%d\n' % (target.as_posix(), level))
                print('[acados] compiling solver (%d large maps); log: %s' %
                      (len(large_sources), directory / 'build.log'), flush=True, file=sys.__stdout__)
                with (directory / 'build.log').open('w') as output:
                    result = subprocess.run(
                        ['make', '-j1', 'ocp_shared_lib'], cwd=str(directory),
                        stdout=output, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('acados compilation failed; see %s' %
                                       (directory / 'build.log'))
            else:
                AcadosOcpSolver.build(str(directory), verbose=False)
            optimization_file.write_text(json.dumps(list(optimize_maps))+'\n')
            print('[acados] solver ready in %.2fs' %
                  (time.perf_counter() - started), flush=True, file=sys.__stdout__)
        if current:
            print('[acados] reusing solver: %s' % ocp.model.name, flush=True, file=sys.__stdout__)
        load_options = dict(generate=False, build=False, verbose=False)
        if reuse_check is not None:
            load_options['check_reuse_possible'] = False
        if legacy_json_argument:
            load_options['json_file'] = str(json_file)
        solver = AcadosOcpSolver(ocp, **load_options)
        if prune_generated:
            removed = prune_acados_intermediates(directory)
            if removed:
                print('[acados] removed %.1f MiB of build intermediates' %
                      (removed / 1024**2), flush=True, file=sys.__stdout__)
        return solver
