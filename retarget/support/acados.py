def _set_acados_code_export_directory(ocp, directory):
    """Support legacy and current acados-template codegen APIs."""
    options = getattr(ocp, "code_gen_opts", None)
    if options is not None:
        options.code_export_directory = directory
    elif hasattr(ocp, "code_export_directory"):
        ocp.code_export_directory = directory
    else:
        raise AttributeError(
            "AcadosOcp exposes neither code_gen_opts nor code_export_directory"
        )

def _load_acados():
    from retarget.support.acados_env import ensure_acados_env
    ensure_acados_env()
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver
