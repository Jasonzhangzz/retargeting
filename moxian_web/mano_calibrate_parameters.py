"""GPU identification of shared MANO/glove parameters and auxiliary poses."""

def fit_capture(model_path,profile,mapping,rows,progress,cancel,**kwargs):
    from .mano_calibration import run_backend
    return run_backend('parameters',model_path,profile,mapping,rows,progress,cancel,**kwargs)
