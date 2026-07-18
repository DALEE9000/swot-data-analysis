# --- Windows DLL-order workaround (must run before any numpy BLAS call) ---
# In conda envs whose numpy uses openblas/libomp, importing pip-installed
# torch AFTER the first BLAS call crashes with WinError 127
# (STATUS_ENTRYPOINT_NOT_FOUND loading shm.dll/fbgemm.dll), and importing it
# BEFORE trips Intel-vs-LLVM OpenMP duplicate detection (OMP Error #15).
# Loading torch first with KMP_DUPLICATE_LIB_OK set lets both runtimes
# coexist; numpy/sklearn/torch results verified correct under this setup.
import os as _os

if _os.name == "nt":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
    except ImportError:
        pass  # ANN backend not installed — RF-only usage is unaffected
    except OSError as _e:
        print(f"swotxai: torch is installed but failed to load ({_e}); "
              "the ANN backend will not be available in this process.")

from swotxai.config import SWOTConfig, load_config, save_config, default_config, AVAILABLE_FEATURES
from swotxai.pipeline import run_pipeline, STEPS, SHARED_STEPS, PER_JOB_STEPS, run_shared_steps, run_per_job_steps
from swotxai.batch import BatchConfig, JobSpec, run_batch
