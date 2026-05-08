# Backwards-compatibility shim — import from the new modular files instead.
from swotxai.data_utils import (
    load_swot_data,
    save_dict,
    load_dict,
    swot_regrid,
    apply_regrid,
    hfr_on_swot,
    hfr_interp,
    interp_to_swot,
    rf_flattening,
    rf_flattening_stencil,
    flattening,
    concat_flattened,
    reshaping,
    reshaping_to_xarray,
    plotter,
    plot_dict_assemble,
    build_frame_dicts,
)
from swotxai.training import train as random_forest_dispatch, predict
