"""Pull DUACS L4 daily geostrophic maps for each region over the science window.

Source: Copernicus Marine Service (free account; credentials from
`copernicusmarine login`). Product SEALEVEL_GLO_PHY_L4_MY_008_047
(reprocessed, 1/8 deg, daily), dataset
cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D. If the
reprocessed series does not reach the end of the window, the gap is filled
from the near-real-time sibling (SEALEVEL_GLO_PHY_L4_NRT_008_046) and the
output records which days came from which.

Nothing here touches S3 or any paid API. Each region subset is ~20-100 MB.
Resumable: existing files are skipped.

Run (project env):
    python scripts/duacs_pull.py
Output: experiments/{rid}/duacs/duacs_l4_{rid}_20230701_20240630.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import copernicusmarine as cm
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# [lon, lat] corners from swotxai.presets (science footprints), padded 0.5 deg
BBOX = {
    "uswc":  ([-126.8, 31.2], [-116.6, 49.9]),
    "usegc": ([-97.7, 22.8], [-68.0, 44.4]),
    "gak":   ([-131.2, 53.6], [-129.8, 54.8]),
    "glna":  ([-85.4, 45.3], [-84.2, 46.4]),
    "prvi":  ([-68.4, 15.8], [-63.8, 19.6]),
    "ushi":  ([-159.4, 19.3], [-154.3, 22.1]),
}
VARS = ["ugos", "vgos", "sla", "adt", "err_ugosa", "err_vgosa"]
MY = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
NRT = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
T0, T1 = "2023-07-01T00:00:00", "2024-06-30T23:59:59"
PAD = 0.5


def pull(rid: str, out_dir: Path, T0: str = T0, T1: str = T1) -> Path:
    tag = T0[:10].replace("-", "") + "_" + T1[:10].replace("-", "")
    out = out_dir / f"duacs_l4_{rid}_{tag}.nc"
    if out.exists():
        print(f"  {rid}: exists, skipping")
        return out
    (lon0, lat0), (lon1, lat1) = BBOX[rid]
    kw = dict(variables=VARS, minimum_longitude=lon0 - PAD, maximum_longitude=lon1 + PAD,
              minimum_latitude=lat0 - PAD, maximum_latitude=lat1 + PAD,
              output_directory=str(out_dir), disable_progress_bar=True, overwrite=True)
    tmp_my = out_dir / f"_my_{rid}.nc"
    if not tmp_my.exists():
        cm.subset(dataset_id=MY, start_datetime=T0, end_datetime=T1, output_filename=tmp_my.name, **kw)
    with xr.open_dataset(tmp_my) as f:          # close before unlink (Windows file lock)
        ds = f.load()
    tmp_my.unlink()
    ds["source"] = ("time", np.array(["MY"] * ds.time.size, dtype="U3"))
    last = np.datetime64(ds.time.values.max(), "D")
    if last < np.datetime64(T1[:10]):
        tmp_nrt = out_dir / f"_nrt_{rid}.nc"
        if not tmp_nrt.exists():
            cm.subset(dataset_id=NRT, start_datetime=str(last + np.timedelta64(1, "D")), end_datetime=T1,
                      output_filename=tmp_nrt.name, **kw)
        with xr.open_dataset(tmp_nrt) as f:
            dn = f.load()
        tmp_nrt.unlink()
        dn["source"] = ("time", np.array(["NRT"] * dn.time.size, dtype="U3"))
        ds = xr.concat([ds, dn[list(ds.data_vars)]], dim="time")
        print(f"  {rid}: MY ends {last}; {dn.time.size} days filled from NRT")
    ds.attrs.update(dataset_my=MY, dataset_nrt=NRT, pulled="copernicusmarine toolbox",
                    note="DUACS allsat L4 daily 1/8 deg; 'source' marks MY (reprocessed) vs NRT days")
    ds.to_netcdf(out)
    print(f"  {rid}: {ds.time.size} days, grid {ds.latitude.size}x{ds.longitude.size}, "
          f"{out.stat().st_size/1e6:.1f} MB -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="uswc,usegc,prvi,ushi,gak,glna")
    ap.add_argument("--start", default=T0, help="ISO datetime; default = science window start")
    ap.add_argument("--end", default=T1, help="ISO datetime; default = science window end")
    a = ap.parse_args()
    for rid in a.regions.split(","):
        out_dir = ROOT / "experiments" / rid / "duacs"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {rid}")
        pull(rid, out_dir, a.start, a.end)
