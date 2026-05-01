import os
import warnings
import requests
import xml.etree.ElementTree as ET
import tqdm
import sys
sys.path.append('../../SWOT-data-analysis/src')
import download_swaths
# Turn off SSL warnings (optional, not ideal in production)
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────
# Configuration: SSH Credentials & THREDDS Server Info
# ─────────────────────────────────────────────────────
ssh_kwargs = {
    "hostname": "ftp-access.aviso.altimetry.fr",
    "port": 2221,
    "username": "tdmonkman@uchicago.edu",
    "password": "2prSvl"
}
version = "v2_0_1"
product = "Unsmoothed"
base_fileserver_url = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/fileServer/"
    "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh"
)
base_catalog_url = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/catalog/"
    "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh"
)
# Local download destination
region_name = "north_pacific_1"
local_data_root = f"/home/tm3076/scratch/SWOT_L3/{region_name}/{version}/{product}"
# ─────────────────────────────────────
# Define Geographic Area and Pass IDs
# ─────────────────────────────────────
sw_corner = [150, 20]       # Southwest corner of bounding box
ne_corner = [-120.0, 55]    # Northeast corner
# Choose orbit type and cycle range
# Science orbit (21-day repeat): 001–028
cycles = [str(c).zfill(3) for c in range(1, 29)]
# Path to orbit file (science swath)
orbit_file = "../../orbit_data/sph_science_swath.zip"
# Get intersecting pass IDs (requires external function)
pass_IDs_list = download_swaths.find_swaths(
    sw_corner, ne_corner, path_to_sph_file=orbit_file
)
# ─────────────────────────────────────────────
# Parse THREDDS Catalog XML for .nc File Names
# ─────────────────────────────────────────────
def list_nc_files_from_thredds_catalog(catalog_url):
    """Return list of .nc filenames from a THREDDS catalog URL."""
    response = requests.get(catalog_url, auth=(ssh_kwargs["username"], ssh_kwargs["password"]))
    response.raise_for_status()
    ns = {'ns': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'}
    root = ET.fromstring(response.content)
    datasets = root.findall('.//ns:dataset', namespaces=ns)
    return [ds.attrib['name'] for ds in datasets if 'name' in ds.attrib and ds.attrib['name'].endswith('.nc')]
# ──────────────────────────────────────────────────
# Download a .nc File with Authentication and TQDM
# ──────────────────────────────────────────────────
def download_nc_file(download_url, save_path):
    """Download a .nc file from THREDDS fileServer with a progress bar."""
    os.makedirs(save_path, exist_ok=True)
    local_filename = os.path.join(save_path, os.path.basename(download_url))
    with requests.get(download_url, stream=True, auth=(ssh_kwargs["username"], ssh_kwargs["password"])) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 512 * 1024  # 512 KB
        with open(local_filename, 'wb') as f, tqdm.tqdm(
            total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_filename)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return local_filename
# ────────────────────────
# Main Loop: Per-Cycle Run
# ────────────────────────
for cycle in cycles:
    catalog_url = f"{base_catalog_url}/{version}/{product}/cycle_{cycle}/catalog.xml"
    nc_files = list_nc_files_from_thredds_catalog(catalog_url)
    for nc_file in nc_files:
        download_url = f"{base_fileserver_url}/{version}/{product}/cycle_{cycle}/{nc_file}"
        save_path = os.path.join(local_data_root, f"cycle_{cycle}")
        download_nc_file(download_url, save_path) 