# Glossary of acronyms in the research notes

Compiled 2026-10-28 from `papers/NOTES.md`, `docs/research-log.md`,
`docs/research-plan.md` and `docs/preprint-style-checklist.md`. Grouped by
topic; within a group, alphabetical. Where an acronym is a product or
dataset, the owner is named so you can cite it.

## Satellite altimetry and the SWOT mission

- **SWOT** — Surface Water and Ocean Topography. NASA/CNES wide-swath altimeter launched December 2022; the project's input.
- **KaRIn** — Ka-band Radar Interferometer, the SWOT instrument that measures the two 50-km swaths.
- **SSH** — sea surface height, the height of the ocean surface above a reference ellipsoid.
- **SSHA** — sea surface height anomaly; SSH minus a mean sea surface. The `ssha_filtered` feature.
- **SLA** — sea level anomaly; the DUACS/CMEMS name for the same quantity as SSHA.
- **ADT** — absolute dynamic topography; SLA plus MDT, the full height field that geostrophy acts on.
- **MDT** — mean dynamic topography; the time-mean of SSH above the geoid, which encodes the mean currents. The `mdt` feature (product CNES-CLS22).
- **MSS** — mean sea surface; the time-mean SSH above the ellipsoid (geoid plus MDT).
- **L1 / L2 / L3 / L4** — processing levels. L1: calibrated instrument data. L2: geophysical variables along the track or swath. L3: swath data with corrections, filtering and cross-calibration applied (the project uses SWOT L3 Expert, reprocessing v3). L4: gap-free gridded maps merged from several satellites (DUACS).
- **DUACS** — Data Unification and Altimeter Combination System. The CNES/CLS processing chain that produces the multi-satellite L3 and L4 altimetry products distributed by AVISO and CMEMS. "DUACS L4" means the daily gridded nadir-altimeter geostrophic maps, the standard baseline in the field.
- **AVISO** — Archiving, Validation and Interpretation of Satellite Oceanographic data; the CNES distribution portal for altimetry, including SWOT L3.
- **CMEMS** — Copernicus Marine Environment Monitoring Service (now Copernicus Marine Service); the European distributor of DUACS L4 and GLORYS.
- **CNES** — Centre National d'Études Spatiales, the French space agency and SWOT co-owner.
- **CLS** — Collecte Localisation Satellites, the French company that runs DUACS for CNES.
- **PO.DAAC / PODAAC** — Physical Oceanography Distributed Active Archive Center; NASA's ocean data archive at JPL, where SWOT and NeurOST are served.
- **NRT** — near real time; the fast, lower-accuracy delivery stream of a product (as opposed to delayed-time or reprocessed).
- **DT2018** — the 2018 delayed-time reprocessing of the DUACS altimetry record.
- **NEURO-STV24** — the PO.DAAC short name of the NeurOST SSH-SST product of Martin et al. 2024.
- **MIOST** — Multiscale Inversion of Ocean Surface Topography; a CLS mapping method used for experimental SWOT-inclusive L4 maps.
- **BFN-QG** — Back-and-Forth Nudging with a quasi-geostrophic model; a dynamical mapping method for altimetry.
- **MASSH** — Mapping Altimetry with Surface State Hydrodynamics; the open-source toolbox that implements BFN-QG and related mappers.
- **WOC** — World Ocean Circulation; the ESA project whose surface-current product (geostrophy plus wind response) is Ubelmann et al. 2025.
- **GOFLOW** — the GOES-based surface current product of Lenain et al. 2026 (SST-gradient U-Net).

## Ocean dynamics

- **QG** — quasi-geostrophic; the approximation for slow, rotating, stratified flow in which velocity is nearly in geostrophic balance with pressure.
- **SQG** — surface quasi-geostrophic; the version of QG in which the interior is passive and surface density (or SSH) determines the flow at depth. Carli et al. 2024 use it to project surface fields downward.
- **IGW** — internal gravity waves; waves inside the stratified ocean, including internal tides. Unbalanced motion that SWOT sees in SSH but that geostrophy misreads as current.
- **ISW** — internal solitary waves; nonlinear, short internal waves.
- **EKE** — eddy kinetic energy; half the variance of the velocity anomaly.
- **KE** — kinetic energy.
- **MLD** — mixed-layer depth; the depth of the well-mixed surface layer, which sets how deep wind momentum is spread (hence the seasonal change in Ubelmann's wind response).
- **FTLE** — finite-time Lyapunov exponent; a Lagrangian measure of stretching used to map transport barriers.
- **M2, S2, N2, K1, O1, S1** — tidal constituents. M2 principal lunar semidiurnal (12.42 h), S2 principal solar semidiurnal (12.00 h), N2 lunar elliptic semidiurnal (12.66 h), K1 lunisolar diurnal (23.93 h), O1 principal lunar diurnal (25.82 h), S1 solar diurnal (24.00 h, mostly radiational).
- **FES2014** — Finite Element Solution 2014; the global barotropic tide model (heights and currents) from LEGOS/CNES used by Kachelein to subtract the phase-locked tide.
- **ACC** — Antarctic Circumpolar Current.
- **CCS** — California Current System; the eastern-boundary upwelling system covered by the uswc radar network.
- **EGC** — East Greenland Current (Jensen et al. 2025).

## Ground truth and in situ data

- **HFR / HF radar** — high-frequency radar; shore-based radar that measures surface currents in the top metre from the Doppler shift of Bragg-scattered radio waves. The project's truth.
- **HFRNet** — the US national HF radar network run by Scripps CORDC for IOOS.
- **IOOS** — Integrated Ocean Observing System; the NOAA program that funds HFRNet.
- **CORDC** — Coastal Observing Research and Development Center at Scripps.
- **CODAR** — the commercial HF radar system most US sites use (originally Coastal Ocean Dynamics Applications Radar).
- **VHF** — very high frequency; higher-frequency radar variant with shorter range and finer resolution.
- **HDOP** — horizontal dilution of precision; the factor by which antenna geometry amplifies radial measurement error into vector error at a grid cell. Lower is better; HFRNet releases cells at or below 1.25 (the project's files were screened at 1.0).
- **DOPX / DOPY** — the east and north components of the dilution of precision.
- **RTV** — real-time vector; the hourly HFRNet surface-current product.
- **ADCP** — acoustic Doppler current profiler; moored or ship-mounted instrument that measures current profiles with depth (Tchonang's truth, at 30–40 m).
- **GDP** — Global Drifter Program; the NOAA/AOML array of satellite-tracked surface drifters drogued at 15 m (truth for Ubelmann, Martin).
- **AOML** — Atlantic Oceanographic and Meteorological Laboratory, the NOAA lab that runs the GDP.
- **GLAD / LASER** — Grand Lagrangian Deployment and LAgrangian Submesoscale ExpeRiment; two large Gulf of Mexico drifter releases (CARTHE) used as high-density truth.
- **NDBC** — National Data Buoy Center; NOAA buoys and the THREDDS mirror of HFR data.
- **DART** — Deep-ocean Assessment and Reporting of Tsunamis; bottom-pressure gauges Kachelein used to check the barotropic tide.
- **THREDDS** — Thematic Real-time Environmental Distributed Data Services; the OPeNDAP-style data server from which HFRNet is pulled.
- **PRVI** — Puerto Rico and the US Virgin Islands (the CARICOOS radar region).
- **USWC / USEGC / GAK / GLNA / USHI / AKNS** — the HFRNet regional grids: US West Coast, US East and Gulf Coast, Gulf of Alaska, Great Lakes North America, US Hawaii, Alaska North Slope.

## Other satellite and model inputs

- **SST** — sea surface temperature.
- **SSS** — sea surface salinity.
- **GOES** — Geostationary Operational Environmental Satellite; the NOAA geostationary series whose ABI imager gives hourly SST over the Americas.
- **ABI** — Advanced Baseline Imager, the GOES-R series instrument.
- **MUR** — Multi-scale Ultra-high Resolution SST; a daily gap-free L4 SST analysis from JPL.
- **ODYSSEA** — a European L4 SST analysis (Ifremer).
- **ERA5** — the fifth-generation ECMWF atmospheric reanalysis; the source of the hourly 10-m winds (`era5_u`, `era5_v`).
- **CDS** — Climate Data Store; the Copernicus portal that serves ERA5.
- **SAR** — synthetic aperture radar; imaging radar that sees surface roughness (fronts, internal waves, slicks).
- **GLORYS / GLORYS12** — the Mercator Ocean global 1/12° ocean reanalysis distributed by CMEMS; Martin et al. 2025's OSSE truth.
- **LLC4320 / LLC2160** — MITgcm global simulations on the lat-lon-cap grid at 1/48° and 1/24°; the standard high-resolution "nature runs" for SWOT studies.
- **MITgcm / MIT** — the MIT general circulation model.
- **CESM / POP** — Community Earth System Model and its ocean component, the Parallel Ocean Program (Sinha and Abernathey 2021's 0.1° run).
- **GCM** — general circulation model.
- **GEOS** — in the notes, short for geostrophic velocity (not the NASA GEOS model).

## Evaluation and statistics

- **RMSE / RMSD** — root-mean-square error / difference.
- **MSE / MAE** — mean squared error / mean absolute error.
- **RMS** — root mean square (of a signal, not an error).
- **R²** — coefficient of determination; 1 minus the error variance over the total variance. Nash–Sutcliffe efficiency is the same formula with the observed mean as reference.
- **CI** — confidence interval.
- **IQR** — interquartile range.
- **STD / SD** — standard deviation.
- **PDF / JPDF** — probability density function / joint PDF (used for vorticity–strain histograms).
- **PSD** — power spectral density.
- **DFT / FFT** — discrete Fourier transform / fast Fourier transform.
- **DSP** — digital signal processing.
- **OI** — optimal interpolation; the statistical mapping behind DUACS L4.
- **OSE / OSSE** — observing system experiment / observing system simulation experiment. OSE: withhold real data and measure the loss. OSSE: run everything inside a model so the truth is known.
- **OOD** — out of distribution; test data unlike the training data (e.g. a new region).
- **UQ** — uncertainty quantification.
- **CRPS** — continuous ranked probability score; a proper score for probabilistic forecasts.
- **QC** — quality control.
- **N_eff** — effective sample size; the number of independent samples implied by the autocorrelation of the data.

## Machine learning

- **ML / DL** — machine learning / deep learning.
- **ANN / NN / MLP** — artificial neural network / neural network / multilayer perceptron (the project's `ann` backend).
- **CNN** — convolutional neural network.
- **RF** — random forest (the project's `rf` backend).
- **GP / GPML** — Gaussian process / Gaussian processes for machine learning (Berlinghieri et al. 2023).
- **HNN** — Helmholtz neural network (a network whose output is a streamfunction and a potential).
- **PINN** — physics-informed neural network.
- **LLM** — large language model (the AlphaEvolve mutator).
- **RL** — reinforcement learning (Biferale et al. 2019).
- **DA** — data assimilation.
- **LR** — learning rate (in training notes); occasionally linear regression.
- **GPU / CPU** — graphics / central processing unit.
- **EMA** — exponential moving average (of weights, in the champion).

## Journals, venues and institutions

- **AGU** — American Geophysical Union (publisher of GRL, JGR, JAMES, ESS).
- **AMS** — American Meteorological Society (publisher of JPO, JTECH, AIES).
- **GRL** — Geophysical Research Letters.
- **JGR** — Journal of Geophysical Research (Oceans).
- **JAMES** — Journal of Advances in Modeling Earth Systems.
- **ESS** — Earth and Space Science.
- **JPO** — Journal of Physical Oceanography.
- **JTECH** — Journal of Atmospheric and Oceanic Technology.
- **AIES** — Artificial Intelligence for the Earth Systems (AMS).
- **ICML** — International Conference on Machine Learning.
- **NOAA** — National Oceanic and Atmospheric Administration.
- **ESA** — European Space Agency.
- **DOI** — digital object identifier.
- **CC-BY** — Creative Commons Attribution licence.

## Project and computing

- **SSV** — the project's target quantity: surface (originally "subsurface") velocity inferred from SWOT. Used as the project name in the notes.
- **API / CLI / UI / DB** — application programming interface / command-line interface / user interface / database.
- **S3** — Amazon Simple Storage Service; the project bucket `swot-ai-ssv`.
- **GB** — gigabyte.
- **UTC** — Coordinated Universal Time.
