"""Load raw BCI Competition IV Dataset 2a .gdf files."""

import mne

mne.set_log_level("WARNING")


def load_raw_gdf(gdf_path: str):
    """Load one raw .gdf EEG file only."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    return raw