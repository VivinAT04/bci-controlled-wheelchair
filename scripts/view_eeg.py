import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_gdf(
    "data/raw/A03T.gdf",
    preload=True
)

print(raw)

raw.plot(
    duration=10,
    n_channels=22,
    block=True
)

plt.show()