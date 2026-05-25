"""Audio preprocessing: bandpass, spectrogram, normalization.

Default settings (per upstream): bandpass 2-10 kHz, 1.5s windows,
log-amplitude in dBFS from -55 to -10. Mel and linear-frequency variants
will be swappable via config.
"""
