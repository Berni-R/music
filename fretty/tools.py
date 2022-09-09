from typing import Optional
import os
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt

from .sounds import Sound


def metronome(
        bpm: float,
        signature: int = 4,
        vol_8th: Optional[float] = None,
        vol_16th: Optional[float] = None,
        volume: float = 1.0,
        play: int = 0,
        click_wavfile: Optional[str] = None,
):
    if click_wavfile is None:
        path = os.path.dirname(__file__)
        click_wavfile = os.path.join(path, "data", "click.wav")

    if bpm <= 0:
        raise ValueError("`bpm` must be positive!")
    if not isinstance(signature, int) or signature < 1:
        raise ValueError("`signature` must be a positive integer!")

    if vol_16th is None:
        vol_16th = 0.0
    if vol_8th is None:
        vol_8th = vol_16th
    if vol_8th < 0 or vol_16th < 0 or volume < 0:
        raise ValueError("Volumnes must be non-negative!")

    if not isinstance(play, int) or play < 0:
        raise ValueError("`play` must be a non-negative integer!")

    click = Sound.from_wavfile(click_wavfile)
    clock = click.copy()
    clock._fs *= 0.8
    cluck = clock.copy()
    cluck._fs *= 0.8

    cps = 4 * bpm / 60  # 16th notes per second; 16th = 4 per beat
    dt = 1 / cps

    if dt < min(click.duration, clock.duration):
        raise ValueError(f"Too many clicks per second!")

    click = click & Sound.silence(dt)
    clock = clock & Sound.silence(dt)
    cluck = cluck & Sound.silence(dt)

    beat_rest = vol_16th * cluck + vol_8th * cluck + vol_16th * cluck
    beat = clock + beat_rest
    beat_emp = click + beat_rest

    bar = beat_emp + (signature - 1) * beat
    bar *= volume

    if play:
        (play * bar).play()

    return bar


def plot_spectrum(
        sound: Sound,
        f_max: float = 1200.0,
        nperseg: int = 2048,
        ax: Optional[mpl.axes.Axes] = None,
) -> mpl.figure.Figure:
    f, t, Zxx = signal.stft(sound._data, sound._fs, nperseg=nperseg)
    i_max = np.abs(f - f_max).argmin()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.pcolormesh(t, f[:i_max], np.abs(Zxx[:i_max]), shading='gouraud', vmin=0.0)
    ax.set_title('STFT Magnitude', fontsize=16)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_xlabel('Time [sec]', fontsize=12)

    return fig, ax
