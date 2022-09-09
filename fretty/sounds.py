from typing import Optional, Union
import pyaudio
import numpy as np
from scipy.io import wavfile
import warnings

from .notes import Note
from .scales import Scale


DEF_SAMPLE_FREQ: int = 44_100


def resample(sound: np.ndarray, orig_fs: int, new_fs: int) -> np.ndarray:
    T = len(sound) / orig_fs
    dt_orig = 1.0 / orig_fs
    dt_new = 1.0 / new_fs

    t_orig = np.linspace(0, T - dt_orig, num=len(sound))
    t_new  = np.arange(0, T, step=dt_new)

    return np.interp(t_new, t_orig, sound)


class Sound:

    def __init__(self, data: np.ndarray, fs: int = DEF_SAMPLE_FREQ):
        self._data = data
        self._fs = fs

    @classmethod
    def silence(cls, duration: float, fs: int = DEF_SAMPLE_FREQ) -> 'Sound':
        data = np.zeros(int(fs * duration))
        return Sound(data, fs)

    @classmethod
    def from_wavfile(cls, filename: str) -> 'Sound':
        fs, data = wavfile.read(filename)

        if np.issubdtype(data.dtype, np.unsignedinteger):
            zero = np.iinfo(data.dtype).max // 2 + 1
            data = (data.astype(float) - zero) / zero
        elif np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)
        else:
            raise RuntimeError(f"Unexpected data type {data.dtype}")

        # combine potential channels into mono channel
        if data.ndim > 1:
            data = data.sum(axis=1)

        return Sound(data, fs=fs)

    def to_wavfile(self, filename: str):
        wavfile.write(filename, self._fs, self._data)

    def resample(self, fs: int, inplace: bool = True) -> 'Sound':
        resampled = resample(self._data, self._fs, fs)
        if inplace:
            self._data = resampled
            self._fs = fs
            return self
        else:
            return Sound(resampled, fs)

    @property
    def duration(self) -> float:
        return len(self._data) / self._fs

    @property
    def sample_freq(self) -> int:
        return self._fs

    def data(self, copy: bool = True, fs: int = DEF_SAMPLE_FREQ):
        if fs != self._fs:
            data = resample(self._data, self._fs, fs)
        else:
            data = np.array(self._data, copy=copy)
        return data

    def times(self) -> float:
        return np.arange(len(self._data)) / self._fs

    def __len__(self):
        return len(self._data)

    def copy(self) -> 'Sound':
        return Sound(self._data.copy(), self._fs)

    def add(self, sound: 'Sound', start_pos: Optional[float] = None) -> 'Sound':
        fs = max(self._fs, sound._fs)
        if start_pos is None:
            start_pos = self.duration
        if start_pos < 0:
            raise ValueError(f"`start_pos` must be non-negative, got {start_pos}.")

        new_duration = max(self.duration, start_pos + sound.duration)

        comb = Sound.silence(new_duration, fs)

        self_data = self.data(copy=False, fs=fs)
        assert len(comb._data) >= len(self_data) - 1
        l = min(len(self_data), len(comb._data))
        comb._data[:l] = self_data[:l]

        i = int(start_pos * fs)
        assert 0 <= i <= len(comb._data)
        sound_data = sound.data(copy=False, fs=fs)
        assert len(comb._data) >= len(sound_data) - 1
        assert len(comb._data) >= i + len(sound_data) - 1
        l = min(len(sound_data), len(comb._data) - i)
        comb._data[i : i + l] += sound_data[:l]

        return comb

    def __add__(self, sound: 'Sound') -> 'Sound':
        return self.add(sound, start_pos=None)

    def __and__(self, sound: 'Sound') -> 'Sound':
        return self.add(sound, start_pos=0.0)

    def __mul__(self, factor: Union[float, int]) -> 'Sound':
        if isinstance(factor, int):
            if factor < 1:
                raise ValueError(f"Cannot play sound {factor} times.")
            sound = self.copy()
            for _ in range(factor - 1):
                sound += self
            return sound
        elif isinstance(factor, float):
            return Sound(factor * self._data, self._fs)
        else:
            raise TypeError(f"Cannot multiply `Sound` with `{type(factor)}`")

    def __rmul__(self, factor: Union[float, int]) -> 'Sound':
        return self.__mul__(factor)

    def __rmul__(self, factor: int) -> 'Sound':
        return self.__mul__(factor)

    def modulate(self, amplitudes: np.ndarray, inplace: bool = True) -> 'Sound':
        if inplace:
            self._data *= amplitudes
            return self
        return Sound(amplitudes * self._data, self._fs)

    def lowpass(self, freq: float, order: int = 5, inplace: bool = True) -> 'Sound':
        from scipy import signal

        sos = signal.butter(order, freq, fs=self._fs, output='sos')
        filtered = signal.sosfilt(sos, self._data)

        if inplace:
            self._data[:] = filtered
            return self
        else:
            return Sound(filtered, self._fs)

    def play(self, lowpass: Optional[float] = 5000.0):
        sound = self if lowpass is None else self.lowpass(lowpass, inplace=False)

        with SoundContext(fs=self._fs) as sc:
            sc.play(sound)


class SineSound(Sound):

    def __init__(self, freq: Union[float, Note, str], duration: float = 0.5, volume: float = 0.1,
                 attacked: bool = True,
                 A4: float = 440.0, fs: int = DEF_SAMPLE_FREQ):
        if isinstance(freq, str):
            freq = Note(freq)
        if isinstance(freq, Note):
            freq = freq.pitch(A4=A4)
        if fs < 10 * freq:
            warnings.warn(f"Sine wave frequency {freq:.3e} not much higher than samping frequency {fs:,d}!")

        data = volume * np.sin(2.0 * np.pi * np.arange(fs * duration) * freq / fs)
        super(SineSound, self).__init__(data, fs)
        self._freq = freq

        if attacked:
            t = self.times()
            amp = 2 * (0.01 / (t + 0.02)) ** 0.5
            self.modulate(amp, inplace=True)

    @property
    def freq(self) -> float:
        return self._freq


class SoundContext:

    def __init__(self, fs: int = DEF_SAMPLE_FREQ):
        self._fs = fs
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def __enter__(self):
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self._fs,
            output=True,
        )
        self.play(Sound.silence(duration=0.0, fs=self._fs))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.play(Sound.silence(duration=0.0, fs=self._fs))

        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

        self._audio.terminate()
        self._audio = None

    def play(self, sound: Union[np.ndarray, Sound], wait: bool = False) -> np.ndarray:
        if isinstance(sound, Sound):
            sound = sound.data()

        if self._stream.is_stopped():
            self._stream.start_stream()

        self._stream.write(sound.astype(np.float32).tobytes())

        if wait:
            self._stream.stop_stream()
            self._stream.start_stream()