from typing import Union
import numpy as np
import regex
from copy import deepcopy


_A4_FREQ = 440.0


class Note:
    f"""A note in a twelve-tone equal temperament.

    Args:
        s:  The name of the note with octave, such as A4 ({_A4_FREQ:.1f} Hz),
            F#3 or Bb4. The name itself ("F" in "F#3") has to be upper case,
            accidentals can be concatenated (like in "F##b#3"), and one has
            to specify the octave.

    Each semitone is simply a frequency ratio of 2^(1/12) and, for instance,
    C# and Db are the same note."""
    __slots__ = ['_note', '_octave']

    def __init__(self, s: str):
        m = regex.match(r"([A-G])([#b]*)(\d+)", s)
        if m is None:
            raise ValueError(f"note name is not valid: '{s}'")
        letter, modifier, octave = m.groups()

        note = "C_D_EF_G_A_B".index(letter)
        note += modifier.count("#")
        note -= modifier.count("b")

        self._note: int = note % 12
        self._octave: int = int(octave) + note // 12

    @classmethod
    def closest_to(
            cls,
            freq: float, ret_cents: bool = False,
            A4: float = _A4_FREQ,
    ) -> 'Note':
        f"""Find the note that is closest to a given frequence
        (assuming A4 has {_A4_FREQ:.1f} Hz).

        Args:
            freq (float):       ...
            ret_cents (bool):   Whether to also return how many cents the given
                                frequency is off the closest note.
            A4 (float):         Frequency of A4."""
        semitones_f = np.log2(freq / A4) * 12
        semitones = np.round(semitones_f)
        note = Note("A4") + int(semitones)
        if ret_cents:
            cents_off = (semitones_f - semitones) * 100
            return note, cents_off
        else:
            return note

    def name(self, flat: bool = False):
        """Get the name of the note (no octave; with octave: `as_str()`).

        Args:
            flat (bool):    If True, use s flat key signature and, for
                            instance, use Gb. If False, use a sharp key
                            signature and then use F# for the same note (we are
                            in a twelve-tone equal temperament)."""
        if flat:
            names = "C Db D Eb E F Gb G Ab A Bb B"
        else:
            names = "C C# D D# E F F# G G# A A# B"
        return names.split()[self._note]

    def octave(self) -> int:
        """Return the octave of the note, so "D#4" -> 4."""
        return self._octave

    def as_str(self, flat: bool = False) -> str:
        """Get the name of the note (including octave; without octave: `name()`).

        Args:
            flat (bool):    If True, use s flat key signature and, for
                            instance, use Gb. If False, use a sharp key
                            signature and then use F# for the same note (we are
                            in a twelve-tone equal temperament)."""
        return f"{self.name(flat=flat)}{self._octave}"

    def copy(self) -> 'Note':
        """Create a (deep) copy of this instance."""
        res = Note.__new__(Note)
        res._note = self._note
        res._octave = self._octave
        return res

    def __str__(self) -> str:
        return f"{self.name()}{self._octave}"

    def __repr__(self) -> str:
        return f"Note('{self}')"

    def pitch(self, A4: float = _A4_FREQ) -> float:
        """Frequency of this note in Hertz."""
        return float(A4) * 2 ** (self._octave - 4 + (self._note - 9) / 12)

    def __truediv__(self, note: 'Note') -> float:
        return self.pitch() / note.pitch()

    def __sub__(self, note: Union['Note', str]) -> float:
        if isinstance(note, Note):
            self_pitch = 12 * self._octave + self._note
            note_pitch = 12 * note._octave + note._note
            return self_pitch - note_pitch
        else:
            return self.__add__(-note)

    def __add__(self, half_tones: int) -> 'Note':
        note = self._note + half_tones

        res = Note.__new__(Note)
        res._note = note % 12
        res._octave = int(self._octave) + note // 12
        return res


INTERVAL_NAME = [
    'unison',
    'semitone',  # 'minor second'
    'major second',
    'minor third',
    'major third',
    'fourth',
    'tritone',
    'fifth',
    'minor sixth',
    'major sixth',
    'minor seventh',
    'major seventh',
    'octave',
]


class Tuning:
    """A simple class holding the notes of the open strings as a list in the
    attribute `strings`.

    You can either pass a string of form "<E>-<A>-<D>-<G>-<B>-<e>", where the
    <X> stand for the individual strings and are note names. An example would
    be `Tuning("E2-A2-D3-G3-B3-E4)"`, which is the standard guitar tuning.
    Or you pass a series of Notes as argument, which then will be used for the
    strings, such as `Tuning(Note("E2"), Note("A2"), ...)`.
    """

    def __init__(self, *strings):
        if len(strings) == 1 and isinstance(strings[0], str):
            strings = strings[0].split("-")
            strings = list(Note(s) for s in strings)
        if len(strings) not in [6, 7, 8]:
            raise ValueError("A guitar should have 6 (or 7 or 8) strings.")
        if not all(isinstance(s, Note) for s in strings):
            raise TypeError("Each string should be of type Note.")
        self.strings = deepcopy(strings)

    @classmethod
    def from_name(cls, name: str) -> 'Tuning':
        try:
            tuning = KNOWN_TUNINGS[name]
        except KeyError:
            known = ', '.join(KNOWN_TUNINGS.keys())
            raise KeyError(f'Unknown tuning "{name}". Choose from: {known}')
        return tuning.copy()

    def copy(self) -> 'Tuning':
        """Create a (deep) copy of this instance."""
        return Tuning(*self.strings)

    def __str__(self) -> str:
        return '-'.join(str(s) for s in self.strings)

    def __repr__(self) -> str:
        return 'Tuning("' + '-'.join(str(s) for s in self.strings) + '")'


STANDARD_TUNING = "E2-A2-D3-G3-B3-E4"
KNOWN_TUNINGS = {
    'standard': Tuning(STANDARD_TUNING),
    'drop D':   Tuning("D2-A2-D3-G3-B3-E4"),
}
