from typing import Union
import numpy as np
import regex


class Note:
    """A note in a twelve-tone equal temperament.
    
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

        self._note = note % 12
        self._octave = int(octave) + note // 12

    @classmethod
    def closest_to(cls, freq: float, A4: float = 440.0) -> 'Note':
        semitones = np.log2(freq / A4) * 12
        return Note("A4") + int(np.round(semitones))

    def note_name(self, flat: bool = False):
        """Get the name of the note (no octave).
        
        Args:
            flat (bool):    If True, use s flat key signature and, for instance,
                            use Gb. If False, use a sharp key signature and then
                            use F# for the same note (we are in a twelve-tone
                            equal temperament)."""
        if flat:
            names = "C Db D Eb E F Gb G Ab A Bb B"
        else:
            names = "C C# D D# E F F# G G# A A# B"
        return names.split()[self._note]

    def octave(self) -> int:
        """Return the octave of the note, so "D#4" -> 4."""
        return self._octave

    def copy(self) -> 'Note':
        """Create a (deep) copy of this instance."""
        res = Note.__new__(Note)
        res._note = self._note
        res._octave = self._octave
        return res

    def __str__(self) -> str:
        return f"{self.note_name()}{self._octave}"

    def __repr__(self) -> str:
        return f"Note('{self}')"

    def pitch(self, A4: float = 440.0) -> float:
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
    'semitone',  #'minor second'
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
    """A simple class holding the notes of the open strings as a list in the attribute `strings`."""

    def __init__(self, *strings):
        if len(strings) == 1 and isinstance(strings[0], str):
            strings = strings[0].split("-")
        self.strings = tuple(Note(s) for s in strings)

    def copy(self) -> 'Tuning':
        """Create a (deep) copy of this instance."""
        strings = [str(s) for s in self.strings]
        return Tuning(*strings)

    def __str__(self) -> str:
        return '-'.join(str(s) for s in self.strings)

    def __repr__(self) -> str:
        return 'Tuning("' + '-'.join(str(s) for s in self.strings) + '")'


TUNING_STANDARD = Tuning("E2-A2-D3-G3-B3-E4")
TUNING_DROP_D = Tuning("D2-A2-D3-G3-B3-E4")
