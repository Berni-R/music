from collections import OrderedDict
from typing import Optional, Iterable, Union
import regex

from .notes import Note


NAMED_SCALES = OrderedDict([
    ('maj', (0, 2, 4, 5, 7, 9, 11)),
    ('min', (0, 2, 3, 5, 7, 8, 10)),
    ('major', (0, 2, 4, 5, 7, 9, 11)),
    ('minor', (0, 2, 3, 5, 7, 8, 10)),

    ('ionian',     (0, 2, 4, 5, 7, 9, 11)),
    ('dorian',     (0, 2, 3, 5, 7, 9, 10)),
    ('phrygian',   (0, 1, 3, 5, 7, 8, 10)),
    ('lydian',     (0, 2, 4, 6, 7, 9, 11)),
    ('mixolydian', (0, 2, 4, 5, 7, 9, 10)),
    ('aeolian',    (0, 2, 3, 5, 7, 8, 10)),
    ('locrian',    (0, 1, 3, 5, 6, 8, 10)),

    ('major pentatonic', (0, 2, 4, 7,  9)),
    ('minor pentatonic', (0, 3, 5, 7, 10)),

    ('blues scale', (0, 3, 5, 6, 7, 10)),

    ('melodic minor', (0, 2, 3, 5, 7, 9, 11)),
    ('melodic major', (0, 2, 4, 5, 7, 8, 11)),

    ('Persian scale', (0, 1, 4, 5, 6, 8, 11)),

    ('chromatic', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)),
])

NAMED_CHORDS = {
    'maj': (0, 4, 7),
    'min': (0, 3, 7),

    'aug': (0, 4, 8),
    'dim': (0, 3, 6),

    '5': (0, 7, 12),  # power chord

    'sus2': (0, 2, 7),
    'sus4': (0, 5, 7),

    'maj7': (0, 4, 7, 11),
    'min7': (0, 3, 7, 10),
}


class BaseScale:

    def __init__(self, base: str, steps: Iterable[int]):
        self._base = str(Note(base + "4"))[:-1]
        self._steps = list(sorted(s % 12 for s in {0} | set(steps)))

    @classmethod
    def _from_name(cls, name: str, show_symbol: str, pattern: str, db: dict):
        m = regex.match(pattern, name)
        if m is None:
            raise ValueError(f"'{name}' does not match the {show_symbol} name pattern.")
        base, name = m.groups()
        if name not in db:
            raise ValueError(f"Unknown {show_symbol} name '{name}'. Known {show_symbol}: {db.keys()}")
        return cls(base.upper(), db[name])

    @property
    def base(self) -> str:
        return self._base

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def note_names(self) -> list[str]:
        """Get a list of the names (no octave) of the notes in this scale."""
        sharps = [(Note(self.base + "4") + s).name() for s in self.steps]
        flats = [(Note(self.base + "4") + s).name(flat=True) for s in self.steps]
        if set(x[0] for x in flats) > set(x[0] for x in sharps):
            return flats
        else:
            return sharps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {[s for s in self.steps]})"

    def rotate(self, n: int) -> 'BaseScale':
        """Return the scale with the same notes, but take the n-th note as base."""
        n = n % len(self._steps)
        steps = self._steps[n:] + [12 + s for s in self._steps[:n]]
        note = Note(self.base + "4")
        shift = steps[0]
        base = (note + shift).name()
        steps = [s - shift for s in steps]
        return BaseScale(base, steps)

    def contains(self, note: Union[Note, str]) -> bool:
        """Does this scale contain the given note?"""
        if isinstance(note, str):
            note = Note(note)
        inter = note - Note(self._base + "4")
        return inter % 12 in self._steps


class Scale(BaseScale):

    @classmethod
    def from_name(cls, name: str) -> 'Scale':
        """Create a scale from a trivial name such as "Cmaj"."""
        return super()._from_name(name, "scale", r"([a-gA-G][#b]*)[ -]?(.*)", NAMED_SCALES)

    def name(self, sep: str = "-") -> Optional[str]:
        """Get the trivial name (e.g. "Emin") of this scale, or None if unknown."""
        steps = set(self._steps)
        if steps == {0, 2, 4, 5, 7, 9, 11}:
            return self._base + "maj"
        if steps == {0, 2, 3, 5, 7, 8, 10}:
            return self._base + "min"
        for name, s in NAMED_SCALES.items():
            if steps == set(s):
                return self._base + sep + name
        return None

    def key_signature(self) -> str:
        """Return the scale's key signature.
        
        This is be an empty string for Cmaj, 'b' for Fmaj, or '###' for Amaj."""
        notes = self.note_names()
        return ''.join(n[1:] for n in notes if len(n) > 1)

    def rotate(self, n: int) -> 'Scale':
        """Return the scale with the same notes, but take the n-th note as base."""
        base = super().rotate(n)
        base.__class__ = self.__class__
        return base

    def chords(self, search_chords: Iterable[str] = ('maj', 'min')) -> list[str]:
        """Return the name of the chords that are in this scale."""
        chords = {name: steps for name, steps in NAMED_CHORDS.items() if name in search_chords}

        chord_names = []
        for r in range(len(self._steps)):
            rebase = self.rotate(r)
            for name, steps in chords.items():
                if set(steps).issubset(rebase.steps):
                    chord_names.append(rebase.base + name)

        return chord_names

    def play(self, octave: int = 3, down: bool = False, note_duration: float = 0.2):
        from .sounds import SoundContext, SineSound

        base = Note(f"{self._base}{octave}")
        notes = [base + s for s in self._steps] + [base + 12]

        if down:
            notes = notes[::-1]

        with SoundContext() as sc:
            for n in notes[:-1]:
                sc.play(SineSound(n, duration=note_duration))
            sc.play(SineSound(notes[-1], duration=7 * note_duration))


class Chord(BaseScale):

    @classmethod
    def from_name(cls, name: str) -> 'Chord':
        """Create a chord from a trivial name such as "Dmin"."""
        return super()._from_name(name, "chord", r"([a-gA-G][#b]*)(.*)", NAMED_CHORDS)

    def name(self) -> Optional[str]:
        """Get the trivial name (e.g. "Emin") of this chord, or None if unknown."""
        steps = set(self._steps)
        for name, s in NAMED_CHORDS.items():
            if steps == set(s):
                return self._base + name
        return None

    def invert(self, n: int) -> 'Chord':
        """Return the n-th chord inversion."""
        base = super().rotate(n)
        base.__class__ = self.__class__
        return base

    def sound(self, octave: int = 3, duration: float = 1.5):
        from .sounds import SineSound

        base = Note(f"{self._base}{octave}")
        chord = SineSound(base, duration=duration)
        for s in self._steps:
            chord &= SineSound(base + s, duration=duration)

        return chord

    def play(self, *args, **kwargs):
        from .sounds import SoundContext

        with SoundContext() as sc:
            sc.play(self.sound(*args, **kwargs))
