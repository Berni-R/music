from typing import Optional, Literal, Union, Callable
import numpy as np
import drawSvg as draw

from .notes import Note, Tuning, TUNING_STANDARD


FRET_WIDTH = 45
FRET_HEIGHT = 25
NOTE_SIZE = 0.9 * FRET_HEIGHT
NUT_SEP = 2


class Fret:

    def __init__(self, fret_cnt: int = 17, tuning: Union[Tuning, str] = TUNING_STANDARD):
        self.fret_cnt = fret_cnt
        self.tuning = tuning.copy() if isinstance(tuning, Tuning) else Tuning(tuning)
        assert len(self.tuning.strings) == 6

    def __repr__(self) -> str:
        return f"Fret({self.fret_cnt}, \"{self.tuning}\")"

    def locate_note(self, note: Union[Note, str]) -> list[int]:
        """Locate the frets on each string that correspond to this note."""
        if isinstance(note, str):
            note = Note(note)

        frets = []
        for s in self.tuning.strings:
            fret = note - s
            frets.append(fret if 0 <= fret <= self.fret_cnt + 1 else None)
        return frets

    def note_at(self, string: int, fret: int) -> Note:
        """Get the note on the given string and fret."""
        return self.tuning.strings[string] + fret

    def _get_def_drawing(self, margin: int = 0):
        w = FRET_WIDTH
        h = FRET_HEIGHT

        return draw.Drawing(margin + (1 + self.fret_cnt + 0.3) * w,
                        6.3 * h, origin=(-margin - 0.3 * w, -0.3 * h), displayInline=False)

    def draw_fret(self,
                  d: Optional[draw.Drawing] = None,
                  font_size: Optional[float] = None,
                  low_markers: bool = True,
                  fret_markers: bool = False,
    ) -> draw.Drawing:
        w = FRET_WIDTH
        h = FRET_HEIGHT
        ns2 = NUT_SEP
        font_size = font_size or h * 0.55
        margin = 1.2 * font_size

        if d is None:
            d = self._get_def_drawing(margin)

        marker_color = 'grey'
        for i, s in enumerate(self.tuning.strings):
            d.append(draw.Text(str(s), font_size, -font_size, (i + 0.5) * h - font_size / 3,
                               fill=marker_color))
        # markers
        if low_markers:
            for i in [3, 5, 9, 15, 17, 21]:
                marker = draw.Circle((i + 0.5) * w, 0, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
            for i in [7, 12, 19, 24]:
                marker = draw.Circle((i + 0.35) * w, 0, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.65) * w, 0, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
        if fret_markers:
            for i in [3, 5, 9, 15, 17, 21]:
                marker = draw.Circle((i + 0.5) * w, 3 * h, 0.2 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.5) * w, 0.12 * h, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
            for i in [7, 19]:
                marker = draw.Circle((i + 0.5) * w, 2.7 * h, 0.2 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.5) * w, 3.3 * h, 0.2 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.35) * w, 0.12 * h, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.65) * w, 0.12 * h, 0.1 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
            for i in [12, 24]:
                marker = draw.Circle((i + 0.5) * w, 2.2 * h, 0.2 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
                marker = draw.Circle((i + 0.5) * w, 3.8 * h, 0.2 * h, stroke_width=0, fill=marker_color)
                d.append(marker)
        # nut
        nut_1 = draw.Line(
            w - ns2, 0.5 * h,
            w - ns2, 5.5 * h,
            stroke="black",
            stroke_width=2,
        )
        d.append(nut_1)
        nut_2 = draw.Line(
            w + ns2, 0.5 * h,
            w + ns2, 5.5 * h,
            stroke="black",
            stroke_width=2,
        )
        d.append(nut_2)
        # draw frets
        for i in range(1, self.fret_cnt + 1):
            fret = draw.Line(
                (1 + i) * w, 0.5 * h,
                (1 + i) * w, 5.5 * h,
                stroke="black",
                stroke_width=4,
            )
            d.append(fret)
        # strings
        for i in range(6):
            string = draw.Line(
                w - ns2 - 1 if i in (0, 5) else w + ns2, (i + 0.5) * h,
                (1 + self.fret_cnt + 0.3) * w + ns2, (i + 0.5) * h,
                stroke="black",
                stroke_width=1.5,
            )
            d.append(string)

        return d

    def draw_note_at(self,
                     d: draw.Drawing,
                     string: int, fret: int,
                     color: str = 'blue',
                     shape: Literal['circle', 'rectangle', 'diamond'] = 'circle',
                     filled: bool = True,
                     size: Optional[float] = None,
    ) -> draw.Drawing:
        w = FRET_WIDTH
        h = FRET_HEIGHT
        size = size or 0.4 * NOTE_SIZE

        if string < 0 or 6 <= string:
            raise ValueError(f"String needs to be in [0, 5].")

        if filled:
            kwargs = {'stroke_width': 1, 'fill': color}
        else:
            kwargs = {'stroke_width': 3, 'stroke': color, 'fill': 'none'}
            size -= 2

        if shape == 'circle':
            note = draw.Circle((fret + 0.5) * w, (string + 0.5) * h, size, **kwargs)
        elif shape == 'rectangle':
            size *= np.pi / 4
            note = draw.Rectangle((fret + 0.5) * w - size, (string + 0.5) * h - size,
                                  2 * size, 2 * size, **kwargs)
        elif shape == 'diamond':
            #size *= np.pi / 4
            c = (fret + 0.5) * w, (string + 0.5) * h
            note = draw.Lines(
                c[0] - size, c[1],
                c[0], c[1] + size,
                c[0] + size, c[1],
                c[0], c[1] - size,
                close=True, **kwargs)
        else:
            raise ValueError(f"Unknown shape = '{shape}'")
        d.append(note)

        return d

    def draw_note(self,
                  d: draw.Drawing,
                  note: Union[Note, int],
                  **kwargs) -> draw.Drawing:
        """Draw a given note (at given octavae) on all possible locaations (i.e. string-fret combinations).
        
        For arguments, see `draw_note_at`."""
        for string, fret in enumerate(fret.locate_note(note)):
            if fret is not None:
                self.draw_note_at(d, string, fret, **kwargs)
        return d

    def draw_all(self, note_styler: Callable[[int, int, Note], Optional[dict]], **fret_args) -> draw.Drawing:
        """Draw the fret will all notes, styled by `note_styler` (which can also not draw a note)."""
        d = self.draw_fret(**fret_args)
        for s in range(6):
            for f in range(self.fret_cnt + 1):
                note = self.note_at(s, f)

                st = note_styler(s, f, note)

                if st is None:
                    continue

                self.draw_note_at(d, s, f, **st)
        return d
