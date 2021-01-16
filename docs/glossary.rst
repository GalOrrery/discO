.. currentmodule:: discO

********
Glossary
********

.. glossary::

   frame-like
       a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance
       or a :class:`~astropy.coordinates.SkyCoord` (or subclass) instance or a
       string that can be converted to a Frame by
       :class:`~astropy.coordinates.sky_coordinate_parsers._get_frame_class`.

   coord-like
       a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance or a
       :class:`~astropy.coordinates.SkyCoord` (or subclass) instance

   coord-like scalar
   coord scalar
       a :term:`coord-like` object with length 1.

   coord-like array
   coord array
       a :term:`coord-like` object with length > 1.
