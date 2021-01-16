================
0.1 (2020-11-17)
================

Adopt Project-template from @nstarman

New Features
------------

discO.common
^^^^^^^^^^^^

Code that can be used all over the package. It's not core, it's common.

- Add types for type hinting: [#17]

    + EllipsisType : ``type(Ellipsis)`` b/c ellipsis fails.
    + UnitType : the type of astropy's ``UnitBase`` and ``FunctionUnitBasse``
    + QuantityType : the type of astropy's ``Quantity``
    + FrameType : the type of astropy's ``BaseCoordinateFrame``
    + SkyCoordType : the type of astropy's ``SkyCoord``
    + CoordinateType : the union of FrameType & SkyCoordType
    + FrameLikeType : the union of CoordinateType & str

        * anything that can be used in ``frame=`` in  ``Skycoord(...,frame=)``


discO.config
^^^^^^^^^^^^

- Add configuration for default frame. Defaults to "icrs". [#17]


discO.core
^^^^^^^^^^

Where the unified architecture is defined.

Modules:

- ``core`` : the base class. [#17]
- ``sample`` : for sampling from a Potential. [#17]
- ``measurement`` : for resampling, given observational errors. [#17]


**discO.core.core**

The base class samplers, fitters, and most everything else.
If a ``package`` is defined as a class argument, it parses the package and
stores it in the class. This is used for registering classes into registry
classes.

subclasses must override the ``_registry`` and ``__call__`` methods.


**discO.core.sample**

PotentialSampler : base class for sampling potentials [#17]

    + registers subclasses. Each subclass is for sampling from potentials from
      a different package. Eg. ``GalpyPotentialSampler`` for sampling ``galpy``
      potentials.
    + PotentialSampler can be used to initialize & wrap any of its subclasses.
      This is controlled by the argument ``return_specific_class``. If False,
      it returns the subclass itself.
    + Takes a ``potential`` and a ``frame`` (astropy CoordinateFrame). The
      potential is used for sampling, but the resultant points are not located
      in any reference frame, which we assign with ``frame``.
    + ``__call__`` and ``sample`` are used to sample the potential
    + ``resample`` (and ``resampler``) sample the potential many times. This can
      be done for many iterations and different sample number points.


**discO.core.measurement**

- MeasurementErrorSampler : abstract base class for resampling a potential given measurement errors [#17]

    + registers subclasses. Each subclass is for resampling in a different way.
    + MeasurementErrorSampler can be used to wrap any of its subclasses.

- GaussianMeasurementErrorSampler : apply uncorrelated Gaussian errors [#17]


discO.data
^^^^^^^^^^

- Add Milky_Way_Sim_100 data [#10]


discO.extern
^^^^^^^^^^^^

Where classes for external packages are held.


discO.extern.agama
^^^^^^^^^^^^^^^^^^

- AGAMAPotentialSampler [#17]

    + Sample from ``agama`` potentials.
    + stores the mass and potential as attributes on the returned ``SkyCoord``


discO.extern.galpy
^^^^^^^^^^^^^^^^^^

- GalpyPotentialSampler [#17]

    + Sample from ``galpy`` potentials with a corresponding distribution function.
    + stores the mass and potential as attributes on the returned ``SkyCoord``


discO.utils
^^^^^^^^^^^

- resolve_framelike [#17]

    Determine the frame and return a blank instance for anything that can be
    used in ``frame=`` in  ``Skycoord(...,frame=)``


API Changes
-----------

N/A


Bug Fixes
---------

N/A


Docs
----

- Added glossary [#17]

    + 'frame-like'
    + 'coord-like'
    + 'coord scalar' and 'coord-like scalar'
    + 'coord array' and 'coord-like array'


Other Changes and Additions
---------------------------

- Alphabetize name in credits [#8]

- PR Template [#5]

    + Updated [#11]

- Use GitHub for CI [#12]

    + On tag [#17]

- Dependabot yml [#13]

- Issues Templates [#14]

- Update from project template [#18]

- Add ``.mailmap`` [#17]


Actions
^^^^^^^

- PR labeler [#18]

- Pre-commit [#18]

    - `isort <https://pypi.org/project/isort/>`_
    - `black <https://pypi.org/project/black/>`_
    - `flake8 <https://pypi.org/project/flake8/>`_
    - many others from `precommit <https://pre-commit.com/hooks.html>`__ [#17]
