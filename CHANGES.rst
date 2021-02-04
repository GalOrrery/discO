================
0.1 (2020-11-17)
================

Adopt Project-template from @nstarman

New Features
------------

discO.type_hints
^^^^^^^^^^^^^^^^

- Add types for type hinting: [#17]

    + EllipsisType : ``type(Ellipsis)`` b/c ellipsis fails.
    + UnitType : the type of Astropy's ``UnitBase`` and ``FunctionUnitBasse``
    + RepresentationOrDifferentialType : the type of Astropy's ``BaseRepresentationOrDifferential`` [#34]
    + RepresentationType : the type of Astropy's ``BaseRepresentation`` [#34]
    + DifferentialType : the type of Astropy's ``BaseDifferential`` [#34]
    + FrameType : the type of Astropy's ``BaseCoordinateFrame``
    + SkyCoordType : the type of Astropy's ``SkyCoord``
    + CoordinateType : the union of FrameType & SkyCoordType
    + GenericPosiionType : RepresentationOrDifferentialType or CoordinateType [#34]
    + FrameLikeType : the union of CoordinateType & parseable str

        * anything that can be used in ``frame=`` in  ``Skycoord(...,frame=)``
    + TableType : the type of Astropy's ``Table`` [#34]
    + QTableType : the type of Astropy's ``QTable`` [#34]
    + UnitType : the type of Astropy's ``UnitBase`` or ``FunctionUnitType`` [#34]
    + UnitLikeType : UnitType or parseable str [#34]
    + QuantityType : the type of astropy's ``Quantity``
    + QuantityType : QuantityType or parseable str [#34]

- changed location from ``common`` to ``type_hints`` [#34]

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
- ``fitter`` : for fitting a Potential given a sample [#20]
- ``pipeline`` : for comboing the analysis [#19]

**discO.core.core**

The base class samplers, fitters, and most everything else.
If a ``package`` is defined as a class argument, it parses the package and
stores it in the class. This is used for registering classes into registry
classes.

subclasses must override the ``_registry`` and ``__call__`` methods.


**discO.core.sample**

``PotentialSampler`` : base class for sampling potentials [#17]

    + registers subclasses. Each subclass is for sampling from potentials from
      a different package. Eg. ``GalpyPotentialSampler`` for sampling
      ``galpy`` potentials.
    + PotentialSampler can be used to initialize & wrap any of its subclasses.
      This is controlled by the argument ``return_specific_class``. If False,
      it returns the subclass itself.
    + Takes a ``potential`` and a ``frame`` (astropy CoordinateFrame). The
      potential is used for sampling, but the resulting points are not located
      in any reference frame, which we assign with ``frame``.
    + ``__call__`` and ``sample`` are used to sample the potential
    + ``sample`` samples the potential many times. This
      can be done for many iterations and different sample number points.
    + ``sample_iter`` samples the potential many times as a generator.


**discO.core.measurement**

- ``MeasurementErrorSampler`` : base class for resampling a potential given
  measurement errors [#17]

    + registers subclasses. Each subclass is for resampling in a different
      way.
    + ``MeasurementErrorSampler`` is a registry wrapper class and can be used
      in-place of any of its subclasses.

- ``GaussianMeasurementErrorSampler`` : uncorrelated Gaussian errors [#17]


**discO.core.fitter**

- ``PotentialFitter`` : base class for fitting potentials [#20]

    + registers subclasses.
    + PotentialFitter can be used to initialize & wrap any of its subclasses.
      This is controlled by the argument ``return_specific_class``. If False,
      it returns the subclass itself.
    + Takes a ``potential_cls`` and ``key`` argument which are used to figure
      out the desired subclass, and how to fit the potential.
    + ``__call__`` and ``fit`` are used to fit the potential, with the latter
      working on N-D samples (multiple iterations).


**discO.core.pipeline**

- ``Pipeline`` : run a full analysis pipeline [#19]

    + ``PotentialSampler`` to ``MeasurementErrorSampler`` to
      ``PotentialFitter`` to ``ResidualMethod`` to ``statistic``.
    + Pipeines can also be created by concatenation.


discO.data
^^^^^^^^^^

- Add Milky_Way_Sim_100 data [#10]


discO.plugin
^^^^^^^^^^^^

Where classes for external packages are held.


discO.plugin.agama
^^^^^^^^^^^^^^^^^^

- AGAMAPotentialSampler [#17]

    + Sample from ``agama`` potentials.
    + Subclass of ``PotentialSampler``
    + stores the mass and potential as attributes on the returned ``SkyCoord``

- AGAMAPotentialFitter [#20]

    + Fit ``agama`` potentials.
    + Subclass of ``PotentialFitter``
    + registers subclasses for different fit methods.
    + AGAMAPotentialFitter can be used to initialize & wrap any of its
      subclasses. This is controlled by the argument ``return_specific_class``. If False, it returns the subclass itself.
    + Takes a ``pot_type`` argument which is used to figure
      out the desired subclass, and how to fit the potential.

- AGAMAMultipolePotentialFitter [#20]

    + Fit ``agama`` potentials with a multipole
    + Subclass of ``AGAMAPotentialFitter``


discO.plugin.galpy
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
