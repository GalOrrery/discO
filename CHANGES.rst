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
    + RepresentationLikeType : RepresentationType or str [#42]
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

- Add configuration for default frame ("icrs"). [#17]
- Add configuration for default representation type ("cartesian") [#45]


discO.core
^^^^^^^^^^

Where the unified architecture is defined.

Modules:

- ``core`` : the base class. [#17]
- ``sample`` : for sampling from a Potential. [#17]
- ``measurement`` : for resampling, given observational errors. [#17]
- ``fitter`` : for fitting a Potential given a sample [#20]
- ``pipeline`` : for comboing the analysis [#19]
- ``wrapper`` : for wrapping potentials [#45]

**discO.core.core**

The base class samplers, fitters, and most everything else.
If a ``package`` is defined as a class argument, it parses the package and
stores it in the class. This is used for registering classes into registry
classes.

subclasses must override the ``_registry`` and ``__call__`` methods.

**discO.core.core**

- ``CommonBase`` : base class

    + provides tools for working with class registries


**discO.core.sample**

- ``PotentialSampler`` : base class for sampling potentials [#17]

    + registers subclasses. Each subclass is for sampling from potentials from
      a different package. Eg. ``GalpyPotentialSampler`` for sampling
      ``galpy`` potentials.
    + PotentialSampler can be used to initialize any of its subclasses.
    + Takes a ``potential`` and a ``frame`` (astropy CoordinateFrame). The
      potential is used for sampling, but the resulting points are not located
      in any reference frame, which we assign with ``frame``.
    + Can also specify representation type [#43]
    + ``__call__`` and ``sample`` are used to sample the potential
    + ``sample`` samples the potential many times. This
      can be done for many iterations and different sample number points.
    + ``frame`` and ``representation_type`` can be None or Ellipse or anything
      that works with ``resolve_framelike``. [#45]


**discO.core.fitter**

- ``PotentialFitter`` : base class for fitting potentials [#20]

    + registers subclasses.
    + PotentialFitter can be used to initialize any of its subclasses. [#44]
    + Takes a ``potential_cls`` and ``key`` argument which are used to figure
      out the desired subclass, and how to fit the potential.
    + ``__call__`` and ``fit`` are used to fit the potential, with the latter
      working on N-D samples (multiple iterations).
    + returns a ``PotentialWrapper`` [#40]
    + Allow for ``frame`` and ``representation``. Care should be taken this
      matches the sampling frame. [#45]
    + ``frame`` and ``representation_type`` can be None or Ellipse or anything
      that works with ``resolve_framelike``. [#45]


**discO.core.measurement**

- ``MeasurementErrorSampler`` : base class for resampling a potential given
  measurement errors [#17]

    + registers subclasses. Each subclass is for resampling in a different
      way.
    + ``MeasurementErrorSampler`` is a registry wrapper class and can be used
      in-place of any of its subclasses.
    + Add method ``resample`` for ND array samples from ``PotentialSampler`` [#38]
    + ``frame`` and ``representation_type`` can be None or Ellipse or anything
      that works with ``resolve_framelike``. [#45]
    + ``c_err`` must be a keyword argument. [#45]

- ``RVS_Continuous`` : scipy rv_continuous distribution [#42]

  + Any scipy rv_continuous distribution.
  + ``rvs`` must be a keyword argument. [#45]

- ``GaussianMeasurementError`` : Gaussian rvs distribution [#42]

  + should work for any normal distribution (if has "norm") in name.

- ``xpercenterror_factory`` : to build ``xpercenterror`` function. [#36]
  Convenience function for construct errors with X% error in each dimension.


**discO.core.pipeline**

- ``Pipeline`` : run a full analysis pipeline [#19]

    + ``PotentialSampler`` to ``MeasurementErrorSampler`` to
      ``PotentialFitter`` to ``ResidualMethod`` to ``statistic``. [#19,#26]
    + Pipelines can also be created by concatenation.
    + Pipeline can take arguments ``frame`` and ``representation_type``. [#45]
    + Calling pipeline can take arguments observer versions of ``frame`` and
      ``representation_type``. [#45]
    + ``frame`` and ``representation_type`` can be None or Ellipse or anything
      that works with ``resolve_framelike``. [#45]
    + convenience properties for ``potential``, ``frame``,
      ``representation_type``, ``potential_frame``,
      ``potential_representation_type``, ``observer_frame``,
      ``observer_representation_type``, ``sampler``, ``measurer``, ``fitter``,
      ``residualer``, ``statisticer``. [#45]
    + Add method ``run_iter`` to iteratively call pipeline. [#26]

- ``PipelineResult`` store results of a pipe [#37]

    + produced by ``Pipeline`` at end of a ``run`` or call.
    + convenience properties for ``samples``, ``potential_frame``,
      ``potential_representation_type``, ``measured``, ``observation_frame``,
      ``observation_representation_type``, ``fit``, ``residual``,
      ``statistic``. [#45]

**discO.core.residual**

- ``ResidualMethod`` : calculate a residual [#26]

  + difference between original and fit potential

- ``GridResidual`` : calculate a residual on a pre-defined grid [#26]

  + difference between original and fit potential
  + need pre-defined grid


**discO.core.wrapper**

- ``PotentialWrapper`` : base class for wrapping Potentials [#39]

    + unified interface for the specific potential and specific force.
    + all methods are both instance and static methods.
    + specific force returns a vector field.
    + ``frame`` and ``representation_type`` can be None or Ellipse or anything
      that works with ``resolve_framelike``. [#45]
    + ``total_mass`` function. [#45]


discO.data
^^^^^^^^^^

- Add Milky_Way_Sim_100 data [#10]


discO.plugin
^^^^^^^^^^^^

Where classes for external packages are held.


discO.plugin.agama
^^^^^^^^^^^^^^^^^^

- ``AGAMAPotentialSampler`` [#17]

    + Sample from ``agama`` potentials.
    + Subclass of ``PotentialSampler``
    + stores the mass and potential as attributes on the returned ``SkyCoord``

- ``AGAMAPotentialFitter`` [#20]

    + Fit ``agama`` potentials.
    + Subclass of ``PotentialFitter``
    + registers subclasses for different fit methods.
    + AGAMAPotentialFitter can be used to initialize any of its subclasses.
    + Takes a ``pot_type`` argument which is used to figure
      out the desired subclass, and how to fit the potential.
    + returns a ``AGAMAPotentialWrapper`` [#40]

- ``AGAMAMultipolePotentialFitter`` [#20]

    + Fit ``agama`` potentials with a multipole
    + Subclass of ``AGAMAPotentialFitter``

- ``AGAMAPotentialWrapper`` : for wrapping Potentials [#39]

    + unified interface for the specific potential and specific force.
    + all methods are both instance and static methods.
    + specific force returns a vector field.
    + ``total_mass`` function. [#45]


discO.plugin.galpy
^^^^^^^^^^^^^^^^^^

- ``GalpyPotentialSampler`` [#17]

    + Sample from ``galpy`` potentials with a corresponding distribution function.
    + stores the mass and potential as attributes on the returned ``SkyCoord``

- ``GalpyPotentialWrapper`` : for wrapping Potentials [#39]

    + unified interface for the specific potential and specific force.
    + all methods are both instance and static methods.
    + specific force returns a vector field.
    + ``total_mass`` function. [#45]

- ``GalpySCFPotentialFitter`` : for fitting an SCF to particles [#41]

    + fit galpy SCF potential
    + returns a ``GalpyPotentialWrapper`` with the specified frame.

discO.utils
^^^^^^^^^^^

- ``resolve_framelike`` [#17]

    + Determine the frame and return a blank instance for anything that can be
      used in ``frame=`` in  ``Skycoord(...,frame=)``.
    + Ellipsis resolves to the configured default frame ("icrs"). [#45]
    + None becomes ``UnFrame()`` [#45]

- ``resolve_representationlike`` [#42]

    + Determine the representation type given a class, instance, or string name.
    + Ellipsis uses default representation type ("cartesian") [#45]

- ``UnFrame`` : unconnected generic coordinate frame [#43]

  + For use when no reference frame is specified.

- vector fields [#35]

    For transforming vector fields between coordinate systems (eg Cartesian to spherical).
    Built on top of Astropy's Representation machinery.

- ``NumpyRNGContext`` : astropy's, extended to ``RandomState`` s [#43]


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
