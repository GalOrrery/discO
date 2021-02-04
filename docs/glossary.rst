.. currentmodule:: discO

********
Glossary
********

.. glossary::


   (`n`,)
       A parenthesized number followed by a comma denotes a tuple with one
       element. The trailing comma distinguishes a one-element tuple from a
       parenthesized ``n``.


   -1
       - **In a dimension entry**, instructs NumPy to choose the length
         that will keep the total number of array elements the same.

           >>> np.arange(12).reshape(4, -1).shape
           (4, 3)

       - **In an index**, any negative value
         `denotes <https://docs.python.org/dev/faq/programming.html#what-s-a-negative-index>`_
         indexing from the right.

   . . .
       An :py:data:`Ellipsis`.

       - **When indexing an array**, shorthand that the missing axes, if they
         exist, are full slices.

           >>> a = np.arange(24).reshape(2,3,4)

           >>> a[...].shape
           (2, 3, 4)

           >>> a[...,0].shape
           (2, 3)

           >>> a[0,...].shape
           (3, 4)

           >>> a[0,...,0].shape
           (3,)

         It can be used at most once; ``a[...,0,...]`` raises an :exc:`IndexError`.

       - **In printouts**, NumPy substitutes ``...`` for the middle elements of
         large arrays. To see the entire array, use ``numpy.printoptions``


   :
       The Python :term:`python:slice`
       operator. In ndarrays, slicing can be applied to every
       axis:

           >>> a = np.arange(24).reshape(2,3,4)
           >>> a
           array([[[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]],
           <BLANKLINE>
                  [[12, 13, 14, 15],
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]])
           <BLANKLINE>
           >>> a[1:,-2:,:-1]
           array([[[16, 17, 18],
                   [20, 21, 22]]])

       Trailing slices can be omitted: ::

           >>> a[1] == a[1,:,:]
           array([[ True,  True,  True,  True],
                  [ True,  True,  True,  True],
                  [ True,  True,  True,  True]])

       In contrast to Python, where slicing creates a copy, in NumPy slicing
       creates a :term:`view`.

       For details, see :ref:`combining-advanced-and-basic-indexing`.


   along an axis
       An operation "along axis n" of array ``a`` behaves as if its argument
       were an array of slices of ``a`` where each slice has a successive
       index of axis "n".

       For example, if ``a`` is a 3 x ``N`` array, an operation along axis 0
       behaves as if its argument were an array containing slices of each row:

           >>> np.array((a[0,:], a[1,:], a[2,:])) #doctest: +SKIP

       To make it concrete, we can pick the operation to be the array-reversal
       function :func:`numpy.flip`, which accepts an ``axis`` argument. We
       construct a 3 x 4 array ``a``:

           >>> a = np.arange(12).reshape(3,4)
           >>> a
           array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])

       Reversing along axis 0 (the row axis) yields

           >>> np.flip(a,axis=0)
           array([[ 8,  9, 10, 11],
                  [ 4,  5,  6,  7],
                  [ 0,  1,  2,  3]])

       Recalling the definition of "along an axis",  ``flip`` along axis 0 is
       treating its argument as if it were

           >>> np.array((a[0,:], a[1,:], a[2,:]))
           array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])

       and the result of ``np.flip(a,axis=0)`` is to reverse the slices:

           >>> np.array((a[2,:],a[1,:],a[0,:]))
           array([[ 8,  9, 10, 11],
                  [ 4,  5,  6,  7],
                  [ 0,  1,  2,  3]])


   array
       Used synonymously in the NumPy docs with :term:`ndarray`.


   array-like
   array_like
       Any :doc:`scalar <reference/arrays.scalars>` or
       :term:`python:sequence`
       that can be interpreted as an ndarray.  In addition to ndarrays
       and scalars this category includes lists (possibly nested and with
       different element types) and tuples. Any argument accepted by
       :doc:`numpy.array <reference/generated/numpy.array>`
       is array_like.
       ::

           >>> a = np.array([[1, 2.0], [0, 0], (1+1j, 3.)])

           >>> a
           array([[1.+0.j, 2.+0.j],
                  [0.+0.j, 0.+0.j],
                  [1.+1.j, 3.+0.j]])


   array scalar
       For uniformity in handling operands, NumPy treats
       a :doc:`scalar <reference/arrays.scalars>` as an array of zero
       dimension.


   axis
       Another term for an array dimension. Axes are numbered left to right;
       axis 0 is the first element in the shape tuple.

       In a two-dimensional vector, the elements of axis 0 are rows and the
       elements of axis 1 are columns.

       In higher dimensions, the picture changes. NumPy prints
       higher-dimensional vectors as replications of row-by-column building
       blocks, as in this three-dimensional vector:

           >>> a = np.arange(12).reshape(2,2,3)
           >>> a
           array([[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[ 6,  7,  8],
                   [ 9, 10, 11]]])

       ``a`` is depicted as a two-element array whose elements are 2x3 vectors.
       From this point of view, rows and columns are the final two axes,
       respectively, in any shape.

       This rule helps you anticipate how a vector will be printed, and
       conversely how to find the index of any of the printed elements. For
       instance, in the example, the last two values of 8's index must be 0 and
       2. Since 8 appears in the second of the two 2x3's, the first index must
       be 1:

           >>> a[1,0,2]
           8

       A convenient way to count dimensions in a printed vector is to
       count ``[`` symbols after the open-parenthesis. This is
       useful in distinguishing, say, a (1,2,3) shape from a (2,3) shape:

           >>> a = np.arange(6).reshape(2,3)
           >>> a.ndim
           2
           >>> a
           array([[0, 1, 2],
                  [3, 4, 5]])

           >>> a = np.arange(6).reshape(1,2,3)
           >>> a.ndim
           3
           >>> a
           array([[[0, 1, 2],
                   [3, 4, 5]]])


   copy
       See :term:`view`.


   coord-like
   coord_like
       a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance or a
       :class:`~astropy.coordinates.SkyCoord` (or subclass) instance


   dimension
       See :term:`axis`.


   dtype
       The datatype describing the (identically typed) elements in an ndarray.
       It can be changed to reinterpret the array contents. For details, see
       :doc:`Data type objects (dtype). <reference/arrays.dtypes>`


   frame-like
   frame_like
       a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance
       or a :class:`~astropy.coordinates.SkyCoord` (or subclass) instance or a
       string that can be converted to a Frame by
       :class:`~astropy.coordinates.sky_coordinate_parsers._get_frame_class`.


   ndarray
      :doc:`NumPy's basic structure <reference/arrays>`.


   number
       Any of :py:data:`int`, :py:data:`float` or numpy equivalent.


   optional
       This argument has a default value. See the signature and/or documentation
       for details.

   scalar
       In NumPy, usually a synonym for :term:`array scalar`.


   shape
       A tuple showing the length of each dimension of an ndarray. The
       length of the tuple itself is the number of dimensions
       (:doc:`numpy.ndim <reference/generated/numpy.ndarray.ndim>`).
       The product of the tuple elements is the number of elements in the
       array. For details, see
       :doc:`numpy.ndarray.shape <reference/generated/numpy.ndarray.shape>`.

   unit-like
   unit_like
       A :class:`~astropy.units.UnitBase` or
       :class:`~astropy.units.FunctionUnitBase` or any :py:data:`str` that can be
       parsed into a Unit object.

   view
       Without touching underlying data, NumPy can make one array appear
       to change its datatype and shape.

       An array created this way is a "view", and NumPy often exploits the
       performance gain of using a view versus making a new array.

       A potential drawback is that writing to a view can alter the original
       as well. If this is a problem, NumPy instead needs to create a
       physically distinct array -- a "copy".

       Some NumPy routines always return views, some always return copies, some
       may return one or the other, and for some the choice can be specified.
       Responsibility for managing views and copies falls to the programmer.
       :func:`numpy.shares_memory` will check whether ``b`` is a view of
       ``a``, but an exact answer isn't always feasible, as the documentation
       page explains.

         >>> x = np.arange(5)
         >>> x
         array([0, 1, 2, 3, 4])

         >>> y = x[::2]
         >>> y
         array([0, 2, 4])

         >>> x[0] = 3 # changing x changes y as well, since y is a view on x
         >>> y
         array([3, 2, 4])

   quantity-like
   quantity_like
       A :class:`~astropy.units.Quantity` or any :py:data:`str` that can be parsed into
       a Quantity.
