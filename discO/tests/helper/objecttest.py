# -*- coding: utf-8 -*-

"""Baseclass for tests on an object."""

__all__ = [
    "ObjectTest",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect

##############################################################################
# CODE
##############################################################################


class ObjectTest:
    """Base class for tests which rely on a class.

    Subclasses must specify a class in the class definition.

        >>> class SubClass(ObjectTest, obj=object):
        ...     '''A SubClass with class=object.'''
        ...     pass
        >>> SubClass.obj == object
        True

    """

    def __init_subclass__(cls, obj: object, **kwargs):
        """Initialize subclass.

        Parameters
        ----------
        obj : object
            The class that will be tested.
        **kwargs
            Arguments into subclass initialization.

        """
        super().__init_subclass__()
        cls.obj = obj

        # -------------------
        # format doc, if there is None

        if cls.__doc__ is None and cls.obj is not None:
            if inspect.isclass(cls.obj):  # class vs instance
                clsname = cls.obj.__name__.strip()
            else:
                clsname = cls.obj.__class__.__name__.strip()

            path = str(cls.obj.__module__).strip()

            cls.__doc__ = rf"Test ::`~{path}.{clsname}`."

    # /def


# /class


##############################################################################
# END
