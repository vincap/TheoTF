"""Exports classes to describe the nature of optimization parameters.

The ptypes module provides a handful of classes used to specify
optimization parameters. Different nature parameters require appropriate
treatment: ptypes' classes tell the optimizers how to deal with each
given parameter.

  Usage example:

  categorical_param = ptypes.Categorical(('Val1', 'Val2', 'Val3'))
  continuous_param = ptypes.CRange(0.0, 1.0)
  discrete_param = ptypes.DRange(32, 256)
"""

class Categorical(tuple) :
    """A parameter taking values in a finite set of elements.
    
    A Categorical object is an immutable sequence (tuple) containing
    all possible values that the corresponding parameter can take.
    """
    
    def __init__(self, values) :
        """Builds a Categorical object.
        
        Builds a Categorical object using values contained in the
        provided Iterable argument (typically a tuple or a list).
        
        Args:
            values (Iterable): an Iterable object (like list or tuple)
            providing values for this Categorical object.
        """
        
        self = tuple(values)

class Range :
    """(Abstract) A parameter taking values within a range.
    
    Abstract class representing parameters that can take values within
    a discrete or continuous range of values.
    """
    
    def __init__(self, lower=None, upper=None) :
        raise NotImplementedError("Cannot create an instance of abstract class Range")

    def lower(self) :
        return self._lower

    def upper(self) :
        return self._upper
    
    def __str__(self) :
        return f'({self._lower}, {self._upper})'

class DRange(Range) :
    """A parameter which can take values within a discrete range.
    
    DRange (Discrete Range) represents a parameter that can take
    arbitrary values in a finite and discrete range of possibilities.
    Range is specified through its lower and upper bounds. For a
    DRange, these are both integer values.
    """
    
    def __init__(self, lower, upper) :
        """Builds a discrete range object with the provided bounds.
        
        Args:
            lower (int): lower bound of this range.
            
            upper (int): upper bound of this range (upper>lower).
            
        Raises:
            ValueError: if lower is greater than upper.
        """
        
        if lower > upper :
            raise ValueError('lower cannot be greater than upper.')
            
        self._lower = int(lower)
        self._upper = int(upper)

class CRange(Range) :
    """A parameter which can take values within a continuous range.
    
    CRange (Continuous Range) represents a parameter that can take
    arbitrary values in a continuous range of values. Range is
    specified through its lower and upper bounds. For a CRange, these
    are both floating-point values.
    """
    
    def __init__(self, lower, upper) :
        """Builds a continuous range object with the provided bounds.
        
        Args:
            lower (float): lower bound of this range.
            
            upper (float): upper bound of this range (upper>lower).
            
        Raises:
            ValueError: if lower is greater than upper.
        """
        
        if lower > upper :
            raise ValueError('lower cannot be greater than upper.')
            
        self._lower = lower
        self._upper = upper