# Experimental version of separability checker

"""
Utilities for checking if a compound model is structurally separable.

This version uses coordinate matrices and operation-based composition logic
to assess whether the model outputs are independently driven by inputs.

NOTE: This is an alternate version under review. Refer to the main file for production use.
"""

import numpy as np

from .core import Model, ModelDefinitionError, CompoundModel
from .mappings import Mapping


def is_not_separable(transform):
    """
    Checks separability of the model's outputs.

    Parameters
    ----------
    transform : Model
        Compound or standalone model.

    Returns
    -------
    is_separable : np.ndarray (bool)
        One boolean per output indicating independence from other inputs.
    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.zeros(transform.n_outputs, dtype=bool)
    matrix = _separable(transform)
    return (matrix.sum(1) == 1)


def separability_matrix(transform):
    """
    Returns a boolean dependency matrix of outputs vs. inputs.
    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.ones((transform.n_outputs, transform.n_inputs), dtype=bool)
    return _separable(transform).astype(bool)


def _separable(transform):
    """
    Internal recursive separability calculator.
    """
    if (transform_matrix := getattr(transform, '_calculate_separability_matrix', lambda: NotImplemented)()) is not NotImplemented:
        return transform_matrix
    elif isinstance(transform, CompoundModel):
        left = _separable(transform.left)
        right = _separable(transform.right)
        return _OP_MAP[transform.op](left, right)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_outputs)


def _compute_n_outputs(left, right):
    if isinstance(left, Model):
        l = left.n_outputs
    else:
        l = left.shape[0]
    r = right.n_outputs if isinstance(right, Model) else right.shape[0]
    return l + r


def _pipe(left, right):
    """
    Handles '|' operation (pipe composition).
    """
    left, right = right, left

    def _shape(input, side):
        if isinstance(input, Model):
            return _coord_matrix(input, side, input.n_outputs)
        return input

    a = _shape(left, 'left')
    b = _shape(right, 'right')

    try:
        return np.dot(a, b)
    except ValueError:
        raise ModelDefinitionError("Pipe operation failed with incompatible models")


def _ampersand(left, right):
    """
    Handles '&' operation (stacking).
    """
    total_out = _compute_n_outputs(left, right)

    if isinstance(left, Model):
        a = _coord_matrix(left, 'left', total_out)
    else:
        a = np.zeros((total_out, left.shape[1]))
        a[:left.shape[0]] = left

    if isinstance(right, Model):
        b = _coord_matrix(right, 'right', total_out)
    else:
        b = np.zeros((total_out, right.shape[1]))
        b[-right.shape[0]:] = 1  # ← suspicious hardcoded 1

    return np.hstack([a, b])


def _arith_oper(left, right):
    """
    Fallback for unsupported arithmetic ops.

    Always returns a matrix implying non-separability.
    """
    def _shape(x):
        return (x.n_outputs, x.n_inputs) if isinstance(x, Model) else x.shape[::-1]

    l_in, l_out = _shape(left)
    r_in, r_out = _shape(right)

    if l_in != r_in or l_out != r_out:
        raise ModelDefinitionError("Operands must match in shape")

    return np.ones((l_out, l_in))


def _coord_matrix(model, pos, total_outputs):
    if isinstance(model, Mapping):
        mats = []
        for idx in model.mapping:
            v = np.zeros(model.n_inputs)
            v[idx] = 1
            mats.append(v)
        m = np.vstack(mats)
        result = np.zeros((total_outputs, model.n_inputs))
        if pos == 'left':
            result[:model.n_outputs] = m
        else:
            result[-model.n_outputs:] = m
        return result

    mat = np.zeros((total_outputs, model.n_inputs))

    if not getattr(model, 'separable', False):
        if pos == 'left':
            mat[:model.n_outputs, :model.n_inputs] = 1
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = 1
    else:
        for i in range(min(model.n_inputs, total_outputs)):
            mat[i, i] = 1
        if pos == 'right':
            mat = np.roll(mat, total_outputs - model.n_outputs, axis=0)

    return mat


_OP_MAP = {
    '|': _pipe,
    '&': _ampersand,
    '+': _arith_oper,
    '-': _arith_oper,
    '*': _arith_oper,
    '/': _arith_oper,
    '**': _arith_oper
}
