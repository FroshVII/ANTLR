import torch

class ANTLR(torch.autograd.Function):
    """
    Modularized version of the ANTLR gradient function, without multi-
    model support.
    """

    # ======= ======= =======
    # Forward Pass
    # ======= ======= ====

    @staticmethod
    def forward(ctx, inp, act_scalar=1, tim_scalar=1):
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.

        act_scalar is λ_{act} from Kim et al, tim_scalar is λ_{tim}.
        """
        ctx.save_for_backward(inp)
        raise NotImplementedError("forward pass for ANTLR not implemented")

    # ======= ======= =======
    # Backward Pass
    # ======= ======= ====

    @staticmethod
    def _act_grad(ctx, out):
        """
        Activation portion of the ANTLR gradient formula, defined in
        Eq. (9) of Kim et al, 2020.
        """
        raise NotImplementedError("_act_grad not implemented")

    @staticmethod
    def _tim_grad(ctx, out):
        """
        Timing portion of the ANTLR gradient formula, defined in Eq.
        (10) of Kim et al, 2020.
        """
        raise NotImplementedError("_tim_grad not implemented")

    @staticmethod
    def backward(ctx, out):
        """
        ANTLR gradient formula, defined in Eq. (8) of Kim et al, 2020
        """
        (inp,) = ctx.saved_tensors
        raise NotImplementedError("backward pass for ANTLR not implemented")
