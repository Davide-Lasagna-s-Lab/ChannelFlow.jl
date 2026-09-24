export @loop_jk_i

#//////////////////////////////////////////////////////////////////////////////#
#///               FOURIER MODE AND CHEBYSHEV INDEX ITERATION               ///#
#//////////////////////////////////////////////////////////////////////////////#

"""
    @loop_jk_i size(U) expression

Traverse `(k, l, n)` storage with contiguous Fourier systems innermost. In the
expression, `_i`, `_j`, `_k` are one-based array indices; `k` and `l` are
integer streamwise and signed spanwise Fourier modes. Physical derivatives
must still multiply by `2π/Lx` or `2π/Lz`. The loop uses `@inbounds`, so all
arrays accessed by the expression must have compatible dimensions.
"""
macro loop_jk_i(SIZE, expr)
    quote
        Nxh, Nz, Ny = $(esc(SIZE))
        @inbounds for $(esc(:_i)) = 1:Ny, $(esc(:_k)) = 1:Nz
            $(esc(:l)) = $(esc(:_k)) <= (Nz >> 1) + 1 ?
                        $(esc(:_k)) - 1 : $(esc(:_k)) - 1 - Nz
            for $(esc(:_j)) = 1:Nxh
                $(esc(:k)) = $(esc(:_j)) - 1
                $(esc(expr))
            end
        end
    end
end
