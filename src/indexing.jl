export @loop_jk_i

#//////////////////////////////////////////////////////////////////////////////#
#///               FOURIER MODE AND CHEBYSHEV INDEX ITERATION               ///#
#//////////////////////////////////////////////////////////////////////////////#

macro loop_jk_i(SIZE, expr)
    quote
        Ny, Nxh, Nz = $(esc(SIZE))
        @inbounds for $(esc(:_k)) = 1:Nz
            $(esc(:k)) = $(esc(:_k)) <= (Nz >> 1) + 1 ?
                        $(esc(:_k)) - 1 : $(esc(:_k)) - 1 - Nz
            for $(esc(:_j)) = 1:Nxh
                $(esc(:j)) = $(esc(:_j)) - 1
                for $(esc(:_i)) = 1:Ny
                    $(esc(expr))
                end
            end
        end
    end
end
