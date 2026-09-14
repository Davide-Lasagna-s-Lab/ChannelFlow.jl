export @loop_jk_i

macro loop_jk_i(SIZE, expr)
    quote
        Ny, Nx, Nzh = $(esc(SIZE))
        @inbounds for $(esc(:k)) = 0:Nzh-1
            $(esc(:_k)) = 1 + $(esc(:k))
            for $(esc(:j)) in 0:(Nx >> 1)
                $(esc(:_j)) = $(esc(:j)) + 1
                for $(esc(:_i)) = 1:Ny
                    $(esc(expr))
                end
            end
            for $(esc(:j)) in (-((Nx - 1) >> 1)):-1
                $(esc(:_j)) = Nx + $(esc(:j)) + 1
                for $(esc(:_i)) = 1:Ny
                    $(esc(expr))
                end
            end
        end
    end
end
