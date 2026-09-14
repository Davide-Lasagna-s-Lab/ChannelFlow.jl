export @loop_jk_i

macro loop_jk_i(SIZE, expr)
    quote
        N1, N2, N3 = $(esc(SIZE))
        @inbounds for $(esc(:_i)) = 1:N3
            for $(esc(:k)) in 0:N2>>1
                $(esc(:_k)) = 1 + $(esc(:k))
                for $(esc(:j)) in 0:N1>>1
                    $(esc(:_j)) = $(esc(:j)) + 1
                    $(esc(expr))
                end
                for $(esc(:j)) in (-N1>>1+1):-1
                    $(esc(:_j)) = N1 + $(esc(:j)) + 1
                    $(esc(expr))
                end
            end
        end
    end
end