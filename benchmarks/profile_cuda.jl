# Included only after loading CUDA: CPU profiling does not require that package.
function profile_cuda(p, s)
    advance() = step!(p.scheme, p.nlterm, velocity(s), stagepressure(s), 0.0; p.constraint...)
    for _ = 1:10
        advance()
    end
    CUDA.synchronize()
    CUDA.@profile trace=true begin
        for _ = 1:3
            advance()
        end
        CUDA.synchronize()
    end
end
