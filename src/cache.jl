


struct SlabTransposeCache{F<:SpectralField}
    tmps::NTuple{2, F}
    function SlabTransposeCache(OUT::F) where {T,
                                               SIZE,
                                               F<:SpectralField{T, SIZE}}
        # global size
        gridsize_ = gridsize(OUT)

        # construct temporaries to be decomposed along direction 2
        a = SpectralField{gridsize}(SlabArray(comm, gridsize_, T, 2))
        b = SpectralField{gridsize}(SlabArray(comm, gridsize_, T, 2))

        # return object
        return new{F}((a, b))
    end
end