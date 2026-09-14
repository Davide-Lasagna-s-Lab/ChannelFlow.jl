
struct FTFieldSize{NPOINTS, ISDEALIASED} end

Base.size(::FTFieldSize{NPOINTS, false}) = 
    (NPOINTS[1], NPOINTS[2]>>1 + 1, NPOINTS[3])

Base.size(::FTFieldSize{NPOINTS,  true}) = 
    (NPOINTS[1], NPOINTS[2]>>1 + 1, NPOINTS[3])