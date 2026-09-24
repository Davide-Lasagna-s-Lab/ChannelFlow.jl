// Standalone benchmark client of upstream Channelflow (no upstream source edits).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "channelflow/dns.h"
using namespace chflow;

// The exchange file contains u,v,w,q sampled on the common padded grid.
void read_fields(const char* path, std::vector<FlowField>& fields) {
    std::ifstream in(path, std::ios::binary);
    for (auto& field : fields) {
        field.makePhysical();
        for (int c=0; c<field.Nd(); ++c)
            for (int y=0; y<field.Ny(); ++y)
                for (int z=0; z<field.Nz(); ++z)
                    for (int x=0; x<field.Nx(); ++x)
                        in.read(reinterpret_cast<char*>(&field(x,y,z,c)), sizeof(double));
        field.makeSpectral();
    }
    if (!in) throw std::runtime_error("Cannot read complete seed");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: step N samples seed.bin [reference.bin]\n";
        return 1;
    }
    const int N=std::stoi(argv[1]), samples=std::stoi(argv[2]);
    // Upstream Nx/Nz already include padding: the retained cutoff is Nx/3-1.
    const int padded=3*N/2;
    fftw_set_timelimit(1.0);
    std::vector<FlowField> fields;
    for (int nd : {3,1})
        fields.emplace_back(padded,N+1,padded,nd,2*pi,2*pi,-1,1,
                            nullptr,Spectral,Spectral,FFTW_MEASURE);
    read_fields(argv[3],fields);
    const auto original=fields;
    DNSFlags flags;
    flags.baseflow=LaminarBase;
    flags.timestepping=CNRK2;
    flags.initstepping=CNRK2;
    flags.nonlinearity=Rotational;
    flags.dealiasing=DealiasXZ;
    flags.taucorrection=true;
    flags.constraint=PressureGradient;
    flags.dPdx=flags.dPdz=0;
    flags.ulowerwall=-1;
    flags.uupperwall=1;
    flags.nu=1.0/400;
    flags.dt=0.002;
    flags.verbosity=Silent;
    DNS dns(fields,flags);
    for (int i=0; i<5; ++i) {
        fields=original;
        dns.advance(fields,1);
    }
    double minimum=1e100, sum=0;
    for (int i=0; i<samples; ++i) {
        fields=original; // restoration excluded, just as in the Julia benchmark
        auto start=std::chrono::steady_clock::now();
        dns.advance(fields,1);
        double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        minimum=std::min(minimum,dt);
        sum+=dt;
    }
    std::cout << std::setprecision(17) << N << ',' << N+1 << ',' << N << ','
              << padded << ',' << samples << ',' << minimum << ',' << sum/samples << '\n';
    if (argc >= 5) {
        auto reference=original;
        read_fields(argv[4],reference);
        for (int f=0; f<2; ++f) {
            fields[f].makePhysical(); reference[f].makePhysical();
            double error=0, scale=0;
            // Pressure is defined up to a spatially uniform constant.
            const double gauge = f == 1 ? fields[f](0,0,0,0)-reference[f](0,0,0,0) : 0;
            for (int c=0; c<fields[f].Nd(); ++c)
                for (int y=0; y<N+1; ++y)
                    for (int z=0; z<padded; ++z)
                        for (int x=0; x<padded; ++x) {
                            error=std::max(error,std::abs(fields[f](x,y,z,c)-reference[f](x,y,z,c)-gauge));
                            scale=std::max(scale,std::abs(reference[f](x,y,z,c)));
                        }
            std::cerr << "field=" << f << " max_abs_error=" << error << " gauge_offset=" << gauge << " max_reference=" << scale << '\n';
        }
    }
}
