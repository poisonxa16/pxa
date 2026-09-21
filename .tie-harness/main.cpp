// A/B harness: legacy tie semantics vs shipped tie semantics on the PXQ6 (pxq6r) quantizer.
// Asserts: (1) total weighted SSE is BIT-IDENTICAL (quality-neutral), (2) forced-tie blocks
// resolve exactly per the deterministic selector, (3) random-data outputs stay sane.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
extern "C" void q_new(const float*, uint8_t*, int64_t, int64_t);
extern "C" void d_new(const uint8_t*, float*, int64_t, int64_t);
extern "C" void q_old(const float*, uint8_t*, int64_t, int64_t);
extern "C" void d_old(const uint8_t*, float*, int64_t, int64_t);

static bool sel(int64_t row, int64_t blk) {  // mirror of the shipped selector
    uint64_t v = ((uint64_t)row ^ (uint64_t)blk) ^ 0x5Aull;
    v ^= v>>32; v ^= v>>16; v ^= v>>8; v ^= v>>4; v ^= v>>2; v ^= v>>1;
    return (v & 1) != 0;
}
static double sse(const std::vector<float>&a, const std::vector<float>&b){
    double s=0; for(size_t i=0;i<a.size();++i){ double d=(double)a[i]-(double)b[i]; s+=d*d; } return s;
}
int main(){
    int fails=0;
    const int64_t R=64,K=64,KB=K/32;
    const int64_t HDR=128, SLAB=1344, CODE_OFF=64;
    const size_t bytes=(size_t)(HDR+KB*SLAB);
    // --- case A: forced ties. block0 of each row anchors; blocks 1..3 are tiny -> all-16-way sub tie
    std::vector<float> srcA(R*K, 0.0f);
    for(int64_t r=0;r<R;++r){ srcA[r*K+0]=1.0f; for(int64_t i=16;i<K;++i) srcA[r*K+i]=1e-7f*(float)((i%7)+1); }
    std::vector<uint8_t> qa(bytes,0), qb(bytes,0);
    q_old(srcA.data(), qa.data(), R, K);
    q_new(srcA.data(), qb.data(), R, K);
    int ties_checked=0, pat_bad=0;
    for(int64_t r=0;r<R;++r){
        for(int64_t blk=1;blk<K/16;++blk){       // tie blocks only
            const int64_t kb=blk/2; const int nib=blk&1;
            const uint8_t so=qa[HDR+kb*SLAB+r], sn=qb[HDR+kb*SLAB+r];
            const int s4o = nib? (so>>4):(so&0xf);
            const int s4n = nib? (sn>>4):(sn&0xf);
            if (s4o != 0) { ++pat_bad; continue; }              // legacy must take lowest
            const int expect = sel(r,blk) ? 15 : 0;             // shipped: hi on sel, else lowest
            if (s4n != expect) ++pat_bad;
            ++ties_checked;
        }
    }
    printf("tie blocks checked: %d, pattern mismatches: %d  -> %s\n", ties_checked, pat_bad, pat_bad?"FAIL":"PASS");
    fails += pat_bad?1:0;
    std::vector<float> da(R*K), db(R*K);
    d_old(qa.data(), da.data(), R, K); d_new(qb.data(), db.data(), R, K);
    double ea=sse(da,srcA), eb=sse(db,srcA);
    printf("forced-tie SSE old=%.17g new=%.17g  -> %s\n", ea, eb, (ea==eb)?"PASS":"FAIL");
    if(ea!=eb) ++fails;
    // --- case B: random data — outputs must carry bit-identical total error
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f,0.1f);
    std::vector<float> srcB(R*K); for(auto&v:srcB)v=nd(rng);
    std::vector<uint8_t> qc(bytes,0), qd(bytes,0);
    q_old(srcB.data(), qc.data(), R, K);
    q_new(srcB.data(), qd.data(), R, K);
    d_old(qc.data(), da.data(), R, K); d_new(qd.data(), db.data(), R, K);
    ea=sse(da,srcB); eb=sse(db,srcB);
    printf("random SSE old=%.17g new=%.17g (bytes %s)  -> %s\n", ea, eb,
           memcmp(qc.data(),qd.data(),bytes)?"differ":"identical", (ea==eb)?"PASS":"FAIL");
    if(ea!=eb) ++fails;
    printf(fails?"HARNESS FAIL\n":"HARNESS ALL PASS\n");
    return fails?1:0;
}
