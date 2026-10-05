// test-pxa-rep-guard-lazy.cpp -- standalone check of the lazy repetition guard detector (common/sampling.cpp pxa_guard_lazy_update, copied verbatim below):
// random text never arms, a span repeated back to back arms at the right token, the guard cools after the hold, prompt tokens never arm. build: g++ -std=c++17 tests/test-pxa-rep-guard-lazy.cpp
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
typedef int llama_token;
struct P { float penalty_repeat=1.0f, dry_multiplier=0.0f; };
struct common_sampler { std::vector<llama_token> prev; int guard_gen=0, guard_hot=0; float guard_repeat=1.15f, guard_dry=0.8f, user_repeat=1.0f, user_dry=0.0f; P params; };
static int pxa_rep_guard_lazy_k(){ return 6; }
static int pxa_rep_guard_lazy_hold(){ return 96; }
static std::atomic<unsigned long long> g_lazy_gen{0}, g_lazy_hot_tok{0}, g_lazy_arms{0};
static void pxa_guard_lazy_update(struct common_sampler * cs, bool is_generated) {
    if (!is_generated) return;
    ++cs->guard_gen;
    g_lazy_gen.fetch_add(1, std::memory_order_relaxed);
    if (cs->guard_hot > 0) g_lazy_hot_tok.fetch_add(1, std::memory_order_relaxed);
    // a SQUARE: the generated tail is the same span twice in a row (a cycle that has come round once), span length L in [K, 128]. An n-gram that merely
    // occurs earlier in the window is ordinary text (lists, code, markdown repeat 6-8-grams all the time: 64% of the tokens of a prose / code mix were
    // "hot" with that rule); a span repeated back to back is the first lap of a loop.
    const int K = pxa_rep_guard_lazy_k();
    const int n = (int) cs->prev.size();
    const int win = std::min(std::min(cs->guard_gen, n), 512);       // generated tokens still in prev, bounded
    bool repeat = false;
    if (win >= 2*K) {
        const llama_token * t = cs->prev.data() + (n - win);          // t[0 .. win)
        const int Lmax = std::min(256, win/2);
        for (int L = K; L <= Lmax && !repeat; ++L) {
            const llama_token * a2 = t + (win - L), * b2 = t + (win - 2*L);
            int m = 0;
            while (m < L && a2[m] == b2[m]) ++m;
            repeat = m == L;
        }
    }
    if (repeat) { if (cs->guard_hot == 0) g_lazy_arms.fetch_add(1, std::memory_order_relaxed); cs->guard_hot = pxa_rep_guard_lazy_hold(); }
    else if (cs->guard_hot > 0) --cs->guard_hot;
    cs->params.penalty_repeat = cs->guard_hot > 0 ? cs->guard_repeat : cs->user_repeat;
    cs->params.dry_multiplier = cs->guard_hot > 0 ? cs->guard_dry    : cs->user_dry;
}


int main(){
  int bad=0;
  common_sampler cs; unsigned x=12345; int hot_seen=0;
  // 1) random text never arms; text with many repeated 6-8-grams that are NOT back to back never arms (a made-up "markdown": a few phrases reused in random order)
  for(int i=0;i<600;i++){ x=x*1664525u+1013904223u; cs.prev.push_back((x>>8)%50000); pxa_guard_lazy_update(&cs,true); if(cs.guard_hot>0) hot_seen=1; }
  printf("random: hot_seen=%d (want 0)\n", hot_seen); bad+=hot_seen;
  // 2) a loop: 30 random tokens repeated -> hot when the second lap completes (token 60)
  common_sampler c2; std::vector<int> cyc; for(int i=0;i<30;i++){ x=x*1664525u+1013904223u; cyc.push_back((x>>8)%50000);} int first_hot=-1;
  for(int r=0;r<4;r++) for(int i=0;i<30;i++){ c2.prev.push_back(cyc[i]); pxa_guard_lazy_update(&c2,true); if(first_hot<0 && c2.guard_hot>0) first_hot=(int)c2.prev.size(); }
  printf("loop: first hot at generated token %d (want 60), repeat now %.2f (want 1.15)\n", first_hot, c2.params.penalty_repeat); bad+=first_hot!=60;
  // 3) period-1 run: the same token 12 times -> hot at token 12
  common_sampler c4; int fh=-1; for(int i=0;i<20;i++){ c4.prev.push_back(7); pxa_guard_lazy_update(&c4,true); if(fh<0&&c4.guard_hot>0) fh=(int)c4.prev.size(); }
  printf("period-1 run: first hot at %d (want 12)\n", fh); bad+=fh!=12;
  // 4) decay
  int cool=-1; for(int i=0;i<300;i++){ x=x*1664525u+1013904223u; c2.prev.push_back((x>>8)%50000); pxa_guard_lazy_update(&c2,true); if(cool<0 && c2.guard_hot==0) cool=i; }
  printf("decay: cooled after %d fresh tokens (want ~95), repeat %.2f (want 1.00)\n", cool, c2.params.penalty_repeat); bad+=!(cool>=90&&cool<=100);
  // 5) prompt tokens never arm
  common_sampler c3; for(int r=0;r<5;r++) for(int i=0;i<30;i++){ c3.prev.push_back(cyc[i]); pxa_guard_lazy_update(&c3,false);} printf("prompt repeats: hot=%d (want 0)\n", c3.guard_hot); bad+=c3.guard_hot!=0;
  printf("%s\n", bad? "FAILED":"ALL PASS"); return bad; }
