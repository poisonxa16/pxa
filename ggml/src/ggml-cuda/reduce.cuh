#include "common.cuh"

// PXA_TSPLIT_P2P_PAIR: peer access is a per-PAIR property, not a per-device one. A group of cards
// that spans two P2P islands (e.g. a Pascal card and a Volta card, which cannot peer at all on this
// class of board) used to turn the fast peer reduce route OFF on every card in the process, because
// ggml_cuda_set_peer_access() ANDed over all visible devices and returned one bool. These read the
// real matrix, so a reduce that runs inside one island keeps the fast route even when the process
// also holds a card outside it.
bool pxa_tsplit_p2p_pair (int a, int b);            // a may dereference b's device pointers
bool pxa_tsplit_p2p_group(const int * idx, int n);  // every ordered pair in the group may
void pxa_tsplit_p2p_dump (const char * tag);        // one banner line with the matrix

#define CUDA_REDUCE_BLOCK_SIZE 256

void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fake_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// PXA_TSPLIT_REDUCE: the seam between the reduce and the dispatch. A tensor-split all-reduce over
// N devices is N per-device HALVES. Half i is enqueued only onto device i's compute stream, by
// whichever host thread owns device i; it orders itself against the peers purely on the GPU and
// never blocks the host. publish() is the lock-free (pointer, event, generation) record a half
// reads to find its peers' partials for a given reduce index.
extern "C" void pxa_tsplit_reduce_publish(int dev_i, long long reduce_index, void * ptr, cudaEvent_t ev,
                                          int nelem, unsigned long long gen);
extern "C" void pxa_tsplit_reduce_half(ggml_backend_cuda_context & ctx_i, ggml_tensor * dst,
                                       long long reduce_index, int n_dev, int dev_i,
                                       const int * devs, int slot, int token);

// A half whose cross-device arrival spin ran out writes NaN instead of a sum and raises a fault
// flag. This turns the flag into a stop. Call it wherever the host waits for the device -- above all
// at the token join, before any logits are read: the host runs far ahead of the device, so a fault
// raised by the LAST reduce of a run would otherwise never be looked at. No-op (one atomic load)
// until the route has built a pipeline.
void pxa_tsplit_reduce_check_fault(void);
