// GPU contract and full-operation benchmark against the per-token and grouped reference kernels.
#include "pxa-q6k-down2-kernel.cuh"
#include "ggml-quants.h"
#include "quantize.cuh"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>

static void require(bool v,const char *s){if(!v)throw std::runtime_error(s);}
struct Memory{
 std::vector<void*> ps;
 template<class T>T* alloc(size_t n){T*p;CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));ps.push_back(p);return p;}
 ~Memory(){for(auto p:ps)cudaFree(p);}
};
static void baseline(const pxa_mv2_args&a){
 for(int t=0;t<2;++t)
  mul_mat_vec_q<GGML_TYPE_Q6_K,1,1,1,2,true,false><<<dim3(2048,8),32>>>
   (a.vx_u,a.vy+t*a.y_st,(float*)(a.dst+t*a.d_st),a.ids+t*a.ids_nb1,nullptr,
    512,2048,512,2048,a.nb02,a.y_sk,a.d_sk,4,0);
}
int main(int argc,char**argv){try{
 require(argc==2,"device required");int device=std::stoi(argv[1]);require(device==0||device==1,"device");
 ggml_context*cpu=ggml_init({65536,nullptr,true});
 require(cpu&&ggml_fp16_to_fp32(0x3c00)==1.f,"CPU reference initialization");
 CUDA_CHECK(cudaSetDevice(device));
 Memory mem;std::mt19937 rng(42);
 constexpr size_t expert_bytes=2048*2*sizeof(block_q6_K), expert_stride=expert_bytes+128;
 std::vector<block_q6_K> weights(256*2048*2);
 for(auto &b:weights){for(auto &v:b.ql)v=uint8_t(rng());for(auto &v:b.qh)v=uint8_t(rng());for(auto &v:b.scales)v=int8_t(int(rng()%17)-8);const float sign=(rng()&1)?1.f:-1.f;b.d=__float2half(sign/(128+rng()%256));}
 auto*x=mem.alloc<char>(256*expert_stride);
 CUDA_CHECK(cudaMemcpy2D(x,expert_stride,weights.data(),expert_bytes,expert_bytes,256,cudaMemcpyHostToDevice));
 std::vector<float> input(16*512);for(auto&v:input)v=float(int(rng()%255)-127)/32;
 std::fill(input.begin(),input.begin()+512,0.f);
 auto*f=mem.alloc<float>(input.size());CUDA_CHECK(cudaMemcpy(f,input.data(),input.size()*4,cudaMemcpyHostToDevice));
 auto*packed=mem.alloc<block_q8_1>(16*16);
 quantize_row_q8_1_cuda(f,packed,512,16,1,512,GGML_TYPE_Q6_K,nullptr);
 CUDA_CHECK(cudaDeviceSynchronize());
 std::vector<block_q8_1> hostq(256);CUDA_CHECK(cudaMemcpy(hostq.data(),packed,hostq.size()*sizeof(block_q8_1),cudaMemcpyDeviceToHost));
 auto*y=mem.alloc<char>(16*1024),*ids=mem.alloc<char>(128);
 auto*out=mem.alloc<char>(160000);
 cudaEvent_t start,end;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&end));
 const bool quick=getenv("PXA_DOWN2_QUICK");
 for(int duplicates=0;duplicates<3;++duplicates)for(int padded=0;padded<2;++padded)for(int overlap=0;overlap<=8;++overlap){
  if(quick && overlap!=0 && overlap!=4 && overlap!=8)continue;
  int selected[16];for(int s=0;s<8;++s)selected[s]=s==7?255:s;
  for(int s=0;s<8;++s)selected[8+s]=s<overlap?selected[s]:32+s;
  if(duplicates==1)selected[1]=selected[0];
  if(duplicates==2)selected[9]=selected[8];
  std::shuffle(selected+8,selected+16,rng);
  pxa_mv2_args a{};a.vx_u=x;a.vy=y;a.dst=out+64;a.ids=ids;a.nb02=expert_stride;
  a.ncols_x=512;a.nrows_x=2048;a.nrows_y=512;a.ny=2;a.n_ids=8;
  a.y_sk=576+(padded?64:0);a.y_st=8*a.y_sk+(padded?128:0);
  a.d_sk=8192+(padded?64:0);a.d_st=8*a.d_sk+(padded?128:0);a.ids_nb0=4;a.ids_nb1=padded?64:32;
  CUDA_CHECK(cudaMemcpy2D(ids,a.ids_nb1,selected,32,32,2,cudaMemcpyHostToDevice));
  for(int t=0;t<2;++t)CUDA_CHECK(cudaMemcpy2D(y+t*a.y_st,a.y_sk,packed+t*8*16,576,576,8,cudaMemcpyDeviceToDevice));
  const size_t bytes=64+a.d_st*2+64;require(bytes<=160000,"output capacity");
  std::vector<unsigned char> reference(bytes),actual(bytes);
  auto invoke=[&](int route){
   if(route==0)baseline(a);
   else if(route==1)require(pxa_moe_verify2_launch(GGML_TYPE_Q6_K,a,nullptr),"generic declined");
   else require(pxa_q6k_down2_launch(GGML_TYPE_Q6_K,a,nullptr,route==2?1:4),"special declined");
   CUDA_CHECK(cudaGetLastError());
  };
  CUDA_CHECK(cudaMemset(out,0xa5,bytes));invoke(0);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(reference.data(),out,bytes,cudaMemcpyDeviceToHost));
  for(int route=1;route<4;++route){
   if(duplicates && route==1)continue; // Generic grouped route assumes unique IDs.
   CUDA_CHECK(cudaMemset(out,0xa5,bytes));invoke(route);CUDA_CHECK(cudaDeviceSynchronize());
   CUDA_CHECK(cudaMemcpy(actual.data(),out,bytes,cudaMemcpyDeviceToHost));
   require(actual==reference,"bit-exact output or sentinel mismatch");
  }
  for(size_t i=0;i<bytes;++i){bool valid=false;for(int t=0;t<2;++t)for(int s=0;s<8;++s){size_t lo=64+t*a.d_st+s*a.d_sk;if(i>=lo&&i<lo+8192)valid=true;}if(!valid)require(reference[i]==0xa5,"baseline guard corruption");}
  for(int t=0;t<2;++t)for(int s=0;s<8;++s)for(int row:{0,1,2047}){
   float decoded[512];dequantize_row_q6_K(weights.data()+selected[t*8+s]*4096+row*2,decoded,512);
   double dot=0;for(int k=0;k<512;++k)dot+=double(decoded[k])*__low2float(hostq[(t*8+s)*16+k/32].ds)*hostq[(t*8+s)*16+k/32].qs[k%32];
   float value;std::memcpy(&value,reference.data()+64+t*a.d_st+s*a.d_sk+row*4,4);
   require(std::isfinite(value)&&std::abs(value-dot)<.002+std::abs(dot)*2e-5,"independent scalar dot mismatch");
  }
  for(int rep=0;rep<(quick?1:4);++rep)for(int j=0;j<4;++j){int route=(j+rep)%4;if(duplicates && route==1)continue;int iterations=quick?1:100;
   for(int warm=0;warm<4;++warm)invoke(route);
   CUDA_CHECK(cudaEventRecord(start));for(int i=0;i<iterations;++i)invoke(route);CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));
   float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));
   std::cout<<"{\"device\":"<<device<<",\"overlap\":"<<overlap<<",\"padded\":"<<padded<<",\"route\":"<<route<<",\"repeat\":"<<rep<<",\"us\":"<<ms*1000/iterations<<",\"bit_exact\":true}"<<std::endl;
  }
 }
 CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(end));ggml_free(cpu);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
