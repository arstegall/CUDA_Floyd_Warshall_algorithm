//Jednostavna implementacija Floyd-Warshall-ovog algoritma najkracih puteva s ​​rekonstrukcijom puta. Namijenjeno za upotrebu na usmjerenim grafovima bez negativnih ciklusa.
#include <iostream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <cuda.h>
#include <ctime>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
#include <crtdbg.h>//za detektiranje curenja memorije hosta
using namespace std;

#define _DTH cudaMemcpyDeviceToHost//kopiranje memorijskog prostora iz GPU na host
#define _HTD cudaMemcpyHostToDevice//kopiranje mem. pr. s hosta na GPU

#define RANGE 997
#define RANDOM_GSIZE 700
#define INF (1<<22)


void _CPU_Floyd(int *G,int *Gpath,int N);

__global__ void _Wake_GPU(int reps);
__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N);

void _GPU_Floyd(int *H_G, int *H_Gpath, const int N);


void _generate_result_file(bool success, unsigned int cpu_time, unsigned int gpu_time, int N);
void _generateRandomGraph(int *G, int N, int range, int density);


int main(){
	char izbor;
	srand(time(NULL)); //generiranje nasumicnog broja
	
	cudaDeviceProp devProp; //deklariranje varijable koja sprema informacije o nVIdijom GPU 
	cudaGetDeviceProperties(&devProp, 0); //pozivanje funkcije koja ucitava informacije

	int MAX_THREADS_PER_BLOCK = devProp.maxThreadsPerBlock;
	int BLOCK_SIZE = devProp.maxGridSize[0];

	cout<<"Zelite li upisati broj bridova (d/n)?\nAko je odgovor ne uzima se unaprijed odredjeni broj."<<endl;
	cin>>izbor;
	int n;
	if(izbor=='d'){
		cout<<"Unesi broj bridova:"<<endl;
		cin>>n;
		}else{
			n = RANDOM_GSIZE;
		}
		
		const int NumBytes=n*n*sizeof(int); //deklariramo velicinu bajtova za varijablu n (broj bridova)

		int *OrigGraph=(int *)malloc(NumBytes);
		int *H_G=(int *)malloc(NumBytes);
		int *H_Gpath=(int *)malloc(NumBytes);
		int *D_G=(int *)malloc(NumBytes);
		int *D_Gpath=(int *)malloc(NumBytes);

		_generateRandomGraph(OrigGraph,n,RANGE,25); //generiranje grafa s obzirom na zadani broj bridova

		
		cout<<"Uspjesno kreiran visoko povezan graf u matrici susjedstva s  "<<n*n<< " elemenata.\n";
		cout<<"Takodjer kreirana dvije matrice za pohranu CPU i GPU rezultata.\n";
		for(int i=0;i<n*n;i++){//kopija za koristenje u racunanju
			H_G[i]=D_G[i]=OrigGraph[i]; //kopija za koristenje u racunanju
			H_Gpath[i]=D_Gpath[i]=-1; 
			
		}
		unsigned int cpu_time=0,gpu_time=0; 
		cout<<"\nFloyd-Warshall na CPU u tijeku: \n";
		DWORD startTime = timeGetTime(); 

		_CPU_Floyd(H_G,H_Gpath,n); //pronalazi najkrace putove na CPU

		DWORD endTime = timeGetTime(); 
		cpu_time=unsigned int(endTime-startTime); 
		printf("CPU vrijeme: %dms\n", cpu_time);
		//budjenje GPU-a iz praznog hoda
		cout<<"\nFloyd-Warshall na GPU u tijeku:\n";
		_Wake_GPU<<<1,BLOCK_SIZE>>>(32);

		
		startTime = timeGetTime();

		_GPU_Floyd(D_G,D_Gpath,n); 

		endTime = timeGetTime();
		gpu_time=unsigned int(endTime-startTime);
		printf("GPU vrijeme (ukljucujuci sve hostove uredjaja, kopije host uredjaja, alokaciju uredjaja i oslobadjanje memorije uredjaja): %dms\n\n", gpu_time);

		
		cout<<"Provjera rezultata matrice konacnog susjedstva i matrice puta...\n";

		int same_adj_Matrix = memcmp(H_G,D_G,NumBytes); //kopiranje vrijednosti s GPU-a na host
		if(same_adj_Matrix==0){
			cout<<"Matrice susjedstva su jednake!\n";
		}else
			cout<<"Matrice susjedstva nisu jendake!\n";

		int same_path_Matrix = memcmp(H_Gpath,D_Gpath,NumBytes);
		if(same_path_Matrix==0){
			cout<<"Matrice obnove puta su jednake!\n";
		}else
			cout<<"Matrice obnove puta nisu jednake!\n";

		_generate_result_file (bool(same_adj_Matrix==0 && same_path_Matrix==0),cpu_time,gpu_time,RANDOM_GSIZE);

		//dealokacija memorije
		free(OrigGraph);
		free(H_G);
		free(H_Gpath);
		free(D_G);
		free(D_Gpath);
		
	_CrtDumpMemoryLeaks();
	return 0;
}

//funkcija za generiranje random grafa
void _generateRandomGraph(int *G,int N,int range, int density){
	int Prange=(100/density);
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			if(i==j){//set G[i][i]=0
				G[i*N+j]=0;
				continue;
			}
			int pr=rand()%Prange;
			G[i*N+j]= pr==0 ? ((rand()%range)+1):INF;
		}
	}
}
//Floydov algoritam na CPU
void _CPU_Floyd(int *G,int *Gpath,int N){ 
	for(int k=0;k<N;++k)for(int i=0;i<N;++i)for(int j=0;j<N;++j){
		int curloc=i*N+j,loca=i*N+k,locb=k*N+j;
		if(G[curloc]>(G[loca]+G[locb])){
			G[curloc]=(G[loca]+G[locb]);
			Gpath[curloc]=k;
		}
	}
}
__global__ void _Wake_GPU(int reps){
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=reps)return;
}
//funkcija Floydovog algoritma koja se izvodi na GPU
__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N){ 
	int col=blockIdx.x*blockDim.x + threadIdx.x; 
	if(col>=N)return; 
	int idx=N*blockIdx.y+col;

	__shared__ int best; 
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];
	__syncthreads(); 
	if(best==INF)return;
	int tmp_b=G[k*N+col];
	if(tmp_b==INF)return;
	int cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
		P[idx]=k;
	}
}

void _GPU_Floyd(int *H_G, int *H_Gpath, const int N){
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);


	int MAX_THREADS_PER_BLOCK = devProp.maxThreadsPerBlock;
	int BLOCK_SIZE = devProp.maxGridSize[0];
	//alociranje memorije uredjaja i kopiranja podataka s grafa hosta
	int *dG,*dP;
	int numBytes=N*N*sizeof(int);
	cudaError_t err=cudaMalloc((int **)&dG,numBytes);	//alokacija memorije na GPU
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((int **)&dP,numBytes);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	//kopiranje informacija o grafu s hosta na uredjaj
	err=cudaMemcpy(dG,H_G,numBytes,_HTD);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(dP,H_Gpath,numBytes,_HTD);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE,N);

	for(int k=0;k<N;k++){ //glavna petlja

		_GPU_Floyd_kernel<<<dimGrid,BLOCK_SIZE>>>(k,dG,dP,N);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	
	err=cudaMemcpy(H_G,dG,numBytes,_DTH);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(H_Gpath,dP,numBytes,_DTH);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	
	err=cudaFree(dG);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(dP);
	if(err!=cudaSuccess){printf("%s u %s u liniji %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
}

//rezultati se zapisuju u datoteku
void _generate_result_file(bool success, unsigned int cpu_time,unsigned int gpu_time, int N){

	if(!success){
		ofstream myfile;
		myfile.open("rezultati.txt");
		myfile<<"CPU timing za sve je bio "<<float(cpu_time)/1000.0f<<" sekundi, a GPU timing(ukljucujuci sve operacije memorije uredjaja - alokacija, kopije itd, za sve je bio "<<float(gpu_time)/1000.0f<<" sekundi.\n";
		myfile<<"GPU rezultat je bio "<<float(cpu_time)/float(gpu_time)<<" puta brzi nego CPU rezultat.\n";
		myfile.close();
	}
}