/*
 * Device_Memory_Allocation.cuh
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */

#ifndef __DEVICE_MEMORY_ALLOCATION_CUH
#define __DEVICE_MEMORY_ALLOCATION_CUH

cudaError_t checkCuda(cudaError_t result)
{
	#if defined(DEBUG) || defined(_DEBUG)
  	if (result != cudaSuccess) {
    		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    		assert(result == cudaSuccess);
  	}
	#endif
  	return result;
}
//******************************TEXTURE MEMORY ALLOCATION********************************

void allocate_texture_memory_float(texture<float, cudaTextureType1D, cudaReadModeElementType> & texMFloat, float * m_float, size_t m_float_size, unsigned n_states){

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc1 =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);

    	cudaArray* cuArray1;
	cudaMallocArray(&cuArray1, &channelDesc1, n_states, 1);  // with this (&texRef.channelDesc) you don't need to fill a channel descriptor


    	// Copy to device memory some data located at address h_data
    	// in host memory
    	cudaMemcpyToArray(cuArray1, 	// destination: the array
			    	0, 0, 	// offset
				m_float, //source data
				m_float_size, // size of the data
                      		cudaMemcpyHostToDevice);

    	// Set texture reference parameters
    	texMFloat.addressMode[0] = cudaAddressModeClamp; // if my indexing is out of bounds: automatically use a valid indexing (0 if negative index, last if too great index)
    	texMFloat.addressMode[1] = cudaAddressModeClamp;
    	//texRef1.filterMode     = cudaFilterModeLinear; // if it is turned on then it returns the interpolated value
    	texMFloat.normalized     = false;   // don't automatically convert fetched data to [0,1[


    	// Bind the array to the texture reference
    	cudaBindTextureToArray(texMFloat, cuArray1, channelDesc1);
}

void allocate_texture_memory_unsigned(texture<unsigned, cudaTextureType1D, cudaReadModeElementType> & texMUnsigned, unsigned * m_unsigned, size_t m_unsigned_size, unsigned n_states){
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc2 =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindUnsigned);


    	cudaArray* cuArray2;
	cudaMallocArray(&cuArray2, &channelDesc2, n_states, 1);  // with this (&texRef.channelDesc) you don't need to fill a channel descriptor


    	// Copy to device memory some data located at address h_data
    	// in host memory
    	cudaMemcpyToArray(cuArray2, 	// destination: the array
			    	0, 0, 	// offset
				m_unsigned, //source data
				m_unsigned_size, // size of the data
                      		cudaMemcpyHostToDevice);

    	// Set texture reference parameters
    	texMUnsigned.addressMode[0] = cudaAddressModeClamp; // if my indexing is out of bounds: automatically use a valid indexing (0 if negative index, last if too great index)
    	texMUnsigned.addressMode[1] = cudaAddressModeClamp;
    	//texRef1.filterMode     = cudaFilterModeLinear; // if it is turned on then it returns the interpolated value
    	texMUnsigned.normalized     = false;   // don't automatically convert fetched data to [0,1[


    	// Bind the array to the texture reference
    	cudaBindTextureToArray(texMUnsigned, cuArray2, channelDesc2);

}


void allocate_texture_memory_int(texture<int, cudaTextureType1D, cudaReadModeElementType> & texMInt, int * m_int, size_t m_int_size, int n_states){
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc2 =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindSigned);


    	cudaArray* cuArray2;
	cudaMallocArray(&cuArray2, &channelDesc2, n_states, 1);  // with this (&texRef.channelDesc) you don't need to fill a channel descriptor


    	// Copy to device memory some data located at address h_data
    	// in host memory
    	cudaMemcpyToArray(cuArray2, 	// destination: the array
			    	0, 0, 	// offset
				m_int, //source data
				m_int_size, // size of the data
                      		cudaMemcpyHostToDevice);

    	// Set texture reference parameters
    	texMInt.addressMode[0] = cudaAddressModeClamp; // if my indexing is out of bounds: automatically use a valid indexing (0 if negative index, last if too great index)
    	texMInt.addressMode[1] = cudaAddressModeClamp;
    	//texRef1.filterMode     = cudaFilterModeLinear; // if it is turned on then it returns the interpolated value
    	texMInt.normalized     = false;   // don't automatically convert fetched data to [0,1[


    	// Bind the array to the texture reference
    	cudaBindTextureToArray(texMInt, cuArray2, channelDesc2);

}





//******************************PAGE LOCK MEMORY ALLOCATION********************************
unsigned * assign_page_locked_memory_unsigned(unsigned * data_unsigned, size_t data_unsigned_size){
	// set the device flags for mapping host memory
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostRegister(data_unsigned, data_unsigned_size, cudaHostRegisterMapped);
	unsigned *devPtr_unsigned;
	cudaHostGetDevicePointer((void **) &devPtr_unsigned, (void *) data_unsigned, 0);
	return devPtr_unsigned;


}

float * assign_page_locked_memory_float(float * data_float, size_t data_float_size){
	// set the device flags for mapping host memory
	cudaSetDeviceFlags(cudaDeviceMapHost);

	cudaHostRegister(data_float, data_float_size, cudaHostRegisterMapped);
	float *devPtr_float;
	cudaHostGetDevicePointer((void **) &devPtr_float, (void *) data_float, 0);
	return devPtr_float;

}

int32_t * assign_page_locked_memory_int32_t(int32_t * data_int32_t, size_t data_int32_t_size){
	// set the device flags for mapping host memory
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostRegister(data_int32_t, data_int32_t_size, cudaHostRegisterMapped);
	int32_t *devPtr_int32_t;
	cudaHostGetDevicePointer((void **) &devPtr_int32_t, (void *) data_int32_t, 0);
	return devPtr_int32_t;


}

int * assign_page_locked_memory_int(int * data_int, size_t data_int_size){
	// set the device flags for mapping host memory
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostRegister(data_int, data_int_size, cudaHostRegisterMapped);
	int *devPtr_int;
	cudaHostGetDevicePointer((void **) &devPtr_int, (void *) data_int, 0);
	return devPtr_int;


}


//******************************SURFACE MEMORY ALLOCATION********************************

#endif
