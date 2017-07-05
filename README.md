
# bcall_1_file_1_block_1event


This bcall file processes 1 .bin file at a time. During the calculation in GPU, it uses 1 block of threads. CPU calls the kernel for every event. 

For an example, the data contains strand#0 and strand#1. And each strand includes two sets of pore model. So we will have 4 groups of data: 1. strand#0, pore model#0, 2.strand#0, pore model#1, 3.strand#1, pore model#0 and 4.strand#1, pore model#1. We need to apply viterbi algorithm or calculate the probability for all these 4 sets of data. Lets call these 4 groups of data as DATASET# 0, 1, 2 and 3. 

The following pseudo code shows how the DATASETs are processed:

      for loop: DATASET 0 to 3

            for loop: i=1 to number_of_events; i++
      
                  send data from cpu to GPU
              
                  kernel_call<<<1 block, 1024 threads/block>>>();
              
                  recieve data from GPU to CPU
              
            end
      
      end 



In the kernel call, I am using only 1024 threads. but the number of states is 4096. I have used loop  (for 4 times) so that 1024 threads will calculated 4096 states. The pseudo code for the kernel call is given below:

      for loop: i=0 to 3; i++

            calculated transition probability
        
            calculated emission probability
        
            update alpha
        
      end

      find the maximum emission probability


//***************************************************

To change the number of threads per block or iteration, change the values from "Predefined_values.cuh" 


//***************************************************

To export the path(if you have not done yet):

      export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

To run the file from terminal:

      $ nvcc bcall_1_block_1_event.cu -use_fast_math -Xptxas -v -arch=sm_30 -lcurand 

I have used -arch=sm_30 because the compute capability of my GPU is 3.0. check yours one. 

then

      $./a.out



