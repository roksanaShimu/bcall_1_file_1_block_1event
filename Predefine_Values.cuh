/*
 * Predefine_Values.cuh
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */

/*
 * Predefine_Values.hpp
 *
 *  Created on: Nov 21, 2016
 *      Author: roksana
 */

#ifndef __PREDEFINE_VALUES_CUH
#define __PREDEFINE_VALUES_CUH


//#define length_of_from_v 21 //21

#define warp_size 32


/*********************************************************************
IMPORTANT: total number of threads should be equal to the number
of states. otherwise few parts of the code won't work
**********************************************************************/
#define number_of_blocks 1 //32 //128//64 //32
#define threads_per_block 1024 //128
#define number_of_block_per_group 4 
#define number_of_iteration 4


#endif
