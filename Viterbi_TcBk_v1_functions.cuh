
//Viterbi_TcBk_v1_functions.cuh


#ifndef VITERBI_TCBK_V1_FUNCTIONS_CUH_
#define VITERBI_TCBK_V1_FUNCTIONS_CUH_





#include<iostream>
#include<cuda.h>
#include <ctime>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>

#define __BCALL_I__
#include "bcall-i.h"
#include "bcall.h"
#include "Kmer.cuh"
#include "Predefine_Values.cuh"
#include "Device_Memory_Allocation.cuh"

using namespace std;



#define Q 18

/* Conversion form float to fixed point */
#define QX(F) ((int32_t)((F) * (1 << (Q))))

/* Convert from fixed point to float point */
#define QXF(Q) ((float)(((float)(Q)) / (1.0 * (1 << (Q)))))

/* Multiply */
#define QXMUL(X1, X2) (int32_t)(((int64_t)(X1) * (int64_t)(X2)) >> (Q))

/* MinION event definition */
struct fix_event {
	int32_t mean;
	int32_t stdv;
	int32_t stdv_inv;
	int32_t log_stdv_1p5;
};

/* Pore model state */
struct fix_pm_state {
	int32_t level_mean;
	int32_t sd_mean;
	int32_t sd_mean_inv;
	int32_t level_stdv_inv_2;
	int32_t sd_lambda_p5;
	int32_t log_level_stdv_2pi;
	int32_t log_sd_lambda_p5;
};

/* Unroll the inner loop */
#define UNROLL_INNER 1

/* Transitions map */
struct fix_map {
	int32_t log_pr[MAX_TRANS];
	struct fix_pm_state pm;
};

/* Length of the trace back buffer */
#define TRACEBACK_LEN 128
/* To disable continuous traceback set TRACEBACK_LEN to N_EVENTS_MAX */
//#define TRACEBACK_LEN N_EVENTS_MAX

/* Period (in events) in which the fixpt_traceback() function is called */
#define TRACEBACK_CNT 32
/* To disable continuous traceback set TRACEBACK_CNT to N_EVENTS_MAX */
//#define TRACEBACK_CNT N_EVENTS_MAX

struct bcall_tcbk {
	/* Transitions map */
	struct fix_map map[N_STATES];
	/* previous state in the MLSS */
	uint16_t beta[TRACEBACK_LEN][N_STATES];
	/* Store the maximum probability for the most probable path */
	int64_t sum_log_p_max;
	/* Sequence of events */
	struct fix_event event[N_EVENTS_MAX];
	unsigned int n_events;
	uint16_t state_seq[N_EVENTS_MAX];
	/* Resulting sequence of bases (output) */
	char base_seq[N_EVENTS_MAX];
	unsigned int base_cnt;
};






/* ---------------------------------------------------------------------------
 * Viterbi algorithm
 * ---------------------------------------------------------------------------
 */


__constant__ int32_t d_ev[4];

static inline int32_t ln_pemission(struct fix_pm_state * pm,
							struct fix_event * e,
							int32_t p_max,  int32_t ln_pt)
{
	int32_t pm_mu = pm->sd_mean;
	int32_t pm_r_mu = pm->sd_mean_inv;
	int32_t pm_lmbd1 = pm->sd_lambda_p5;
	int32_t pm_mean = pm->level_mean;
	int32_t pm_r_stdv1 = pm->level_stdv_inv_2;
	int32_t pm_l_lmbd1 = pm->log_sd_lambda_p5;
	int32_t pm_l_stdv1 = pm->log_level_stdv_2pi;

	int32_t e_stdv = e->stdv;
	int32_t e_r_stdv = e->stdv_inv;
	int32_t e_mean = e->mean;
	int32_t e_l_stdv1 = e->log_stdv_1p5;

	/* First stage */
	int32_t x1;
	int32_t x2;
	int32_t x3;
	int32_t x4;
	int32_t x5;
	int32_t x6;
	int32_t x7;
	int32_t x8;
	/* Second stage */
	int32_t y1;
	int32_t y2;
	int32_t y3;
	int32_t y4;
	int32_t y5;
	/* Third stage */
	int32_t z1;
	int32_t z2;
	int32_t z3;
	int32_t z4;
	/* Forth stage */
	int32_t w1;
	int32_t w2;
	/* Fifth stage */
	int32_t v;

	/* First stage */
	x1 = ln_pt + pm_l_lmbd1;
	x2 = e_l_stdv1 + p_max;
	x3 = pm_l_stdv1;
	x5 = pm_r_stdv1;
	x8 = QXMUL(pm_lmbd1, e_r_stdv);
	x4 = e_mean - pm_mean;
	x6 = e_stdv - pm_mu;
	x7 = pm_r_mu;

	/* Second stage */
	y1 = x1 - x2;
	y2 = x3;
	y3 = QXMUL(x4, x5);
	y4 = QXMUL(x6, x7);
	y5 = x8;

	/* Third stage */
	z1 = y1 - y2;
	z2 = QXMUL(y3, y3);
	z3 = QXMUL(y4, y4);
	z4 = y5;

	/* Forth stage */
	w1 = z1 - z2;
	w2 = QXMUL(z3, z4);

	/* Fifth stage */
	v = w1 - w2;

	return v;
}


static inline void fixpt_traceback(struct bcall_tcbk * fix,
								   int to, unsigned int j_max)
{
	unsigned int from;
	unsigned int i;

	if (to < TRACEBACK_LEN) {
		from = 0;
	} else {
		from = to - TRACEBACK_LEN;
	}

	//DBG(DBG_INFO, "from=%d to=%d", from, to);
	//cout<<"from="<<from<<"   to="<<to<<endl;

	for (i = to - 1; i > from; --i) {
		fix->state_seq[i] = j_max;
		j_max = fix->beta[i % TRACEBACK_LEN][j_max];
	}

	fix->state_seq[i] = j_max;
}


__device__ static inline unsigned int dev_kmer_step(unsigned int state, unsigned int base) {
	return (state >> 2) + (base << (KMER_SIZE * 2 - 2));
}

__device__ static inline unsigned int dev_kmer_skip(unsigned int state, unsigned int seq) {
	return (state >> 4) + (seq << (KMER_SIZE * 2 - 4));
}


__device__ int32_t dev_ln_ptransition(int32_t * devPtr_log_pr, unsigned j, int32_t *alpha, uint16_t *beta){

	unsigned int j1;
	int32_t log_p;
	int32_t y;
	uint16_t temp_beta;



	//start index of devPtr_log_pr
	unsigned group_no=j/warp_size;
	unsigned start_index=group_no*warp_size*21 +j%warp_size;

	//if(j<10)printf("inside dev_ln_ptransition: j=%d   devPtr_log_pr[start_index]=%d \n", j,devPtr_log_pr[start_index] );

	/* Stay */
	log_p = devPtr_log_pr[start_index] + alpha[j];
	temp_beta = j;

	/* Step */
	j1 = dev_kmer_step(j, 0);
	y = devPtr_log_pr[start_index+1*warp_size]+ alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_step(j, 1);
	y = devPtr_log_pr[start_index+2*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_step(j, 2);
	y = devPtr_log_pr[start_index+3*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_step(j, 3);
	y = devPtr_log_pr[start_index+4*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}


	/* Skip */
	j1 = dev_kmer_skip(j, 0);
	y = devPtr_log_pr[start_index+5*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 1);
	y = devPtr_log_pr[start_index+6*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 2);
	y = devPtr_log_pr[start_index+7*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 3);
	y = devPtr_log_pr[start_index+8*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 4);
	y = devPtr_log_pr[start_index+9*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 5);
	y = devPtr_log_pr[start_index+10*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 6);
	y = devPtr_log_pr[start_index+11*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 7);
	y = devPtr_log_pr[start_index+12*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 8);
	y = devPtr_log_pr[start_index+13*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 9);
	y = devPtr_log_pr[start_index+14*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 10);
	y = devPtr_log_pr[start_index+15*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 11);
	y = devPtr_log_pr[start_index+16*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 12);
	y = devPtr_log_pr[start_index+17*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 13);
	y = devPtr_log_pr[start_index+18*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 14);
	y = devPtr_log_pr[start_index+19*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}
	j1 = dev_kmer_skip(j, 15);
	y = devPtr_log_pr[start_index+20*warp_size] + alpha[j1];
	if (y > log_p) {
		log_p = y;
		temp_beta = j1;
	}


	/*if(j<0 || j>=N_STATES){
		printf("j= %d is the out of range\n", j) ;
	}else{
		printf("j is in the range\n");
	}*/

	beta[j]=temp_beta;
	return log_p;

}





__device__ static inline int32_t dev_ln_pemission(int32_t *devPtr_pm, int32_t p_max,  int32_t ln_pt, unsigned j)
{


	//start index for devPtr_pm
	unsigned group_no=j/warp_size;
	unsigned start_index=group_no*warp_size*7 +j%warp_size;


	/* First stage */
	int32_t x1 = ln_pt + devPtr_pm[start_index+ 5*warp_size];
	int32_t x2 = d_ev[3] + p_max;
	int32_t x8 = QXMUL(devPtr_pm[start_index+ 2*warp_size], d_ev[1]);
	int32_t x4 = d_ev[2] - devPtr_pm[start_index+ 3*warp_size];
	int32_t x6 = d_ev[0] - devPtr_pm[start_index];


	/* Second stage */
	 x1 = x1 - x2;

	x2 = QXMUL(x4, devPtr_pm[start_index+ 4*warp_size]);
	x4 = QXMUL(x6, devPtr_pm[start_index+ 1*warp_size]);


	/* Third stage */
	x1 = x1 - devPtr_pm[start_index+ 6*warp_size];
	x2 = QXMUL(x2, x2);
	x4 = QXMUL(x4, x4);


	/* Forth stage */
	x1 = x1 - x2;
	x2 = QXMUL(x4, x8);

	/* Fifth stage */
	x1 = x1 - x2;

	return x1;
	//return 0;
}


struct bcall_tcbk * tcbk_blk_alloc(void)
{
	struct bcall_tcbk * fix;

	fix = (struct bcall_tcbk *)malloc(sizeof(struct bcall_tcbk));

	return fix;
}



/* Apply drift correction */
void tcbk_events_init(struct bcall_tcbk * fix,
					  const struct event_entry entry[],
					  unsigned int len, double drift)
{
	unsigned int i;

	assert(len < N_EVENTS_MAX);

	for (i = 0; i < len; ++i) {
		double mean;
		double stdv;

		mean = entry[i].mean - drift * entry[i].start;
		stdv = entry[i].stdv;

		fix->event[i].mean = QX(mean);
		fix->event[i].stdv = QX(stdv);
		fix->event[i].stdv_inv = QX(1.0/stdv);
		fix->event[i].log_stdv_1p5 = QX(1.5 * log(stdv));
	}

	//DBG(DBG_INFO, "entries=%d", i);
	cout<<" entries= "<< i<<endl;
	fix->n_events = i;
}

int tcbk_strand_load(struct bcall_tcbk * fix, struct sequence * seq,
					 unsigned int st_no, unsigned int pm_no)
{
	struct pm_state pm_state[N_STATES];
	unsigned int i;

	if (st_no >= seq->st_cnt)
		return -1;

	if (pm_no >= seq->st[st_no].pm_cnt)
		return -1;

	//DBG(DBG_INFO, "\"%s\"", seq->st[st_no].pm[pm_no].name);
	cout<<seq->st[st_no].pm[pm_no].name<<endl;

	pore_model_init(pm_state, seq->st[st_no].pm[pm_no].pm_entry, N_STATES);
	pore_model_scale(pm_state, seq->st[st_no].pm[pm_no].pm_param);

	for (i = 0; i < N_STATES; ++i) {
		fix->map[i].pm.level_mean = QX(pm_state[i].level_mean);
		fix->map[i].pm.sd_mean = QX(pm_state[i].sd_mean);
		fix->map[i].pm.sd_mean_inv = QX(1.0/pm_state[i].sd_mean);
		fix->map[i].pm.level_stdv_inv_2 = QX(1.0/ (sqrt(2.0) * pm_state[i].level_stdv));
		fix->map[i].pm.sd_lambda_p5 = QX(pm_state[i].sd_lambda / 2);
		fix->map[i].pm.log_level_stdv_2pi = QX(pm_state[i].log_level_stdv +
											   LOG_2PI);
		fix->map[i].pm.log_sd_lambda_p5 = QX(pm_state[i].log_sd_lambda / 2);
	}

	tcbk_events_init(fix, seq->st[st_no].ev, seq->st[st_no].ev_cnt,
					 seq->st[st_no].pm[pm_no].pm_param->drift);

	return seq->st[st_no].ev_cnt;
}

static void transition_add(struct bcall_tcbk * fix, unsigned int n,
						   unsigned int from, unsigned int to,
						   double p_stay, double p_step,
						   double p_skip_1)
{
	double p = get_trans_prob(from, to, p_stay, p_step, p_skip_1);
	fix->map[to].log_pr[n] = QX(log(p));
}
void tcbk_compute_transitions(struct bcall_tcbk * fix, struct st_params * param)
{
	double p_stay = param->p_stay;
	double p_skip = param->p_skip;
	double p_step = 1.0 - p_stay - p_skip;
	double p_skip_1 = p_skip / (p_skip + 1.0);
	unsigned int i;

	for (i = 0; i < N_STATES; ++i) {
		unsigned int n = 0;
		unsigned int j1;
		unsigned int k;

		/* Stay */
		transition_add(fix, n++, i, i, p_stay, p_step, p_skip_1);
		/* Step */
		for (k = 0; k < 4; ++k) {
			j1 = kmer_step(i, k);
			transition_add(fix, n++, j1, i, p_stay, p_step, p_skip_1);
		}
		/* Skip */
		for (k = 0; k < 16; ++k) {
			j1 = kmer_skip(i, k);
			transition_add(fix, n++, j1, i, p_stay, p_step, p_skip_1);
		}
	}
}


int64_t tcbk_fill_state_seq(struct bcall_tcbk * fix)
{
	int64_t ret;

	ret = fix->sum_log_p_max;
#if 0
	free(fix);
#endif
	return ret;
}

char * tcbk_fill_base_seq(struct bcall_tcbk * fix)
{
	int base_cnt;

	base_cnt = encode_base_seq(fix->base_seq, fix->state_seq, fix->n_events);
	fix->base_cnt = base_cnt;

	return fix->base_seq;
}

void tcbk_blk_free(struct bcall_tcbk * fix)
{
	free(fix);
}



#endif
