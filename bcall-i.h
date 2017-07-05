/*
 /*
 /*
 * bcall - Bob Base Call Test Application
 *
 * This file is part of bcall.
 *
 * Ell is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/*!
 * \file      bcall-i.h
 * \brief     Bob Basecall test application
 * \author    Robinson Mittmann <bobmittmann@gmail.com>
 * \copyright 2016, Bob Mittmann
 */


#ifndef __BCALL_I_H__
#define __BCALL_I_H__

#ifndef __BCALL_I__
#error "Never use <bcall-i.h> directly; include <bcall.h> instead."
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include <math.h>
#include <assert.h>

#define DEBUG_LEVEL 5
//#include "debug.h"

#define MAX_K_LEN 8

//
// This struct represents the expected signal measured
// given the kmer sequence that is in the pore when the
// the observations are made. A pore model consists
// of 1024 of these entries (one per 5-mer) and global
// shift/scaling parameters.
//
struct model_entry
{
	long long variant;
	double level_mean;
	double level_stdv;
	double sd_mean;
	double sd_stdv;
	double weight;
	char kmer[MAX_K_LEN];
};

/* This struct represents the global transformations
   that must be applied to each Model_Entry */
struct model_params
{
	double drift;
	double scale;
	double scale_sd;
	double shift;
	double var;
	double var_sd;
};

/* State transition parameters */
struct st_params {
	double p_skip;
	double p_stay;
};

//
// This struct represents an observed event.
// The members of the struct are the same as
// the fields encoded in the FAST5 file.
//
struct event_entry
{
	double mean;
	double stdv;
	double start;
	double length;
	double p_model_state;
	double p_mp_state;
	double p_A;
	double p_C;
	double p_G;
	double p_T;
	long long move;
	char model_state[MAX_K_LEN];
	char mp_state[MAX_K_LEN];
};

/* ---------------------------------------------------------------------------
 * Fixed point basic math
 * ---------------------------------------------------------------------------
 */

/* Conversion form float to fixed point Q15.16 */
#define Q16(F) ((int32_t)((F) * (1 << 16)))

/* Convert from fractional Q15.16 to float point */
#define Q16F(Q) ((float)(((float)(Q)) / (1.0 * (1 << 16))))

/* Q16 Multiply */
#define Q16MUL(X1, X2) (((int64_t)(X1) * (int64_t)(X2) + (1 << 15)) >> 16)

#define Q16ADD(X1, X2) (INT32_MAX - (X1) < (X2) ? (INT32_MAX) :\
						INT32_MIN + (X1) < (X2) ? (INT32_MAX) : (X1) + (X2))

#define Q16SUB(X1, X2) (((int64_t)(X1) * (int64_t)(X2) + (1 << 15)) >> 16)

/* Q16 Divide */
#define Q16DIV(X, Y) (((int64_t)(X) << 16) / (Y))

/* ---------------------------------------------------------------------------
 * Configuration options
 * ---------------------------------------------------------------------------
 */

#ifndef KMER_SIZE
#define KMER_SIZE   6
#endif

#ifndef N_EVENTS_MAX
#define N_EVENTS_MAX (32 * 1024)
#endif

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif

/* ---------------------------------------------------------------------------
 * Kmer fast operations
 * ---------------------------------------------------------------------------
 */

#define N_STATES    (1 << (2 * (KMER_SIZE)))

static inline unsigned int kmer_prefix(unsigned int i, unsigned int k) {
	return i >> (2 * (KMER_SIZE - k));
}

static inline unsigned int kmer_suffix(unsigned int i, unsigned int k) {
	return i & ((1 << (2 * k)) - 1);
}

static inline unsigned int next_state(unsigned int state, unsigned int base) {
	return (kmer_suffix(state, KMER_SIZE - 1) << 2) + base;
}

static inline unsigned int prev_state(unsigned int state,
									  unsigned int base) {
	return kmer_prefix(state, KMER_SIZE - 1) + (base << (KMER_SIZE * 2 - 2));
}

static inline unsigned int prev_prev_state(unsigned int state,
										   unsigned int seq) {
	return kmer_prefix(state, KMER_SIZE - 2) + (seq << (KMER_SIZE * 2 - 4));
}

static inline unsigned int kmer_step(unsigned int state, unsigned int base) {
	return (state >> 2) + (base << (KMER_SIZE * 2 - 2));
}

static inline unsigned int kmer_skip(unsigned int state, unsigned int seq) {
	return (state >> 4) + (seq << (KMER_SIZE * 2 - 4));
}

/* ---------------------------------------------------------------------------
 * useful constants
 * ---------------------------------------------------------------------------
 */
#define SQRT_2PI 2.5066282746310002
#define LOG_2PI 1.8378770664093453

/* ---------------------------------------------------------------------------
 * Floating point base calling structures
 * ---------------------------------------------------------------------------
 */
typedef FLOAT_TYPE bcfloat_t;

/* MinION pore model state */
struct pm_state {
	bcfloat_t level_mean;
	bcfloat_t level_stdv;
	bcfloat_t sd_mean;
	bcfloat_t sd_stdv;
	bcfloat_t sd_lambda;

	bcfloat_t log_level_mean;
	bcfloat_t log_level_stdv;
	bcfloat_t log_sd_mean;
	bcfloat_t log_sd_stdv;
	bcfloat_t log_sd_lambda;
};

/* MinION event definition */
struct event {
	bcfloat_t mean;
	bcfloat_t stdv;
	bcfloat_t start;
	bcfloat_t length;
	bcfloat_t log_mean;
	bcfloat_t log_stdv;
	bcfloat_t log_start;
};

#define MAX_TRANS (1 + 4 + 16)

/* Transitions map */
struct transition {
	unsigned int cnt;
	uint16_t from[MAX_TRANS];
	bcfloat_t log_pr[MAX_TRANS];
};

#ifdef __cplusplus
extern "C" {
#endif

const char * kmer_to_string(char * s, unsigned int k);
unsigned int kmer_min_skip(unsigned int k1, unsigned int k2);
double get_trans_prob(unsigned i, unsigned j, double p_stay,
					  double p_step, double p_skip_1);

void pore_model_init(struct pm_state pm_state[],
					 const struct model_entry me[], unsigned int len);
void pore_model_scale(struct pm_state pm_state[], struct model_params * param);

void transitions_init(struct st_params * param);

int encode_base_seq(char base_seq[], uint16_t state[], unsigned int cnt);

#ifdef __cplusplus
}
#endif

#endif /* __BCALL_I_H__ */
