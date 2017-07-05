/*
 /*
 * bcall.h
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */
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
 * \file      bcall.h
 * \brief     Bob Basecall public API
 * \author    Robinson Mittmann <bobmittmann@gmail.com>
 * \copyright 2016, Bob Mittmann
 */


#ifndef __BCALL_H__
#define __BCALL_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

/* Opaque structures */
struct event;
struct model_params;
struct st_params;
struct pm_state;
struct bcfblk;
struct bcall_tcbk;

/* Pore Model */
struct pore_model {
	char * name;
	struct st_params * st_param;
	struct model_params * pm_param;
	struct model_entry * pm_entry;
};

/* Strand */
struct strand {
	char * id;
	uint8_t no;
	unsigned int pm_cnt;
	struct pore_model * pm;
	unsigned int ev_cnt;
	struct event_entry * ev;
};

/* Strand File */
struct sequence {
	char * name;
	unsigned int st_cnt;
	struct strand * st;
};

#ifdef __cplusplus
extern "C" {
#endif

int sequence_read(FILE * f);
int pore_model_read(FILE * f);
int transitions_read(FILE * f);
int state_trans_param_read(FILE * f);
void compute_transitions_fast(struct bcfblk * bp, struct st_params * param);
void dump_transitions(FILE * f);
void dump_transitions_prob(FILE * f);

int pore_model_dump(FILE * f);

void sequence_fill(struct bcfblk * bp);
double fill_state_seq(struct bcfblk * bp);

char * fill_base_seq(struct bcfblk * bp);
void dump_base_seq(FILE * f, char * base_seq);

unsigned int fixpt_copy(void);
void fixpt_sequence_fill(void);
int64_t fixpt_fill_state_seq(void);

struct bcall_tcbk * tcbk_blk_alloc(void);
void tcbk_blk_free(struct bcall_tcbk * fix);
int tcbk_strand_load(struct bcall_tcbk * fix, struct sequence * seq,
					 unsigned int st_no, unsigned int pm_no);
void tcbk_sequence_fill(struct bcall_tcbk * fix);
int64_t tcbk_fill_state_seq(struct bcall_tcbk * fix);
void tcbk_compute_transitions(struct bcall_tcbk * fix,
							  struct st_params * param);
char * tcbk_fill_base_seq(struct bcall_tcbk * fix);
void tcbk_dump_transitions(struct bcall_tcbk * fix, FILE * f);
void tcbk_dump_events(struct bcall_tcbk * fix, FILE * f);
void tcbk_dump_pm(struct bcall_tcbk * fix, FILE * f);


unsigned int hw_copy(void);
void hw_sequence_fill(void);
int64_t hw_fill_state_seq(void);
void hw_compute_transitions(struct st_params * param);
void hw_dump_transitions(FILE * f);
void hw_dump_events(FILE * f);
void hw_dump_pm(FILE * f);
void hw_pack_pm(FILE * f);
int hw_sequence_strand_load(struct sequence * seq,
							unsigned int st_no,
							unsigned int pm_no);

struct sequence * sequence_from_bin_file(const char * fname);

void sequence_free(struct sequence * seq);

int sequence_strand_load(struct bcfblk * bp,
						 struct sequence * seq,
						 unsigned int st_no,
						 unsigned int pm_no);

void bcall_blk_free(struct bcfblk * bp);
struct bcfblk * bcall_blk_alloc(void);

#ifdef __cplusplus
}
#endif

#endif /* __BCALL_H__ */
