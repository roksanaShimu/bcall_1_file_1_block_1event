/*
 * Kmer.cuh
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */
#ifndef __KMER_CUH
#define __KMER_CUH

#include<iostream>

using namespace std;

#define __BCALL_I__
#include "bcall-i.h"
#include <stdio.h>
#include <inttypes.h>
#include <string.h>



double get_trans_prob(unsigned i, unsigned j, double p_stay,
					  double p_step, double p_skip_1)
{
	unsigned int l;
	double p = 0;

	if (i == j) {
		p += p_stay;
	}

	if (kmer_suffix(i, KMER_SIZE - 1) == kmer_prefix(j, KMER_SIZE - 1)) {
		p += p_step / 4;
	}

	for (l = 2; l < KMER_SIZE; ++l) {
		if (kmer_suffix(i, KMER_SIZE - l) == kmer_prefix(j, KMER_SIZE - l))
			p += pow(p_skip_1, l - 1) / (1 << (2 * l));
	}

	p += (pow(p_skip_1, 5) / (1.0 - p_skip_1)) / N_STATES;

	return p;
}

const char * kmer_to_string(char * s, unsigned int k)
{
	static const char base[] = "ACGT";
	unsigned int j;

	for (j = 0; j < KMER_SIZE; ++j) {
		s[j] = base[(k >> (2 * (KMER_SIZE - j - 1))) & 0x3];
	}
	s[j] = '\0';

	return s;
}

unsigned int kmer_min_skip(unsigned int k1, unsigned int k2)
{
	unsigned int k;

	if (k1 == k2) {
		return 0;
	} else {
		for (k = KMER_SIZE - 1; k > 0; --k) {
			if ((k1 & ((1 << (2 * k)) - 1)) == (k2 >> (2 * (KMER_SIZE - k)))) {
				return KMER_SIZE - k;
			}
		}
		return KMER_SIZE;
	}
}


int encode_base_seq(char base_seq[], uint16_t state[], unsigned int n_events)
{
	unsigned int i;
	unsigned int j;
	char * cp = base_seq;
	char s[KMER_SIZE + 1];

	for (i = 0; i < n_events - 1; ++i) {
		unsigned int n = kmer_min_skip(state[i], state[i + 1]);

		kmer_to_string(s, state[i]);
		for (j = 0; j < n; ++j)
			*cp++ = s[j];
	}
	kmer_to_string(cp, state[n_events - 1]);
	cp += KMER_SIZE;
	*cp = '\0';

	return cp - base_seq;
}


void dump_base_seq(FILE * f, char * base_seq)
{
	char s[128];
	char * cp;
	bool done;

	assert(f != NULL);
	assert(base_seq != NULL);

	cp = base_seq;
	do {
		strncpy(s, cp, 80);
		done = (s[79] == '\0') ? true : false;
		s[80] = '\0';
		fprintf(f, "%s\n", s);
		cp += 80;
	} while (!done);

	fflush(f);
}


#endif
