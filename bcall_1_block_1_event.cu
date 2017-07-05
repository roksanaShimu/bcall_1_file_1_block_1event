/*
 ============================================================================
 Name        : bcall_1_block_1_event.cu
 Author      : Roksana Hossain
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */


#include<iostream>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>
#include <errno.h>
#include <libgen.h>
#include <stdbool.h>
#include <dirent.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>


#include "bcall.h"
#include "Pore_Model.cuh"
#include "Sequence.cuh"
#include "Viterbi_TcBk_v1_multi_events.cuh"

using namespace std;

int verbose = 0;
char * prog;

int load_pore_model(const char * fname)
{
	FILE * f;
	int ret;

	if (!(f = fopen(fname, "rb"))) {
		fprintf(stderr, "%s: can't open file \"%s\"\n", prog, fname);
		fflush(stderr);
		return -1;
	};

	ret = pore_model_read(f);

	fclose(f);

	return ret;
}


int load_transitions(const char * fname)
{
	FILE * f;
	int ret;

	if (!(f = fopen(fname, "rb"))) {
		fprintf(stderr, "%s: can't open file \"%s\"\n", prog, fname);
		fflush(stderr);
		return -1;
	};
#if 0
	if ((ret = transitions_read(f)) < 0) {
		fprintf(stderr, "%s: error reading transitions\n", prog);
		fflush(stderr);
		return ret;
	}
#endif
	fclose(f);

	return ret;
}

int basecall(const char * fname, bool trans, bool fixpt, bool tcbk)
{
	char prefix[PATH_MAX + 1];
	char outfn[PATH_MAX + 1];
	struct sequence * seq;
	char * s = NULL;
	uint32_t begin;
	int32_t dt;
	char * cp;
	FILE * f;
	int i;
	int j;

	printf("- Loading file: \"%s\"\n", fname);
	fflush(stdout);



	if ((seq = sequence_from_bin_file(fname)) == NULL) {
		fprintf(stderr,
				"%s: invalid sequence file \"%s\"\n", prog, fname);
		fflush(stderr);
		return -1;
	}

	cout<<"loading completed..."<<endl;
	/* No prefix from the command line, use the filename */
	strcpy(prefix, basename((char *)fname));
	/* strip the file extension */
	if ((cp = strrchr(prefix, '.')) != NULL)
		*cp = '\0';

	sprintf(outfn, fixpt ? "%s-fix.txt" : "%s.txt", prefix);
	if (!(f = fopen(outfn, "w"))) {
		fprintf(stderr,
				"%s: can't create file \"%s\"\n", prog, fname);
		fflush(stderr);
		return -1;
	};




	struct bcall_tcbk * bcall_tcbk;

	bcall_tcbk = tcbk_blk_alloc();

	for (i = 0; i < seq->st_cnt; ++i) {
		int64_t p_max = INT64_MIN;

		printf("- Reading strand: %s:%d (%d).\n", seq->st[i].id,
			   seq->st[i].no, seq->st[i].ev_cnt);
		fflush(stdout);

		for (j = 0; j < seq->st[i].pm_cnt; ++j) {
			unsigned int size;
			int64_t p;

			tcbk_strand_load(bcall_tcbk, seq, i, j);

			if (tcbk) {
				if (trans) {
					printf("- Computing transitions map...\n");
					fflush(stdout);
					tcbk_compute_transitions(bcall_tcbk, seq->st[i].pm[j].st_param);
				}
				size  = 0;
			}

			/*printf("- Fixed point base calling (Mem=%d.%1dKib)...\n",
				   size / 1024, (size % 1024) / 10);
			fflush(stdout);*/
			//begin = __timestamp();

			if (tcbk) {
				tcbk_sequence_fill(bcall_tcbk);
				p = tcbk_fill_state_seq(bcall_tcbk);
				cout<<"strand no:"<<i<<"   pore model:"<<j<<"   p="<<p<<endl;
			}

			//dt = (int32_t)(__timestamp() - begin);
			//printf(" %02d.%03d seconds.\n", dt / 1000, dt % 1000); \
			//fflush(stdout);

			if (p > p_max) {
				p_max = p;
				s = tcbk_fill_base_seq(bcall_tcbk);
			}

		}


		fprintf(f, ">%s:%s:%d\n", seq->st[i].id,
				seq->name, seq->st[i].no);
		dump_base_seq(f, s);

	}

	tcbk_blk_free(bcall_tcbk);


	fclose(f);
	sequence_free(seq);


	return 0;
}

int main(int argc,  char **argv)
{
	char trans[256];
	bool trans_set = false;

	char pmodel[256];
	bool pmodel_set = false;

	bool fixpt = true;
	bool tcbk = true;



	if (trans_set) {
		printf("- Loading transition map: \"%s\"\n", trans);
		fflush(stdout);
		if (load_transitions(trans) < 0)
			return 4;

	}

	if (pmodel_set) {
		printf("- Loading pore model: \"%s\"\n", pmodel);
		fflush(stdout);
		if (load_pore_model(pmodel) < 0)
			return 3;
	}


	clock_t parallel_start = clock();

	char fname[PATH_MAX + 1];
	strcpy(fname, "LomanLabz_PC_Ecoli_K12_MG1655_20150928_MAP006_1020_1_ch2_file1_strand.bin");

	if (basecall(fname, !trans_set, fixpt, tcbk) < 0){

		return 9;
	}

	clock_t parallel_end = clock();

	cout<< "parallel calculation is over"<<endl;
	printf("Time taken for parallel_code: %.6fs\n", (double)(parallel_end - parallel_start)/CLOCKS_PER_SEC);

	return 0;
}
