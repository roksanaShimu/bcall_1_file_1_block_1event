/*
 * Sequence.cuh
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */

#ifndef __SEQUENCE_CUH
#define __SEQUENCE_CUH

#include<iostream>

using namespace std;

struct sequence * sequence_alloc(const char * name)
{
	struct sequence * seq;

	seq = (struct sequence *)malloc(sizeof(struct sequence));
	assert(seq != NULL);

	seq->name = (char *)malloc(strlen(name) + 1);
	assert(seq->name != NULL);
	strcpy(seq->name, name);

	seq->st_cnt = 0;
	seq->st = NULL;

	//DBG(DBG_INFO, "\"%s\"", seq->name);

	return seq;
}


//---------------------------------------------------------------------
//---------------------------------------------------------------------

int pore_model_entries_read(struct pore_model * pm, FILE * f)
{
	unsigned int i;

	pm->pm_entry = (struct model_entry *)malloc(N_STATES *
												sizeof(struct model_entry));
	assert(pm->pm_entry != NULL);

	for (i = 0; i < N_STATES; ++i) {
		struct __attribute__((packed)) {
			double level_mean;
			double level_stdv;
			double sd_mean;
			double sd_stdv;
		} buf;
		struct model_entry * e = &pm->pm_entry[i];

		if (fread(&buf, sizeof(buf), 1, f) != 1) {
			//DBG(DBG_WARNING, "fread() failed!");
			cout<<"fread() failed!"<<endl;
			return -1;
		}

//		DBG(DBG_INFO, "%4d %f %f %f %f",
//			i, buf.level_mean, buf.level_stdv, buf.sd_mean, buf.sd_stdv);

		e->level_mean = buf.level_mean;
		e->level_stdv = buf.level_stdv;
		e->sd_mean = buf.sd_mean;
		e->sd_stdv = buf.sd_stdv;
	}

	return 0;
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------


int strand_events_read(struct strand * st, FILE * f)
{
	struct __attribute__((packed)) {
		char magic[8];
		char id[51];
		uint8_t strand;
		uint32_t cnt;
	} ev_hdr;
	unsigned int i;
	uint32_t cnt;

	if (fread(&ev_hdr, sizeof(ev_hdr), 1, f) != 1)
		return -1;

	if (strncmp(ev_hdr.magic, "EVENTS", 8) != 0) {
		//DBG(DBG_WARNING, "invalid section");
		cout<<"invalid section"<<endl;
		return -1;
	}

	/* Strand id */
	st->id = (char *)malloc(strlen(ev_hdr.id) + 1);
	assert(st->id != NULL);
	strcpy(st->id, ev_hdr.id);
	st->no = ev_hdr.strand;
	cnt = ev_hdr.cnt;

	//DBG(DBG_INFO, "\"%s\".%d (%d)", st->id, st->no, cnt);
	cout<<"st->id:"<<st->id<< "   st->no:"<< st->no<< "   cnt:"<< cnt<<endl;

	assert(st->ev == NULL);
	st->ev = (struct event_entry *)malloc(cnt * sizeof(struct event_entry));
	assert(st->ev != NULL);
	st->ev_cnt = cnt;

	for (i = 0; i < cnt; ++i) {
		struct __attribute__((packed)) {
			double start;
			double length;
			double mean;
			double stdv;
		} buf;
		struct event_entry * e;

		if (fread(&buf, sizeof(buf), 1, f) != 1) {
			//DBG(DBG_WARNING, "fread() failed!");
			cout<<"fread() failed!"<<endl;
			return -1;
		}

		e = &st->ev[i];
		e->mean = buf.mean;
		e->stdv = buf.stdv;
		e->start = buf.start;
		e->length = buf.length;
	}

	/* Load complete, return 1 */
	return 1;
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------

int sequence_strand_read(struct sequence * seq, FILE * f)
{

	struct __attribute__((packed)) {
		char magic[8];
		uint32_t cnt;
	} pm_hdr;
	struct strand * st;
	unsigned int i;
	int ret;

	if (feof(f))
		return 0;

	if (fread(&pm_hdr, sizeof(pm_hdr), 1, f) != 1) {
		if (feof(f))
			return 0;
		//DBG(DBG_WARNING, "fread() failed!");
		return -1;
	}

	if (strncmp(pm_hdr.magic, "POREMOD", 8) != 0) {
		//DBG(DBG_WARNING, "invalid section");
		return -1;
	}

	//DBG(DBG_INFO, "loading sequence...");
	cout<<"loading sequence..."<<endl;

	if (seq->st == NULL) {
		//DBG(DBG_INFO, "1.");
		assert(seq->st_cnt == 0);
		seq->st = (struct strand *)malloc(sizeof(struct strand));
	} else {
		//DBG(DBG_INFO, "2.");
		assert(seq->st_cnt > 0);
		seq->st = (struct strand *)realloc(seq->st, (seq->st_cnt + 1) *
										   sizeof(struct strand));
	}
	assert(seq->st != NULL);
	st = &seq->st[seq->st_cnt++];
	memset(st, 0, sizeof(struct strand));

	for (i = 0; i < pm_hdr.cnt; ++i) {
		struct __attribute__((packed)) {
			char name[32];
			double scale;
			double shift;
			double drift;
			double var;
			double scale_sd;
			double var_sd;
			double p_skip;
			double p_stay;
		} buf;
		struct pore_model * pm;

		if (fread(&buf, sizeof(buf), 1, f) != 1) {
			//DBG(DBG_WARNING, "fread() failed!");
			cout<<"fread() failed!"<<endl;
			return -1;
		}

		//DBG(DBG_INFO, "loading pore model...");
		cout<<"loading pore model..."<<endl;

		if (st->pm == NULL) {
			//DBG(DBG_INFO, "1.");
			assert(st->pm_cnt == 0);
			st->pm = (struct pore_model *)malloc(sizeof(struct pore_model));
		} else {
			//DBG(DBG_INFO, "2.");
			assert(st->pm_cnt > 0);
			st->pm = (struct pore_model *)realloc(st->pm, (st->pm_cnt + 1) *
												  sizeof(struct pore_model));
		}
		//DBG(DBG_INFO, "3.");
		assert(st->pm != NULL);
		//DBG(DBG_INFO, "4.");
		pm = &st->pm[st->pm_cnt++];
		memset(pm, 0, sizeof(struct pore_model));

		/* Pore model name */
		pm->name = (char *)malloc(strlen(buf.name) + 1);
		assert(pm->name != NULL);
		strcpy(pm->name, buf.name);

		//DBG(DBG_INFO, "\"%s\"", pm->name);
		cout<<pm->name<<endl;

		pm->pm_param = (struct model_params *)malloc(sizeof(struct
															model_params));
		assert(pm->pm_param != NULL);
		/* Pore model parameters */
		pm->pm_param->drift = buf.drift ;
		pm->pm_param->scale = buf.scale ;
		pm->pm_param->scale_sd = buf.scale_sd ;
		pm->pm_param->shift = buf.shift ;
		pm->pm_param->var = buf.var ;
		pm->pm_param->var_sd = buf.var_sd ;

		pm->st_param = (struct st_params *)malloc(sizeof(struct st_params));
		assert(pm->st_param != NULL);
		/* State transition parameters */
		pm->st_param->p_skip = buf.p_skip;
		pm->st_param->p_stay = buf.p_stay;

		//DBG(DBG_INFO, "pstay=%f p_skip=%f",
		//	pm->st_param->p_stay, pm->st_param->p_skip);
		cout<<"p_stay="<< pm->st_param->p_stay << "   p_skip=" << pm->st_param->p_skip <<endl;

		//DBG(DBG_MSG, "loading pore model entries...");
		cout<<"loading pore model entries..."<<endl;

		if ((ret = pore_model_entries_read(pm, f)) < 0){
			cout<<"ret="<<ret;
			return ret;
		}

	}

	//DBG(DBG_INFO, "loading strand events...");
	cout<<"loading strand events..."<<endl;


	return strand_events_read(st, f);


}


//---------------------------------------------------------------------
//---------------------------------------------------------------------
struct sequence * sequence_from_bin_file(const char * fname)
{
	cout<<"in sequence bin file"<<endl;
	struct __attribute__((packed)) {
		char magic[8];
		char name[120];
	} sq_hdr;
	struct sequence * seq;
	FILE * f;

	if (!(f = fopen(fname, "rb"))) {
		//DBG(DBG_WARNING, "can't open file \"%s\"\n", fname);
		return NULL;
	};

	if (fread(&sq_hdr, sizeof(sq_hdr), 1, f) != 1) {
		//DBG(DBG_WARNING, "invalid file");
		fclose(f);
		return NULL;
	}

	if (strncmp(sq_hdr.magic, "DNASEQ", 8) != 0) {
		//DBG(DBG_WARNING, "invalid file header");
		fclose(f);
		return NULL;
	}

	seq = sequence_alloc(sq_hdr.name);

	while (sequence_strand_read(seq, f) > 0);

	fclose(f);

	return seq;
}
void sequence_free(struct sequence * seq)
{
	int i;
	int j;

	assert(seq != NULL);

	//DBG(DBG_INFO, ".1");
	assert(seq->name != NULL);
	free(seq->name);

	if (seq->st_cnt > 0) {
		assert(seq->st != NULL);

		for (i = 0; i < seq->st_cnt; ++i) {
			assert(seq->st[i].id != NULL);
			free(seq->st[i].id);

			if (seq->st[i].pm_cnt) {
				assert(seq->st[i].pm != NULL);
				for (j = 0; j < seq->st[i].pm_cnt; ++j) {
					assert(seq->st[i].pm[j].name != NULL);
					free(seq->st[i].pm[j].name);

					assert(seq->st[i].pm[j].pm_param != NULL);
					free(seq->st[i].pm[j].pm_param);

					assert(seq->st[i].pm[j].st_param != NULL);
					free(seq->st[i].pm[j].st_param);

					assert(seq->st[i].pm[j].pm_entry != NULL);
					free(seq->st[i].pm[j].pm_entry);
				}
				free(seq->st[i].pm);
			}

			if (seq->st[i].ev_cnt) {
				assert(seq->st[i].ev != NULL);
				free(seq->st[i].ev);
			}
		}
		free(seq->st);
	}

	free(seq);
}




#endif
