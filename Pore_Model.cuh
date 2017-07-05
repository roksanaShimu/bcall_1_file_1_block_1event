/*
 * Pore_Model.cuh
 *
 *  Created on: Dec 23, 2016
 *      Author: roksana
 */

#ifndef __PORE_MODEL_CUH
#define __PORE_MODEL_CUH

#include<iostream>



#define __BCALL_I__
#include "bcall-i.h"

using namespace std;


static bcfloat_t pm_mean;
static bcfloat_t pm_stdv;

// update sd_lambda based on sd_mean & sd_stdv
static void update_sd_lambda(struct pm_state * st)
{
	st->sd_lambda = pow(st->sd_mean, 3.0) / pow(st->sd_stdv, 2.0);
}

// update sd_stdv based on sd_mean & sd_lambda
static void update_sd_stdv(struct pm_state * st)
{
	st->sd_stdv = pow(pow(st->sd_mean, 3.0) / st->sd_lambda, .5);
}

// update logs
static void update_logs(struct pm_state * st)
{
	st->log_level_mean = log(st->level_mean);
	st->log_level_stdv = log(st->level_stdv);
	st->log_sd_mean = log(st->sd_mean);
	st->log_sd_lambda = log(st->sd_lambda);
}


static void update_statistics(struct pm_state pm_state[])
{
	unsigned int n;
	bcfloat_t s = 0.0;
	bcfloat_t s2 = 0.0;
	bcfloat_t mean;
	bcfloat_t stdv;

	for (n = 0; n < N_STATES; ++n) {
		struct pm_state * st = &pm_state[n];
		bcfloat_t x = st->level_mean;
		s += x;
		s2 += x * x;
	}

	mean = s / n;
	stdv = sqrt((s2 - s * mean * 2.0 + mean * mean * (bcfloat_t)n) / (n - 1));

	pm_mean = mean;
	pm_stdv = stdv;
}



//int pore_model_read(FILE * f, struct pm_state pm_state[])

int pore_model_read(FILE * f)
{
	pm_state pm_state[N_STATES];
	char magic[8];
	struct __attribute__((packed)) {
		double level_mean;
		double level_stdv;
		double sd_mean;
		double sd_stdv;
	} buf;
	unsigned int i;

	if (fread(&magic, sizeof(magic), 1, f) != 1)
		return -1;

	if (strncmp(magic, "POREMOD", 8) != 0) {
		//DBG(DBG_WARNING, "invalid section");
		return -1;
	}

	for (i = 0; i < N_STATES; ++i) {
		struct pm_state * st = &pm_state[i];

		if (fread(&buf, sizeof(buf), 1, f) != 1)
			break;

		//DBG(DBG_MSG, "%4d %f %f %f %f",
		//	i, buf.level_mean, buf.level_stdv, buf.sd_mean, buf.sd_stdv);

		st->level_mean = buf.level_mean;
		st->level_stdv = buf.level_stdv;
		st->sd_mean = buf.sd_mean;
		st->sd_stdv = buf.sd_stdv;
		update_sd_lambda(st);
		update_logs(st);
	}

	//DBG(DBG_INFO, "entries=%d", i);

	assert(i == N_STATES);

	update_statistics(pm_state);

	return 0;
}

void pm_state_init(struct pm_state * st, const struct model_entry * e)
{
	st->level_mean = e->level_mean;
	st->level_stdv = e->level_stdv;
	st->sd_mean = e->sd_mean;
	st->sd_stdv = e->sd_stdv;
	update_sd_lambda(st);
	update_logs(st);
}

void pore_model_init(struct pm_state pm_state[],
					 const struct model_entry me[], unsigned int len)
{
	int i;

	assert(len >= N_STATES);

	for (i = 0; i < N_STATES; ++i) {
		struct pm_state * st = &pm_state[i];

		pm_state_init(st, &me[i]);
	}

	update_statistics(pm_state);
}

void pm_state_scale(struct pm_state * st, const struct model_params * params)
{
	// these functions are provided by ONT
	st->level_mean = st->level_mean * params->scale + params->shift;
	st->level_stdv = st->level_stdv * params->var;
	st->sd_mean = st->sd_mean * params->scale_sd;
	st->sd_lambda = st->sd_lambda * params->var_sd;
	update_sd_stdv(st);
	update_logs(st);
}

void pore_model_scale(struct pm_state pm_state[], struct model_params * param)
{
	int i;

	//DBG(DBG_INFO, "scale=%f shift=%f drift=%f var=%f scale_sd=%f var_sd=%f",
	//	param->scale, param->shift, param->drift, param->var,
	//	param->scale_sd, param->var_sd);
	cout<< "scale="<< param->scale<< "   shift="<< param->shift<< "    drift="<< param->drift<< "    var="<< param->var<< "     scale_sd=" << param->scale_sd<<  "    var_sd="<< param->var_sd<<endl;

	for (i = 0; i < N_STATES; ++i) {
		struct pm_state * st = &pm_state[i];

		pm_state_scale(st, param);
	}

	update_statistics(pm_state);
}




#endif
