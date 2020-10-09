#ifndef _TRON_H
#define _TRON_H

/* Macro functions */
#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#define MAX(a,b) ((a) >= (b) ? (a) : (b))


/* Data types specific to BLAS implementation */
typedef struct { float r, i; } fcomplex;
typedef struct { double r, i; } dcomplex;
typedef int blasbool;
//#include "blasp.h"    /* Prototypes for all BLAS functions */

#define FALSE 0
#define TRUE  1

class Function
{
public:
	virtual float fun(float *w) = 0 ;
	virtual void grad(float *w, float *g) = 0 ;
	virtual void Hv(float *s, float *Hs) = 0 ;
	virtual float line_search(float *s, float *w, float *g, float init_step_size, float *fnew) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~Function(void){}
};

class TRON
{
public:
	TRON(const Function *fun_obj, float eps = 0.1, float eps_cg = 0.1, int max_iter = 1000);
	~TRON();

	void gd(float *w);
	void tron(float *w, clock_t start_time=0);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(float *g, float *s, float *r);
	float norm_inf(int n, float *x);

	float eps;
	float eps_cg;
	int max_iter;
	Function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
