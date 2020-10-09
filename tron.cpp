#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "tron.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
extern float dnrm2_(int *, float *, int *);
extern float ddot_(int *, float *, int *, float *, int *);
extern int daxpy_(int *, float *, float *, int *, float *, int *);
extern int dscal_(int *, float *, float *, int *);
*/

#ifdef __cplusplus
}
#endif

float ddot_(int *n, float *sx, int *incx, float *sy, int *incy)
{
  long int i, m, nn, iincx, iincy;
  float stemp;
  long int ix, iy;

  /* forms the dot product of two vectors.
     uses unrolled loops for increments equal to one.
     jack dongarra, linpack, 3/11/78.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;
  iincy = *incy;

  stemp = 0.0;
  if (nn > 0)
  {
    if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
    {
      m = nn-4;
      for (i = 0; i < m; i += 5)
        stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                 sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

      for ( ; i < nn; i++)        /* clean-up loop */
        stemp += sx[i] * sy[i];
    }
    else /* code for unequal increments or equal increments not equal to 1 */
    {
      ix = 0;
      iy = 0;
      if (iincx < 0)
        ix = (1 - nn) * iincx;
      if (iincy < 0)
        iy = (1 - nn) * iincy;
      for (i = 0; i < nn; i++)
      {
        stemp += sx[ix] * sy[iy];
        ix += iincx;
        iy += iincy;
      }
    }
  }

  return stemp;
} /* ddot_ */

int dscal_(int *n, float *sa, float *sx, int *incx)
{
  long int i, m, nincx, nn, iincx;
  float ssa;

  /* scales a vector by a constant.
     uses unrolled loops for increment equal to 1.
     jack dongarra, linpack, 3/11/78.
     modified 3/93 to return if incx .le. 0.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;
  ssa = *sa;

  if (nn > 0 && iincx > 0)
  {
    if (iincx == 1) /* code for increment equal to 1 */
    {
      m = nn-4;
      for (i = 0; i < m; i += 5)
      {
        sx[i] = ssa * sx[i];
        sx[i+1] = ssa * sx[i+1];
        sx[i+2] = ssa * sx[i+2];
        sx[i+3] = ssa * sx[i+3];
        sx[i+4] = ssa * sx[i+4];
      }
      for ( ; i < nn; ++i) /* clean-up loop */
        sx[i] = ssa * sx[i];
    }
    else /* code for increment not equal to 1 */
    {
      nincx = nn * iincx;
      for (i = 0; i < nincx; i += iincx)
        sx[i] = ssa * sx[i];
    }
  }

  return 0;
} /* dscal_ */


int daxpy_(int *n, float *sa, float *sx, int *incx, float *sy,
           int *incy)
{
  long int i, m, ix, iy, nn, iincx, iincy;
  register float ssa;

  /* constant times a vector plus a vector.
     uses unrolled loop for increments equal to one.
     jack dongarra, linpack, 3/11/78.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  ssa = *sa;
  iincx = *incx;
  iincy = *incy;

  if( nn > 0 && ssa != 0.0 )
  {
    if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
    {
      m = nn-3;
      for (i = 0; i < m; i += 4)
      {
        sy[i] += ssa * sx[i];
        sy[i+1] += ssa * sx[i+1];
        sy[i+2] += ssa * sx[i+2];
        sy[i+3] += ssa * sx[i+3];
      }
      for ( ; i < nn; ++i) /* clean-up loop */
        sy[i] += ssa * sx[i];
    }
    else /* code for unequal increments or equal increments not equal to 1 */
    {
      ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
      iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
      for (i = 0; i < nn; i++)
      {
        sy[iy] += ssa * sx[ix];
        ix += iincx;
        iy += iincy;
      }
    }
  }

  return 0;
} /* daxpy_ */


float dnrm2_(int *n, float *x, int *incx)
{
  long int ix, nn, iincx;
  float norm, scale, absxi, ssq, temp;

/*  DNRM2 returns the euclidean norm of a vector via the function
    name, so that
       DNRM2 := sqrt( x'*x )
    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.   */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;

  if( nn > 0 && iincx > 0 )
  {
    if (nn == 1)
    {
      norm = fabs(x[0]);
    }
    else
    {
      scale = 0.0;
      ssq = 1.0;

      /* The following loop is equivalent to this call to the LAPACK
         auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

      for (ix=(nn-1)*iincx; ix>=0; ix-=iincx)
      {
        if (x[ix] != 0.0)
        {
          absxi = fabs(x[ix]);
          if (scale < absxi)
          {
            temp = scale / absxi;
            ssq = ssq * (temp * temp) + 1.0;
            scale = absxi;
          }
          else
          {
            temp = absxi / scale;
            ssq += temp * temp;
          }
        }
      }
      norm = scale * sqrt(ssq);
    }
  }
  else
    norm = 0.0;

  return norm;

} /* dnrm2_ */


static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const Function *fun_obj, float eps, float eps_cg, int max_iter)
{
	this->fun_obj=const_cast<Function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::gd(float *w)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	float step_size, one=1.0;
	float f, fnew, actred;
	float init_step_size = 1;
	int search = 1, iter = 1, inc = 1;
	float *s = new float[n];
	float *r = new float[n];
	float *g = new float[n];


	float *w0 = new float[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	float gnorm0 = dnrm2_(&n, g, &inc);
	delete [] w0;
    w0 = nullptr;
	//printf("eps = %.16e, |g0| = %.16e\n", eps, gnorm0);
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	float gnorm = dnrm2_(&n, g, &inc);

	//printf("initial gnorm: %4.8e, f: %4.8e\n",gnorm, f);

	if (gnorm <= eps*gnorm0)
		search = 0;

	iter = 1;
	// calculate gradient norm at w=0 for stopping condition.
	//float *w_new = new float[n];

	//constant stepsize:
	float L = 0.25 + 1.0;
	step_size = 1/L;
	while (iter <= max_iter && search)
	{
		//memcpy(w_new, w, sizeof(float)*n);
		//daxpy_(&n, &one, s, &inc, w_new, &inc);

		//clock_t line_time = clock();

		for(int i=0; i<n; i++)
			w[i] -= step_size*g[i];
			//s[i] = -g[i];
		//step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew);  //fangh comment out


		//printf("stepsize: %1.3e\n", step_size);
		//line_time = clock() - line_time;
		//actred = f - fnew;

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}
		daxpy_(&n, &step_size, s, &inc, w, &inc);
		//clock_t t = clock();
		//float snorm = dnrm2_(&n, s, &inc);
		//info("iter %2d f %5.10e |g| %5.10e CG %3d step_size %5.3e snorm %5.10e cg_time %f line_time %f time %f \n", iter, f, gnorm, cg_iter, step_size, snorm
		//	,(float(cg_time)/CLOCKS_PER_SEC), (float(line_time)/CLOCKS_PER_SEC), (float(t-start_time))/CLOCKS_PER_SEC);

		f = fnew;
		iter++;

		fun_obj->grad(w, g);

		gnorm = dnrm2_(&n, g, &inc);
		//printf("gnorm: %4.8e, f: %4.8e\n",gnorm, f);
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
	}
	printf("num iter: %i\n", iter);
	//printf("time: %f\n", (float(t-start_time))/CLOCKS_PER_SEC );

	delete[] g;
    g = nullptr;
	delete[] r;
    r = nullptr;
	//delete[] w_new;
	delete[] s;
    s = nullptr;
}


void TRON::tron(float *w, clock_t start_time)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	float step_size, one=1.0;
	float f, fnew, actred;
	float init_step_size = 1;
	int search = 1, iter = 1, inc = 1;
	float *s = new float[n];
	float *r = new float[n];
	float *g = new float[n];

	// calculate gradient norm at w=0 for stopping condition.
	float *w0 = new float[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	float gnorm0 = dnrm2_(&n, g, &inc);
	delete [] w0;
    w0 = nullptr;
	//printf("eps = %.16e, |g0| = %.16e\n", eps, gnorm0);
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	float gnorm = dnrm2_(&n, g, &inc);

	//printf("initial gnorm: %4.8e, f: %4.8e\n",gnorm, f);
	if (gnorm <= eps*gnorm0)
		search = 0;

	iter = 1;

	float *w_new = new float[n];
	while (iter <= max_iter && search)
	{
		clock_t cg_time = clock();
		cg_iter = trcg(g, s, r);
		cg_time = clock() - cg_time;

		memcpy(w_new, w, sizeof(float)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		clock_t line_time = clock();
		step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew);
		line_time = clock() - line_time;
		actred = f - fnew;

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}
		daxpy_(&n, &step_size, s, &inc, w, &inc);
		clock_t t = clock();
        float snorm = dnrm2_(&n, s, &inc);
		//info("iter %2d f %5.10e |g| %5.10e CG %3d step_size %5.3e snorm %5.10e cg_time %f line_time %f time %f \n", iter, f, gnorm, cg_iter, step_size, snorm
		//	,(float(cg_time)/CLOCKS_PER_SEC), (float(line_time)/CLOCKS_PER_SEC), (float(t-start_time))/CLOCKS_PER_SEC);

		f = fnew;
		iter++;

		fun_obj->grad(w, g);

		gnorm = dnrm2_(&n, g, &inc);
		//printf("gnorm: %4.8e, f: %4.8e\n",gnorm, f);
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		/*
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}*/
	}
	clock_t t = clock();
	printf("num iter: %i\n", iter);
	//printf("time: %f\n", (float(t-start_time))/CLOCKS_PER_SEC );

	delete[] g;
    g = nullptr;
	delete[] r;
    r = nullptr;
	delete[] w_new;
    w_new = nullptr;
	delete[] s;
    s = nullptr;
}

int TRON::trcg(float *g, float *s, float *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	float one = 1;
	float *d = new float[n];
	float *Hd = new float[n];
	float rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = eps_cg*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
    d = nullptr;
	delete[] Hd;
    Hd = nullptr;

	return(cg_iter);
}

float TRON::norm_inf(int n, float *x)
{
	float dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
