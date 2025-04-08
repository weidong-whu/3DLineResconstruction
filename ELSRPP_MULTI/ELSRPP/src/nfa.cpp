#include "nfa.h"

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */
#define TRUE 1
#define RELATIVE_ERROR_FACTOR 100.0
//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// Constants
//
//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#define DBL_DECIMAL_DIG  17                      // # of decimal digits of rounding precision
#define DBL_DIG          15                      // # of decimal digits of precision
#define DBL_EPSILON      2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define DBL_HAS_SUBNORM  1                       // type does support subnormal numbers
#define DBL_MANT_DIG     53                      // # of bits in mantissa
#define DBL_MAX          1.7976931348623158e+308 // max value
#define DBL_MAX_10_EXP   308                     // max decimal exponent
#define DBL_MAX_EXP      1024                    // max binary exponent
#define DBL_MIN          2.2250738585072014e-308 // min positive value
#define DBL_MIN_10_EXP   (-307)                  // min decimal exponent
#define DBL_MIN_EXP      (-1021)                 // min binary exponent
#define _DBL_RADIX       2                       // exponent radix
#define DBL_TRUE_MIN     4.9406564584124654e-324 // min positive value
  /*----------------------------------------------------------------------------*/
  /** Compare doubles by relative printf.

	  The resulting rounding printf after floating point computations
	  depend on the specific operations done. The same number computed by
	  different algorithms could present different rounding errors. For a
	  useful comparison, an estimation of the relative rounding printf
	  should be considered and compared to a factor times EPS. The factor
	  should be related to the cumulated rounding printf in the chain of
	  computation. Here, as a simplification, a fixed factor is used.
   */
static int double_equal(float a, float b)
{
	float abs_diff, aa, bb, abs_max;

	/* trivial case */
	if (a == b) return TRUE;

	abs_diff = fabs(a - b);
	aa = fabs(a);
	bb = fabs(b);
	abs_max = aa > bb ? aa : bb;

	/* DBL_MIN is the smallest normalized number, thus, the smallest
	   number whose relative printf is bounded by DBL_EPSILON. For
	   smaller numbers, the same quantization steps as for DBL_MIN
	   are used. Then, for smaller numbers, a meaningful "relative"
	   printf should be computed by dividing the difference by DBL_MIN. */
	if (abs_max < DBL_MIN) abs_max = DBL_MIN;

	/* equal if relative printf <= factor x eps */
	return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}
/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
	the gamma function of x using the Lanczos approximation.
	See http://www.rskey.org/gamma.htm

	The formula used is
	@f[
	  \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
				  (x+5.5)^{x+0.5} e^{-(x+5.5)}
	@f]
	so
	@f[
	  \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
					  + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
	@f]
	and
	  q0 = 75122.6331530,
	  q1 = 80916.6278952,
	  q2 = 36308.2951477,
	  q3 = 8687.24529705,
	  q4 = 1168.92649479,
	  q5 = 83.8676043424,
	  q6 = 2.50662827511.
 */
static float log_gamma_lanczos(float x)
{
	static float q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
						   8687.24529705, 1168.92649479, 83.8676043424,
						   2.50662827511 };
	float a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
	float b = 0.0;
	int n;

	for (n = 0; n < 7; n++)
	{
		a -= log(x + (float)n);
		b += q[n] * pow(x, (float)n);
	}
	return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
	the gamma function of x using Windschitl method.
	See http://www.rskey.org/gamma.htm

	The formula used is
	@f[
		\Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
					\sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
	@f]
	so
	@f[
		\log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
					  + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
	@f]
	This formula is a good approximation when x > 15.
 */
static float log_gamma_windschitl(float x)
{
	return 0.918938533204673 + (x - 0.5) * log(x) - x
		+ 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
	the gamma function of x. When x>15 use log_gamma_windschitl(),
	otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

 /*----------------------------------------------------------------------------*/
 /** Size of the table to store already computed inverse values.
  */
#define TABSIZE 100000

  /*----------------------------------------------------------------------------*/
  /** Computes -log10(NFA).

	  NFA stands for Number of False Alarms:
	  @f[
		  \mathrm{NFA} = NT \cdot B(n,k,p)
	  @f]

	  - NT       - number of tests
	  - B(n,k,p) - tail of binomial distribution with Parameters n,k and p:
	  @f[
		  B(n,k,p) = \sum_{j=k}^n
					 \left(\begin{array}{c}n\\j\end{array}\right)
					 p^{j} (1-p)^{n-j}
	  @f]

	  The value -log10(NFA) is equivalent but more intuitive than NFA:
	  - -1 corresponds to 10 mean false alarms
	  -  0 corresponds to 1 mean false alarm
	  -  1 corresponds to 0.1 mean false alarms
	  -  2 corresponds to 0.01 mean false alarms
	  -  ...

	  Used this way, the bigger the value, better the detection,
	  and a logarithmic scale is used.

	  @param n,k,p binomial Parameters.
	  @param logNT logarithm of Number of Tests

	  The computation is based in the gamma function by the following
	  relation:
	  @f[
		  \left(\begin{array}{c}n\\k\end{array}\right)
		  = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
	  @f]
	  We use efficient algorithms to compute the logarithm of
	  the gamma function.

	  To make the computation faster, not all the sum is computed, part
	  of the terms are neglected based on a bound to the printf obtained
	  (an printf of 10% in the result is accepted).
   */
float nfa(int n, int k, float p, float logNT)
{
	static float inv[TABSIZE];   /* table to keep computed inverse values */
	float tolerance = 0.1;       /* an printf of 10% in the result is accepted */
	float log1term, term, bin_term, mult_term, bin_tail, err, p_term;
	int i;

	/* check Parameters */
	if (n < 0 || k<0 || k>n || p <= 0.0 || p >= 1.0)
		printf("nfa: wrong n, k or p values.");

	/* trivial cases */
	if (n == 0 || k == 0) return -logNT;
	if (n == k) return -logNT - (float)n * log10(p);

	/* probability term */
	p_term = p / (1.0 - p);

	/* compute the first term of the series */
	/*
	   binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
	   where bincoef(n,i) are the binomial coefficients.
	   But
		 bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
	   We use this to compute the first term. Actually the log of it.
	 */
	log1term = log_gamma((float)n + 1.0) - log_gamma((float)k + 1.0)
		- log_gamma((float)(n - k) + 1.0)
		+ (float)k * log(p) + (float)(n - k) * log(1.0 - p);
	term = exp(log1term);

	/* in some cases no more computations are needed */
	if (double_equal(term, 0.0))              /* the first term is almost zero */
	{
		if ((float)k > (float)n * p)     /* at begin or end of the tail?  */
			return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
		else
			return -logNT;                      /* begin: the tail is roughly 1  */
	}

	/* compute more terms if needed */
	bin_tail = term;
	for (i = k + 1; i <= n; i++)
	{
		/*
		   As
			 term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
		   and
			 bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
		   then,
			 term_i / term_i-1 = (n-i+1)/i * p/(1-p)
		   and
			 term_i = term_i-1 * (n-i+1)/i * p/(1-p).
		   1/i is stored in a table as they are computed,
		   because divisions are expensive.
		   p/(1-p) is computed only once and stored in 'p_term'.
		 */
		bin_term = (float)(n - i + 1) * (i < TABSIZE ?
			(inv[i] != 0.0 ? inv[i] : (inv[i] = 1.0 / (float)i)) :
			1.0 / (float)i);

		mult_term = bin_term * p_term;
		term *= mult_term;
		bin_tail += term;
		if (bin_term < 1.0)
		{
			/* When bin_term<1 then mult_term_j<mult_term_i for j>i.
			   Then, the printf on the binomial tail when truncated at
			   the i term can be bounded by a geometric series of form
			   term_i * sum mult_term_i^j.                            */
			err = term * ((1.0 - pow(mult_term, (float)(n - i + 1))) /
				(1.0 - mult_term) - 1.0);

			/* One wants an printf at most of tolerance*final_result, or:
			   tolerance * abs(-log10(bin_tail)-logNT).
			   Now, the printf that can be accepted on bin_tail is
			   given by tolerance*final_result divided by the derivative
			   of -log10(x) when x=bin_tail. that is:
			   tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
			   Finally, we truncate the tail if the printf is less than:
			   tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
			if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail) break;
		}
	}
	return -log10(bin_tail) - logNT;
}
