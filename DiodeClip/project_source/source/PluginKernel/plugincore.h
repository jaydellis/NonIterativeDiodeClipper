// -----------------------------------------------------------------------------
//    ASPiK Plugin Kernel File:  plugincore.h
//
/**
    \file   plugincore.h
    \author Will Pirkle
    \date   17-September-2018
    \brief  base class interface file for ASPiK plugincore object
    		- http://www.aspikplugins.com
    		- http://www.willpirkle.com
*/
// -----------------------------------------------------------------------------
#ifndef __pluginCore_h__
#define __pluginCore_h__

#include "pluginbase.h"

#include "rchundenormal.h"
#include "Iir.h"

#include "fxobjects.h"
#include "nonlinmodels.h"

#include <emmintrin.h>


// **--0x7F1F--**

enum controlID {
	inputgain = 0,
	outputgain = 1,
	resistor = 2,
	alpha = 3,
	capacitor = 4,
	satcurrent = 5,
	thermalvoltage = 6,
	hipass = 7,
	hipasscutoff = 8,
	assymetry = 9,
	oversampling = 10,
	ssecontrol = 11,
	ampgain = 12,
	ampVt = 13,
	cathodeR = 14,
	threshold = 15,
	overbias = 16,
	dccutoff = 17,
	lowshelfsag = 18,
	lowshelfreq = 19,
	fbthresh = 20,
	release = 21,
	sagVol = 22,
	sagVt = 23,
	sagLP = 24,
	sagIntrim = 25
};


// **--0x0F1F--**

/**
\class PluginCore
\ingroup ASPiK-Core
\brief
The PluginCore object is the default PluginBase derived object for ASPiK projects.
Note that you are fre to change the name of this object (as long as you change it in the compiler settings, etc...)


PluginCore Operations:
- overrides the main processing functions from the base class
- performs reset operation on sub-modules
- processes audio
- processes messages for custom views
- performs pre and post processing functions on parameters and audio (if needed)

\author Will Pirkle http://www.willpirkle.com
\remark This object is included in Designing Audio Effects Plugins in C++ 2nd Ed. by Will Pirkle
\version Revision : 1.0
\date Date : 2018 / 09 / 7
*/
class PluginCore : public PluginBase
{
public:
    PluginCore();

	/** Destructor: empty in default version */
    virtual ~PluginCore(){}

	// --- PluginBase Overrides ---
	//
	/** this is the creation function for all plugin parameters */
	bool initPluginParameters();

	/** called when plugin is loaded, a new audio file is playing or sample rate changes */
	virtual bool reset(ResetInfo& resetInfo);

	/** one-time post creation init function; pluginInfo contains path to this plugin */
	virtual bool initialize(PluginInfo& _pluginInfo);

	/** preProcess: sync GUI parameters here; override if you don't want to use automatic variable-binding */
	virtual bool preProcessAudioBuffers(ProcessBufferInfo& processInfo);

	/** process frames of data (DEFAULT MODE) */
	virtual bool processAudioFrame(ProcessFrameInfo& processFrameInfo);

	/** Pre-process the block with: MIDI events for the block and parametet smoothing */
	virtual bool preProcessAudioBlock(IMidiEventQueue* midiEventQueue = nullptr);

	/** process sub-blocks of data (OPTIONAL MODE) */
	virtual bool processAudioBlock(ProcessBlockInfo& processBlockInfo);

	/** This is the native buffer processing function; you may override and implement
	     it if you want to roll your own buffer or block procssing code */
	// virtual bool processAudioBuffers(ProcessBufferInfo& processBufferInfo);

	/** preProcess: do any post-buffer processing required; default operation is to send metering data to GUI  */
	virtual bool postProcessAudioBuffers(ProcessBufferInfo& processInfo);

	/** called by host plugin at top of buffer proccess; this alters parameters prior to variable binding operation  */
	virtual bool updatePluginParameter(int32_t controlID, double controlValue, ParameterUpdateInfo& paramInfo);

	/** called by host plugin at top of buffer proccess; this alters parameters prior to variable binding operation  */
	virtual bool updatePluginParameterNormalized(int32_t controlID, double normalizedValue, ParameterUpdateInfo& paramInfo);

	/** this can be called: 1) after bound variable has been updated or 2) after smoothing occurs  */
	virtual bool postUpdatePluginParameter(int32_t controlID, double controlValue, ParameterUpdateInfo& paramInfo);

	/** this is ony called when the user makes a GUI control change */
	virtual bool guiParameterChanged(int32_t controlID, double actualValue);

	/** processMessage: messaging system; currently used for custom/special GUI operations */
	virtual bool processMessage(MessageInfo& messageInfo);

	/** processMIDIEvent: MIDI event processing */
	virtual bool processMIDIEvent(midiEvent& event);

	/** specialized joystick servicing (currently not used) */
	virtual bool setVectorJoystickParameters(const VectorJoystickData& vectorJoysickData);



	
	/** create the presets */
	bool initPluginPresets();

	// --- example block processing template functions (OPTIONAL)
	//
	/** FX EXAMPLE: process audio by passing through */
	bool renderFXPassThrough(ProcessBlockInfo& blockInfo);

	/** SYNTH EXAMPLE: render a block of silence */
	bool renderSynthSilence(ProcessBlockInfo& blockInfo);

	// --- BEGIN USER VARIABLES AND FUNCTIONS -------------------------------------- //
	//	   Add your variables and methods here


	struct cascadedbiq {

	public:
		cascadedbiq(void) {};
		~cascadedbiq(void) {};

		void setparams(Iir::Cascade::Storage coefs) {

			auto stagecoefs = coefs.stageArray;
			stages = coefs.maxStages;

			for (int i = 0; i < stages; i++) {
				aa1[i] = stagecoefs[i].getA1();
				aa2[i] = stagecoefs[i].getA2();
				bb0[i] = stagecoefs[i].getB0();
				bb1[i] = stagecoefs[i].getB1();
				bb2[i] = stagecoefs[i].getB2();

				coeff[i][0] = _mm_set1_pd( stagecoefs[i].getA1() );
				coeff[i][1] = _mm_set1_pd( stagecoefs[i].getA2() );
				coeff[i][2] = _mm_set1_pd( stagecoefs[i].getB0() );
				coeff[i][3] = _mm_set1_pd( stagecoefs[i].getB1() );
				coeff[i][4] = _mm_set1_pd( stagecoefs[i].getB2() );

				coeffs[i][0] = _mm_set1_ps(stagecoefs[i].getA1());
				coeffs[i][1] = _mm_set1_ps(stagecoefs[i].getA2());
				coeffs[i][2] = _mm_set1_ps(stagecoefs[i].getB0());
				coeffs[i][3] = _mm_set1_ps(stagecoefs[i].getB1());
				coeffs[i][4] = _mm_set1_ps(stagecoefs[i].getB2());
			}

			
		}

		inline double process( double x) {

			for (int i = 0; i < stages; i++) {

				out[i] = x * bb0[i] + state1[i];
				state1[i] = x * bb1[i] + state2[i] - aa1[i] * out[i];
				state2[i] = x * bb2[i] - out[i] * aa2[i];
				x = out[i];

	//			x = procStage(x, i);
			}
			return x;
		}

			inline double procStage(const double x, const int i) {

				out[i] = x * bb0[i] + state1[i];
				state1[i] = x * bb1[i] + state2[i] - aa1[i] * out[i];
				state2[i] = x * bb2[i] - out[i] * aa2[i];

			return out[i];
		}

		 inline double processUnrolled8p(const double x) {

					 out[0]    = x * bb0[0] + state1[0];
					 state1[0] = x * bb1[0] + state2[0] - aa1[0] * out[0];
					 state2[0] = x * bb2[0] - out[0] * aa2[0];

					 out[1] =    out[0] * bb0[1] + state1[1];
					 state1[1] = out[0] * bb1[1] + state2[1] - aa1[1] * out[1];
					 state2[1] = out[0] * bb2[1] - out[1] * aa2[1];

					 out[2]    = out[1] * bb0[2] + state1[2];
					 state1[2] = out[1] * bb1[2] + state2[2] - aa1[2] * out[2];
					 state2[2] = out[1] * bb2[2] - out[2] * aa2[2];

					 out[3]    = out[2] * bb0[3] + state1[3];
					 state1[3] = out[2] * bb1[3] + state2[3] - aa1[3] * out[3];
					 state2[3] = out[2] * bb2[3] - out[3] * aa2[3];

				 return out[3];
			 }

		 inline __m128d processSSE(__m128d x) {
			 for (int i = 0; i < stages; i++) {
				 outm[i] = _mm_add_pd(_mm_mul_pd(x, coeff[i][2]), state1m[i]);
				 state1m[i] = _mm_sub_pd( _mm_add_pd( _mm_mul_pd( x , coeff[i][3] ) , state2m[i] ) , _mm_mul_pd( coeff[i][0] , outm[i] ) );
				 state2m[i] = _mm_sub_pd( _mm_mul_pd( x , coeff[i][4] ) , _mm_mul_pd( outm[i] , coeff[i][1] ) );
				 x = outm[i];
			 }
			 return x;
		 }

		 inline __m128 processSSEf(__m128 x) {

			 for (int i = 0; i < stages; i++) {

				 outf[i] = _mm_add_ps(_mm_mul_ps(x, coeffs[i][2]), state1f[i]);
				 state1f[i] = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(x, coeffs[i][3]), state2f[i]), _mm_mul_ps(coeffs[i][0], outf[i]));
				 state2f[i] = _mm_sub_ps(_mm_mul_ps(x, coeffs[i][4]), _mm_mul_ps(outf[i], coeffs[i][1]));
				 x = outf[i];
			 }
			 return x;
		 }


	private:
		double aa1[8] = { -1.7 };
		double aa2[8] = { .8 };
		double bb0[8] = { 1. };
		double bb1[8] = { -.08 };
		double bb2[8] = { .01 };

		double state1[12] = { 0.0 };
		double state2[12] = { 0.0 };
		double out[12] = { 0.00 };

		int stages = 4;

		__m128d coeff[8][5];

		__m128 coeffs[8][5];

		__m128d outm[12] = { _mm_set1_pd(0.00) };
		__m128d state1m[12] = { _mm_set1_pd(0.0)};
		__m128d state2m[12] = { _mm_set1_pd(0.0)};

		__m128 outf[12] = { _mm_set1_ps(0.00) };
		__m128 state1f[12] = { _mm_set1_ps(0.00) };
		__m128 state2f[12] = { _mm_set1_ps(0.000) };
	};

	static inline double lagrange3xx(double x, double x1, double x2, double x3, double D) {

		const double D1 = D - 1.;   
		const double D2 = D1 - 1.;
		const double D3 = D2 - 1.;

		return ((x3 * D - x * D3) * D2 * D1) * 0.16666666666666666666666666666667 + ((x1 * D2 - x2 * D1) * D3  * D) * .5;
	}


	static inline double lagrange6xx(double x, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double D, int read, int max) {

		double D1 = D - 1.;    double D2 = D1 - 1.;
		double D3 = D2 - 1.;   double D4 = D3 - 1.;
		double D5 = D4 - 1.;   double D6 = D5 - 1.;

		double D65 = D6 * D5;
		double D12 = D1 * D2;
		double D6543 = D65 * D3 * D4;
		double DD12 = D * D12;
		double DD123 = DD12 * D3;
		double DD1234 = DD123 * D4;

		double a = x * D6543 * D12;
		double b = x1 * D6543 * (D2)* D;
		double c = x2 * D6543 * (D1)* D;
		double d = x3 * D65   * (D4)* DD12;
		double e = x4 * D65   * DD123;
		double f = x5 * (D6)* DD1234;
		double g = x6 * (D5)* DD1234;

		return		((a + g) * 0.001388888888888889
			+ (b + f) * -0.008333333333333333
			+ (c + e) * 0.02083333333333333
			+ d * -0.02777777777777778);
	}

	template <class T>			
	inline T FastAbs(const T& x) {
		T a = (x > 0.);

		return a * x + (-1. + a) * x;
	}

	template <class T>
	inline T FastMax(const T& left, const T& right)
	{
		return left > right ? left : right;
	}

	template <class T>
	inline T FastMin(const T& left, const T& right)
	{
		return left < right ? left : right;
	}

	inline double exp2a(double x) {		//https://codingforspeed.com/using-faster-exponential-approximation/
		x = 1.0 + x * 0.0009765625;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x;
		return x;
	}

	inline __m128d _exp14a(__m128d x) {		//https://codingforspeed.com/using-faster-exponential-approximation/
		x = _mm_add_pd(_mm_set1_pd(1.), _mm_mul_pd(x, _mm_set1_pd(0.00006103515625 ))); //14th order
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);
		x = _mm_mul_pd(x, x);

		return x;
	}


	//chexck http://www.chokkan.org/software/dist/fastexp.c.html todoo
	

	static inline double hermite(double t2, double t1, double t0, double t_1, double x) {

		double xx = x * x;
		double hlfx = x * .5;
		double xxx = x * xx;

		double a = ((xx - (xxx*.5)) - hlfx) * t_1;
		double b = (xxx*1.5 - xx * 2.5 + 1.0)* t0;
		double c = ((2.*x - 1.5*xxx) + hlfx) * t1;
		double d = (.5*xxx - .5*xx)*t2;

		return a + b + c + d;
	}

	double hermite2(double t2, double t1, double t0, double t_1, double x) {
	
		double a = .5*(t1 - t_1);
		double b = .5*(t2 - t0);
		double c = t0 - t_1;
		double d = a + c;
		double e = b + d;
		double f = c + e;
		double g = d + f;

		return t0 + x * (a + x * ((x * f) - g));
	}


	/* max. rel. error <= 1.72886892e-3 on [-87.33654, 88.72283] */
__m128 fast_exp_sse (__m128 x)
{
    __m128 f, p, r;
    __m128i t, j;
    const __m128 a = _mm_set1_ps (12102203.0f); /* (1 << 23) / log(2) */
    const __m128i m = _mm_set1_epi32 (0xff800000); /* mask for integer bits */
    const __m128 ttm23 = _mm_set1_ps (1.1920929e-7f); /* exp2(-23) */
    const __m128 c0 = _mm_set1_ps (0.3371894346f);
    const __m128 c1 = _mm_set1_ps (0.657636276f);
    const __m128 c2 = _mm_set1_ps (1.00172476f);

    t = _mm_cvtps_epi32 (_mm_mul_ps (a, x));
    j = _mm_and_si128 (t, m);            /* j = (int)(floor (x/log(2))) << 23 */
    t = _mm_sub_epi32 (t, j);
    f = _mm_mul_ps (ttm23, _mm_cvtepi32_ps (t)); /* f = (x/log(2)) - floor (x/log(2)) */
    p = c0;                              /* c0 */
    p = _mm_mul_ps (p, f);               /* c0 * f */
    p = _mm_add_ps (p, c1);              /* c0 * f + c1 */
    p = _mm_mul_ps (p, f);               /* (c0 * f + c1) * f */
    p = _mm_add_ps (p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    r = _mm_castsi128_ps (_mm_add_epi32 (j, _mm_castps_si128 (p))); /* r = p * 2^i*/
    return r;
}

__m128d FastExpSsef(__m128d x) // https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
{
	__m128 xx =_mm_castpd_ps(x);
	__m128 a = _mm_set1_ps(12102203.0f); /* (1 << 23) / log(2) */  ///not for doubles
	__m128i b = _mm_set1_epi32(127 * (1 << 23) - 298765);
	__m128i t = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(a, xx)), b);
	return _mm_castps_pd(_mm_castsi128_ps(t));
}

inline __m128 BetterFastExpSse(__m128 x)  // https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
{
	const __m128 a = _mm_set1_ps((1 << 22) / float( 0.69314718055994530942 ) );  // to get exp(x/2)
	const __m128i b = _mm_set1_epi32(127 * (1 << 23));       // NB: zero shift!				 // still not good enough accuracy/ maybe sounds better!?!??
	__m128i r = _mm_cvtps_epi32(_mm_mul_ps(a, x));
	__m128i s = _mm_add_epi32(b, r);
	__m128i t = _mm_sub_epi32(b, r);
	return _mm_div_ps(_mm_castsi128_ps(s), _mm_castsi128_ps(t));
}


 const __m128 BFExpSseA = _mm_set1_ps((1 << 22) / float(0.69314718055994530942));  // to get exp(x/2)
 const __m128i BFExpSseB = _mm_set1_epi32(127 * (1 << 23));       // NB: zero shift!				

 //inline __m128d BetterFastExpSsed(__m128d in)  // https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
//{
//		in = _mm_min_pd(_mm_max_pd(in, _explim), _nexplim );
//	const __m128i r = _mm_cvtps_epi32(_mm_mul_ps(BFExpSseA, _mm_castpd_ps(in)));
//	return _mm_div_pd(_mm_castsi128_pd(_mm_add_epi32(BFExpSseB, r) ), _mm_castsi128_pd(_mm_sub_epi32(BFExpSseB, r) ));
//}

 inline __m128 BetterFastExpSsef(__m128 in)  // https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
 {
	 in = _mm_min_ps(_mm_max_ps(in, _explim), _nexplim);
	 const __m128i r = _mm_cvtps_epi32(_mm_mul_ps(BFExpSseA, in));
	 return _mm_div_ps(_mm_castsi128_ps(_mm_add_epi32(BFExpSseB, r)), _mm_castsi128_ps(_mm_sub_epi32(BFExpSseB, r)));
 }


inline __m128d rexp(__m128d x) {
	double ex[2];
	_mm_store_pd(ex, x);

		ex[0] = exp(ex[0]);
		ex[1] = exp(ex[1]);
	
		return 	_mm_load_pd(ex);
}

inline __m128 rexpf(__m128 x) {
	float ex[4];
	_mm_store_ps(ex, x);

	ex[0] = expapproxf(ex[0]);
	ex[1] = expapproxf(ex[1]);

	return 	_mm_load_ps(ex);
}


	/** approximation for 2^x, optimized on the range [0, 1] */   //http://www.dangelo.audio/code/omega.h
	template <typename T>
	inline T pow2_approx(T x)
	{
		constexpr T alpha = (T) 0.07944154167983575;
		constexpr T beta = (T) 0.2274112777602189;
		constexpr T gamma = (T) 0.6931471805599453;
		constexpr T zeta = (T) 1.0;

		return zeta + x * (gamma + x * (beta + x * alpha));
	}

	/** approximation for exp(x) (64-bit) */
	inline double exp_approx64(double x)
	{
		x = fmax(-126.0, 1.442695040888963 * x);

		union
		{
			int64_t i;
			double d;
		} v;

		int64_t xi = (int64_t)x;
		int64_t l = x < 0.0 ? xi - 1 : xi;
		double d = x - (double)l;
		v.i = (l + 1023) << 52;

		return v.d * pow2_approx<double>(d);
	}

 static inline double catmull( double x0, const double x1,  double x2, const double x3, const double D) {
	
		x0 = (x2 - x0) * .5;
		x2 = x0 + x1 - x2;

		const double a0 = x2 + x2 - x0 + ((x3 - x1) * .5);	

		return a0 * D - (a0 + x2) * D + x0 * D + x1 * D;
	}

 //inline __m128d ssecatmull(__m128d x0, __m128d x1, __m128d x2, __m128d x3, __m128d D) {
 //
//	 x0 = _mm_mul_pd(_mm_sub_pd(x2, x0), _halfd);
//	 x2 = _mm_sub_pd( _mm_add_pd( x0, x1), x2 );
//	 const __m128d a0 = _mm_add_pd( _mm_sub_pd(_mm_add_pd( x2 , x2 ) , x0 ) , (_mm_mul_pd(_mm_sub_pd(x3 , x1) , _halfd)) );
 //
//	 return _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a0, D), _mm_mul_pd(_mm_add_pd(a0, x2), D)), _mm_add_pd(_mm_mul_pd(x0, D), _mm_mul_pd(x1, D)));
 //}

 struct ssecat {
	
	 void ssecatmullset(__m128d x0, __m128d x1, __m128d x2, __m128d x3 ) {

		 x0 = _mm_mul_pd(_mm_sub_pd(x2, x0), _halfdd);
		 x2 = _mm_sub_pd(_mm_add_pd(x0, x1), x2);
		 const __m128d a0 = _mm_add_pd(_mm_sub_pd(_mm_add_pd(x2, x2), x0), (_mm_mul_pd(_mm_sub_pd(x3, x1), _halfdd)));

		 aa0 = a0;
		 ax0 = x0;
		 ax1 = x1;
		 ax2 = x2;
	 }

	 inline __m128d ssecatmullproc( __m128d D) {

		 return _mm_add_pd(_mm_sub_pd(_mm_mul_pd(aa0, D), _mm_mul_pd(_mm_add_pd(aa0, ax2), D)), _mm_add_pd(_mm_mul_pd(ax0, D), _mm_mul_pd(ax1, D)));
	 }

 private:
	 __m128d aa0 = { 0. };
	 __m128d ax0 = { 0. };
	 __m128d ax1 = { 0. };
	 __m128d ax2 = { 0. };

	 const __m128d caa0 = { 0. };
	 const __m128d cax0 = { 0. };
	 const __m128d cax1 = { 0. };
	 const __m128d cax2 = { 0. };
	 const __m128d _halfdd = _mm_set1_pd(0.5);
 };

 
 
inline class zdf {
	 public:
		 zdf() {};
		 ~zdf() {};

		 void setfilter(double w2, double r) {
			 w_2f = _mm_set1_ps(w2);
			 _2kf = _mm_set1_ps(r);
			 denumf = _mm_div_ps(UnityVectf, _mm_add_ps(_mm_add_ps(_mm_mul_ps(w_2f, w_2f), _mm_mul_ps(w_2f, _2kf)), UnityVectf));
			 resf = _mm_set1_ps(-(w2 + r));

			w_2 = _mm_set1_pd(w2);
			_2k = _mm_set1_pd(r);
			denum = _mm_div_pd(UnityVect, _mm_add_pd(_mm_add_pd(_mm_mul_pd(w_2, w_2), _mm_mul_pd(w_2, _2k)), UnityVect));
			res = _mm_set1_pd(-(w2 + r));
		 }

		 inline __m128 zdf2pLP(__m128 inn ) {
			
			 highf = _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_mm_mul_ps(resf, s1f), s2f), inn), denumf);
			 bandf = _mm_add_ps(_mm_mul_ps(w_2f, highf), s1f);
			 lowf = _mm_add_ps(_mm_mul_ps(w_2f, bandf), s2f);
			 s1f = _mm_add_ps(_mm_mul_ps(w_2f, highf), bandf);
			 s2f = _mm_add_ps(_mm_mul_ps(w_2f, bandf), lowf);

			 return lowf;
		 }

		 inline __m128 zdf2pHP(__m128 inn ) {

			 highf = _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_mm_mul_ps(resf, s1f), s2f), inn), denumf);
			 bandf = _mm_add_ps(_mm_mul_ps(w_2f, highf), s1f);
			 lowf = _mm_add_ps(_mm_mul_ps(w_2f, bandf), s2f);
			 s1f = _mm_add_ps(_mm_mul_ps(w_2f, highf), bandf);
			 s2f = _mm_add_ps(_mm_mul_ps(w_2f, bandf), lowf);

			 return highf;
		 }


		 inline __m128d zdf2pLPd(__m128d inn) {

			 high = _mm_mul_pd(_mm_add_pd(_mm_sub_pd(_mm_mul_pd(res, s1), s2), inn), denum);
			 band = _mm_add_pd(_mm_mul_pd(w_2, high), s1);
			 low = _mm_add_pd(_mm_mul_pd(w_2, band), s2);
			 s1 = _mm_add_pd(_mm_mul_pd(w_2, high), band);
			 s2 = _mm_add_pd(_mm_mul_pd(w_2, band), low);

			 return low;
		 }

		 inline __m128d zdf2pHPd(__m128d inn) {

			 high = _mm_mul_pd(_mm_add_pd(_mm_sub_pd(_mm_mul_pd(res, s1), s2), inn), denum);
			 band = _mm_add_pd(_mm_mul_pd(w_2, high), s1);
			 low = _mm_add_pd(_mm_mul_pd(w_2, band), s2);
			 s1 = _mm_add_pd(_mm_mul_pd(w_2, high), band);
			 s2 = _mm_add_pd(_mm_mul_pd(w_2, band), low);

			 return high;
		 }

	private:
		 __m128 s1f = _mm_set1_ps(0.003);
		 __m128 s2f = _mm_set1_ps(0.00013);
		 __m128 highf = _mm_set1_ps(0.00);
		 __m128 bandf = _mm_set1_ps(0.000);
		 __m128 lowf = _mm_set1_ps(0.0000);
		 __m128 UnityVectf = _mm_set1_ps(1.f);

		 __m128 w_2f = _mm_set1_ps(.1);
		 __m128 denumf = _mm_set1_ps(.9);
		 __m128 resf = _mm_set1_ps(-1.);
		 __m128 _2kf = _mm_set1_ps(1.414);

		 __m128d s1 = _mm_set1_pd(0.003);
		 __m128d s2 = _mm_set1_pd(0.00013);
		 __m128d high = _mm_set1_pd(0.00);
		 __m128d band = _mm_set1_pd(0.000);
		 __m128d low = _mm_set1_pd(0.0000);

		 __m128d w_2   = _mm_set1_pd(.1);
		 __m128d denum = _mm_set1_pd(.9);
		 __m128d res   = _mm_set1_pd(-1.);
		 __m128d _2k   = _mm_set1_pd(1.414);
		 __m128d UnityVect = _mm_set1_pd(1.);
	 };


class dcblk {
public:
	dcblk() {};
	~dcblk() {};

	//set frequency normalized
	void set(double fc) {
		_dccutf = _mm_set1_ps(fc);
		_dccut = _mm_set1_pd(fc);
	}

	double process(double in) {

		return in;
	}

	__m128 process(__m128 in) {
		_dcstatef = _dcblockf;
		_dcblockf = _mm_add_ps(in, _mm_mul_ps(_dcblockf, _dccutf));
		return  _mm_sub_ps(_dcblockf, _dcstatef);
	}

	__m128d process_128d(__m128d in) {
		_dcstate = _dcblock;
		_dcblock = _mm_add_pd(in, _mm_mul_pd(_dcblock, _dccut));
		return  _mm_sub_pd(_dcblock, _dcstate);
	}

private: protected:
	__m128 _dcblockf = _mm_set1_ps(0.);
	__m128 _dcstatef = _mm_set1_ps(0.);
	__m128 _dccutf = _mm_set1_ps(0.1);

	__m128d _dcblock = _mm_set1_pd(0.);
	__m128d _dcstate = _mm_set1_pd(0.);
	__m128d _dccut = _mm_set1_pd(0.1);
	};

class ZDF1pole {
	public:
		ZDF1pole() {};
		~ZDF1pole() {};

		void setfilter(double cutoff, double SR) {

			double fc = kPi * cutoff / SR;

			double g = tan( fc ) ;

			k = g / (1.0 + g);
	}

		double process1poleLP(double x) {

			double vn = (x - state)*k;

			double lpf = ((x - state)*k) + state;

			state = vn + lpf;

			return lpf;
		}

private: protected:

	double g = 0.1;
	double k = 0.0;
	double state = 0.00001;
};

	ZDF1pole lowshelfL;
	ZDF1pole lowshelfR;

	 zdf zdf_low;
	 zdf zdf_high;

	 zdf envfollower;

	 zdf twopolelowp;
	 __m128 sagfactor = _unityf;
	 float sag_p = 0;

	 float sagVol_p = 0.01;
	 float sagVt_p = 0.01;
	 float sagLP_p = 0.01;
	 float sagintrim_p = 0.01;
	__m128 sagVolume = _mm_set1_ps(1.);
	__m128 sagVtf  = _mm_set1_ps(1.);
	__m128 sagInf = _mm_set1_ps(1.);

	float sagInput = 0.;
	 float sagLPf = 0;

	cascadedbiq upss[2];
	cascadedbiq dwnn[2];

	dcblk DCblk;

	ssecat scrm[2];

	Iir::ChebyshevII::LowPass<12> ups[2];
	Iir::ChebyshevII::LowPass<12> dwn[2];

	TransistorModels ClassA;
	TransistorModels ClassB;

	DiodeModels DiodePair;

	DiodeModels doubleDiodePair[2];

//	cbiq dwnn[2];

	double ingainfactor = 1.;
	double ingainsm = 1.;
	double outgainfactor = 1.;
	double outgainsm = 1.;
	double ampgainfactor = 1.;
	double ampgainsm = 1.;
	double dccutsm = 1.;

	double hipasscutoff_p = 10.;
	int hipass_p = 0;

	double xout = 0.;
	double xoutR = 0.;
	float xoutput[32] = { 0. };

	__m128 _xoutput[32] = { _mm_set1_ps( 0.) };

	DynamicsProcessor dyn[2];

	double expandL = 1.;
	double expandR = 1.;

	double intp[4][2] = { 0 };

	double x[8] = { 0. };

	double dcblock[2] = { 0. };
	double dcstate[2] = { 0.01 }; 
	double dccut = .999;

	double dcblock1[2] = { 0. };
	double dcstate1[2] = { 0. };

	double dccutOS = .999;
	double dccut_p = 5;

	__m128 _dcblock = _mm_set1_ps(0.);
	__m128 _dcstate = _mm_set1_ps(0.);
	__m128 _dcout = _mm_set1_ps(0.);
	__m128	degen = _mm_set1_ps(0.);
	__m128 _gridout = _mm_set1_ps(0.);
	__m128 _sumout = _mm_set1_ps(0.);

	double dthreshold = 1.;
	__m128 _threshold = _mm_set1_ps(1);
	double threshold_p = 0;

	__m128 _diodeout = _mm_set1_ps(0.);

	double inputgain_p = 0.;
	double outputgain_p = 0.;

	double feedback_p = 0;
	double feedbacklow_p = 50.;
	float fbthresh_p = -12;
	double release_p = -12;

	double sdiode = 0;

	double alpha_p = 5.;

	int resistor_p = 1000;

	double Is = 0.1;
	double vt = 0.1;
	double C = 0.1;



	double capacitor_p = 0.000000033;
	double thermalvoltage_p = 0.026;
	double satcurrent_p = 0.00000000252;

	double assym_p = 0.0;
	double assym_sm = 0.0;

	double ampgain_p = 1.;
	double ampVt_p = 0.023;
	double cathodeR_p = 0.5;

	double overbias_p = 0.;
	double crossover = 0;
	__m128 _overbias = _mm_set1_ps(0.);
	double dccutoff = 5;

	double fx = 0.0001;

	int oversampling_p = 8;
	double oversampsampinv = .125;
	int oversampling_pOld = 83179;
	double invSR = 1. / 48000.;

	int sse_p = 1;

	const __m128 _unityd = _mm_set1_ps( 1.); // = 1
	const __m128 _nunityd = _mm_set1_ps(-1.);	// = -1
	const __m128 _zerod = _mm_set1_ps( 0.);	// = 0
	const __m128 _halfd = _mm_set1_ps(0.5);	// = .5
	const __m128 _sqfperror = { _mm_set1_ps(0.000001) };
	const __m128 absmask = _mm_castsi128_ps(_mm_srli_epi32(_mm_set1_epi32(-1), 1));

	const __m128 _dunityd = _mm_set1_ps(2.); // = 2
	const __m128 _dnunityd = _mm_set1_ps(-2.);	// = -2
	const __m128 _explim = _mm_set1_ps(80); // = 80
	const __m128 _nexplim = _mm_set1_ps(-80); // = -80


	__m128 _fx = _mm_set1_ps(0.1);

	__m128 _x[8] = { _mm_set1_ps(0.0) };

	WDFIdealRLCHPF rlchipass[2];

	WDFIdealRLCHPF1pole wdfhipass[2];

	WDFIdealRLCHighshelf wdfhishelf[2];



	inline float log2f_approx(float x) {

		union {
			int	i;
			float	f;
		} v;
		v.f = x;
		int ex = v.i & 0x7f800000;
		int e = (ex >> 23) - 127;
		v.i = (v.i - ex) | 0x3f800000;
		return (float)e - 2.213475204444817f + v.f * (3.148297929334117f + v.f * (-1.098865286222744f + v.f * 0.1640425613334452f));
	}


	inline float logf_approx(float x) {
		return 0.693147180559945f * log2f_approx(x);
	}

	/** approximation for log(x) (64-bit) */
	template <typename T>
	inline T log2_approx(T x)
	{
		constexpr T alpha = (T) 0.1640425613334452;
		constexpr T beta = (T)-1.098865286222744;
		constexpr T gamma = (T) 3.148297929334117;
		constexpr T zeta = (T)-2.213475204444817;

		return zeta + x * (gamma + x * (beta + x * alpha));
	}

	inline double log_approxd(double x)				// Fails for high gains in Omega3
	{
		union
		{
			int64_t i;
			double d;
		} v;
		v.d = x;
		int64_t ex = v.i & 0x7ff0000000000000;
		int64_t e = (ex >> 53) - 510;
		v.i = (v.i - ex) | 0x3ff0000000000000;

		return 0.693147180559945 * ((double)e + log2_approx<double>(v.d));
	}


	float exp_cst1_f = 2139095040.f;			// GOOD!!		https://github.com/jhjourdan/SIMD-math-prims/blob/master/simd_math_prims.h
	float exp_cst2_f = 0.f;
	/* Relative error bounded by 1e-5 for normalized outputs
   Returns invalid outputs for nan inputs
   Continuous error */
	inline float expapproxf(float val) {
		union { int32_t i; float f; } xu, xu2;
		float val2, val3, val4, b;
		int32_t val4i;
		val2 = 12102203.1615614f*val + 1065353216.f;
		val3 = val2 < exp_cst1_f ? val2 : exp_cst1_f;
		val4 = val3 > exp_cst2_f ? val3 : exp_cst2_f;
		val4i = (int32_t)val4;
		xu.i = val4i & 0x7F800000;
		xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
		b = xu2.f;

		/* Generated in Sollya with:
		   > f=remez(1-x*exp(-(x-1)*log(2)),
					 [|(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x|],
					 [1.000001,1.999999], exp(-(x-1)*log(2)));
		   > plot(exp((x-1)*log(2))/(f+x)-1, [1,2]);
		   > f+x;
		*/
		return
			xu.f * (0.509871020f + b * (0.312146713f + b * (0.166617139f + b *
			(-2.190619930e-3f + b * 1.3555747234e-2f))));
	}



	double zmiL = 0.0;
	double zmiR = 0.;
	double zmi2L = 0.0;
	double zmi2R = 0.;

	double envinmi, envinmiR = 0.1;
	double envpolmi, envpolmiR = 0.1;
	double envinmi2L, envinmi2R = 0.1;
	double envpolmi2L, envpolmi2R = 0.1;




	double rmsil = 0.01;
	double rmsir = 0.01;
	double rmsol = 0.01;
	double rmsor = 0.01;
	double meterinl = 0.;
	double meterinr = 0.;

	double relmi = (0.693147 / (300.0 * 0.001*48000.));
	double rmse = 1.;

	double dctimer = 0;	//to kill initial dc

	double viewoutput = 0.;

	ICustomView* knobView = nullptr;
	ICustomView* knobView2 = nullptr;

	ICustomView* dynknob[8] = { nullptr };

	ICustomView* SpectrumView2 = nullptr;
	moodycamel::ReaderWriterQueue<float, 16> customViewDataQueue;

	std::atomic<bool> queueEnabler;		///< atomic bool for enabling/disabling the queue
	bool isCustomViewDataQueueEnabled() const { return queueEnabler.load(std::memory_order_relaxed); }			///< set atomic variable with float
	void enableCustomViewDataQueue(bool value) { queueEnabler.store(value, std::memory_order_relaxed); }	///< get atomic variable as float

	// --- END USER VARIABLES AND FUNCTIONS -------------------------------------- //

protected:

private:
	//  **--0x07FD--**

	// **--0x1A7F--**
    // --- end member variables

public:
    /** static description: bundle folder name

	\return bundle folder name as a const char*
	*/
    static const char* getPluginBundleName();

    /** static description: name

	\return name as a const char*
	*/
    static const char* getPluginName();

	/** static description: short name

	\return short name as a const char*
	*/
	static const char* getShortPluginName();

	/** static description: vendor name

	\return vendor name as a const char*
	*/
	static const char* getVendorName();

	/** static description: URL

	\return URL as a const char*
	*/
	static const char* getVendorURL();

	/** static description: email

	\return email address as a const char*
	*/
	static const char* getVendorEmail();

	/** static description: Cocoa View Factory Name

	\return Cocoa View Factory Name as a const char*
	*/
	static const char* getAUCocoaViewFactoryName();

	/** static description: plugin type

	\return type (FX or Synth)
	*/
	static pluginType getPluginType();

	/** static description: VST3 GUID

	\return VST3 GUID as a const char*
	*/
	static const char* getVSTFUID();

	/** static description: 4-char code

	\return 4-char code as int
	*/
	static int32_t getFourCharCode();

	/** initalizer */
	bool initPluginDescriptors();

    /** Status Window Messages for hosts that can show it */
    void sendHostTextMessage(std::string messageString)
    {
        HostMessageInfo hostMessageInfo;
        hostMessageInfo.hostMessage = sendRAFXStatusWndText;
        hostMessageInfo.rafxStatusWndText.assign(messageString);
        if(pluginHostConnector)
            pluginHostConnector->sendHostMessage(hostMessageInfo);
    }

};




#endif /* defined(__pluginCore_h__) */


