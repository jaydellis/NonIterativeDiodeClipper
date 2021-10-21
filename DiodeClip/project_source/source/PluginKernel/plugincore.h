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
	ssecontrol = 11

};


#define SSE2_DOUBLE_CONST(Key, v) \
    static const __declspec(align(16)) double _D_##Key[2] = { v, v }
// repeat a const to be in packed form

#define SSE2_INT_CONST(Key, v) \
    static const __declspec(align(16)) int _I_##Key[4] = {v, v, v, v}

/*  SSE2 defs start */
SSE2_DOUBLE_CONST(1, 1.0);
SSE2_DOUBLE_CONST(0p5, 0.5);

SSE2_DOUBLE_CONST(cephes_LOG2, 1.44269504088896341);
SSE2_DOUBLE_CONST(cephes_C1, 0.693359375);
SSE2_DOUBLE_CONST(cephes_C2, -2.12194440e-4);

SSE2_DOUBLE_CONST(cephes_p0, 1.9875691500E-4);
SSE2_DOUBLE_CONST(cephes_p1, 1.3981999507E-3);
SSE2_DOUBLE_CONST(cephes_p2, 8.3334519073E-3);
SSE2_DOUBLE_CONST(cephes_p3, 4.1665795894E-2);
SSE2_DOUBLE_CONST(cephes_p4, 1.6666665459E-1);
SSE2_DOUBLE_CONST(cephes_p5, 5.0000001201E-1);

SSE2_DOUBLE_CONST(exp_upper, 88.3762626647949);
SSE2_DOUBLE_CONST(exp_lower, -88.3762626647949);

SSE2_INT_CONST(0x7f, 0x7f);
/*  SSE2 defs end */

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

	private:
		double aa1[8] = { -1.7 };
		double aa2[8] = { .8 };
		double bb0[8] = { 1. };
		double bb1[8] = { -.08 };
		double bb2[8] = { .01 };

		double state1[8] = { 0.0 };
		double state2[8] = { 0.0 };
		double out[8] = { 0.00 };

		int stages = 4;

		__m128d coeff[8][5];

		__m128d outm[8] = { _mm_set1_pd(0.00) };
		__m128d state1m[8] = { _mm_set1_pd(0.0)};
		__m128d state2m[8] = { _mm_set1_pd(0.0)};

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

	inline double exp2a(double x) {		//https://codingforspeed.com/using-faster-exponential-approximation/
		x = 1.0 + x * 0.0009765625;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x;
		return x;
	}

	inline __m128d _exp14a(__m128d x) {		//https://codingforspeed.com/using-faster-exponential-approximation/
		x = _mm_add_pd(_unityd, _mm_mul_pd(x, _mm_set1_pd(0.00006103515625 ))); //14th order
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

	inline __m128d _exp10a(__m128d x) {		//https://codingforspeed.com/using-faster-exponential-approximation/
		x = _mm_add_pd(_unityd, _mm_mul_pd(x, _mm_set1_pd(0.0009765625))); //10th order
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
	//???????????//
	/////////??

	__m128d exp_sse2(__m128d x)  // total BS
	{

		__m128d tmp = _mm_setzero_pd();

		__m128d fx;

		__m128i emm0;

		__m128d oneD = *(__m128d*) _D_1;

		x = _mm_min_pd(x, *(__m128d*)_D_exp_upper);
		x = _mm_max_pd(x, *(__m128d*)_D_exp_lower);

		// exp(x) = exp( z + n log(2) ) = exp(z) * n^2 , where 0<=z < 1

		fx = _mm_mul_pd(x, *(__m128d*)_D_cephes_LOG2);
		fx = _mm_add_pd(fx, *(__m128d*)_D_0p5);

		emm0 = _mm_cvttpd_epi32(fx); //overloaded to handle _m128d
		tmp = _mm_cvtepi32_pd(emm0);

		__m128d flag = _mm_cmpgt_pd(tmp, fx); // ? 0xffffffffffffffff : 0x0
		flag = _mm_and_pd(flag, oneD);
		fx = _mm_sub_pd(tmp, flag);

		tmp = _mm_mul_pd(fx, *(__m128d*)_D_cephes_C1);
		__m128d tmp2 = _mm_mul_pd(fx, *(__m128d*)_D_cephes_C2);
		x = _mm_sub_pd(x, tmp);

		x = _mm_mul_pd(x, tmp2);

		__m128d z = _mm_mul_pd(x, x);

		__m128d y = *(__m128d*)_D_cephes_p0;
		y = _mm_mul_pd(y, x);
		y = _mm_add_pd(y, *(__m128d*)_D_cephes_p1);
		y = _mm_mul_pd(y, x);
		y = _mm_add_pd(y, *(__m128d*)_D_cephes_p2);
		y = _mm_mul_pd(y, x);
		y = _mm_add_pd(y, *(__m128d*)_D_cephes_p3);
		y = _mm_mul_pd(y, x);
		y = _mm_add_pd(y, *(__m128d*)_D_cephes_p4);
		y = _mm_mul_pd(y, x);
		y = _mm_add_pd(y, *(__m128d*)_D_cephes_p5);
		y = _mm_mul_pd(y, z);
		y = _mm_add_pd(y, x);
		y = _mm_add_pd(y, oneD);

		/* build 2^n */

		emm0 = _mm_cvttpd_epi32(fx);
		emm0 = _mm_add_epi32(emm0, *(__m128i*)_I_0x7f);
		emm0 = _mm_slli_epi32(emm0, 23);
		__m128d pow2n = _mm_castsi128_pd(emm0);

		y = _mm_mul_pd(y, pow2n);

		return y;
	}


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
	const __m128i b = _mm_set1_epi32(127 * (1 << 23));       // NB: zero shift!
	__m128i r = _mm_cvtps_epi32(_mm_mul_ps(a, x));
	__m128i s = _mm_add_epi32(b, r);
	__m128i t = _mm_sub_epi32(b, r);
	return _mm_div_ps(_mm_castsi128_ps(s), _mm_castsi128_ps(t));
}

 static inline double catmull( double x0, const double x1,  double x2, const double x3, const double D) {
	
	//	const double a2 = (x2 - x0) * .5;
	//	const double ad = a2 + x1 - x2;

		x0 = (x2 - x0) * .5;
		x2 = x0 + x1 - x2;

		const double a0 = x2 + x2 - x0 + ((x3 - x1) * .5);	
	//		const double a0 = ad + ad - a2 + ((x3 - x1) * .5);
	//	const double a1 = a0 + ad ;
	//	const double a1 = a0 + x2 ;

		return a0 * D - (a0 + x2) * D + x0 * D + x1 * D;
	//	return a0 * D - a1 * D + a2 * D + x1 * D;
	}

 inline __m128d ssecatmull(__m128d x0, __m128d x1, __m128d x2, __m128d x3, __m128d D) {

	 x0 = _mm_mul_pd(_mm_sub_pd(x2, x0), _halfd);
	 x2 = _mm_sub_pd( _mm_add_pd( x0, x1), x2 );
	 const __m128d a0 = _mm_add_pd( _mm_sub_pd(_mm_add_pd( x2 , x2 ) , x0 ) , (_mm_mul_pd(_mm_sub_pd(x3 , x1) , _halfd)) );

	 return _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a0, D), _mm_mul_pd(_mm_add_pd(a0, x2), D)), _mm_add_pd(_mm_mul_pd(x0, D), _mm_mul_pd(x1, D)));
 }

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


	cascadedbiq upss[2];
	cascadedbiq dwnn[2];

	ssecat scrm[2];

	Iir::ChebyshevII::LowPass<8> ups[2];

	Iir::ChebyshevII::LowPass<8> dwn[2];

//	cbiq dwnn[2];

	double ingainfactor = 1.;
	double ingainsm = 1.;
	double outgainfactor = 1.;
	double outgainsm = 1.;

	double hipasscutoff_p = 10.;
	int hipass_p = 0;

	double xout = 0.;
	double xoutR = 0.;
	double xoutput[32] = { 0. };

	__m128d _xoutput[32] = { _mm_set1_pd( 0.) };

	double x[8] = { 0. };

	double dcblock[8] = { 0. };
	double dcstate[8] = { 0. }; 
	double dccut = .999;

	double inputgain_p = 0.;
	double outputgain_p = 0.;

	double alpha_p = 5.;

	int resistor_p = 1000;

	double Is = 0.1;
	double vt = 0.1;
	double C = 0.1;

	double c1 = 0.1;
	double c2 = 0.1;
	double c3 = 0.1;
	double c4 = 0.1;

	double capacitor_p = 0.000000033;
	double thermalvoltage_p = 0.026;
	double satcurrent_p = 0.00000000252;

	double assym_p = 0.0;
	double assym_sm = 0.0;

	double fx = 0.0001;

	double intin[8][2] = { 0. };

	__m128d intxin[8][2] = { 0. };

	int oversampling_p = 8;
	double oversampsampinv = .125;
	int oversampling_pOld = 83179;

	int sse_p = 0;

	const __m128d _unityd = _mm_set1_pd( 1.);
	const __m128d _nunityd = _mm_set1_pd(-1.);
	const __m128d _zerod = _mm_set1_pd( 0.);
	const __m128d _halfd = _mm_set1_pd(0.5);
	const __m128d _sqfperror = { _mm_set1_pd(0.000001) };

	const __m128d _dunityd = _mm_set1_pd(2.);
	const __m128d _dnunityd = _mm_set1_pd(-2.);

	__m128d _c1 = _mm_set1_pd( 0.1 );
	__m128d _c2 = _mm_set1_pd( 0.1 );
	__m128d _c3 = _mm_set1_pd( 0.1 );
	__m128d _c4 = _mm_set1_pd( 0.1 );

	__m128d _fx = _mm_set1_pd(0.1);

	__m128d _x[8] = { _mm_set1_pd(0.0) };

	WDFIdealRLCHPF rlchipass[2];

	WDFIdealRLCHPF1pole wdfhipass[2];

	WDFIdealRLCHighshelf wdfhishelf[2];

	WDFSpeakerImp wdfimp[2];


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

	double relmi = (0.693147 / (300.0 * 0.001*48000.));
	double rmse = 1.;

	double viewoutput = 0.;

	ICustomView* SpectrumView2 = nullptr;
	moodycamel::ReaderWriterQueue<float, 8> customViewDataQueue;

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


