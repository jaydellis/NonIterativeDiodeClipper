// -----------------------------------------------------------------------------
//    ASPiK Plugin Kernel File:  plugincore.cpp
//
/**
    \file   plugincore.cpp
    \author Will Pirkle
    \date   17-September-2018
    \brief  Implementation file for PluginCore object
    		- http://www.aspikplugins.com
    		- http://www.willpirkle.com
*/
// -----------------------------------------------------------------------------
#include "plugincore.h"
#include "plugindescription.h"

#include <iostream>
#include <fstream>



/**
\brief PluginCore constructor is launching pad for object initialization

Operations:
- initialize the plugin description (strings, codes, numbers, see initPluginDescriptors())
- setup the plugin's audio I/O channel support
- create the PluginParameter objects that represent the plugin parameters (see FX book if needed)
- create the presets
*/
PluginCore::PluginCore()
{
    // --- describe the plugin; call the helper to init the static parts you setup in plugindescription.h
    initPluginDescriptors();

    // --- default I/O combinations
	// --- for FX plugins
	if (getPluginType() == kFXPlugin)
	{
		addSupportedIOCombination({ kCFMono, kCFMono });
		addSupportedIOCombination({ kCFMono, kCFStereo });
		addSupportedIOCombination({ kCFStereo, kCFStereo });
	}
	else // --- synth plugins have no input, only output
	{
		addSupportedIOCombination({ kCFNone, kCFMono });
		addSupportedIOCombination({ kCFNone, kCFStereo });
	}

	// --- for sidechaining, we support mono and stereo inputs; auxOutputs reserved for future use
	addSupportedAuxIOCombination({ kCFMono, kCFNone });
	addSupportedAuxIOCombination({ kCFStereo, kCFNone });

	// --- create the parameters
    initPluginParameters();

    // --- create the presets
    initPluginPresets();
}

/**
\brief initialize object for a new run of audio; called just before audio streams

Operation:
- store sample rate and bit depth on audioProcDescriptor - this information is globally available to all core functions
- reset your member objects here

\param resetInfo structure of information about current audio format

\return true if operation succeeds, false otherwise
*/
bool PluginCore::reset(ResetInfo& resetInfo)
{
    // --- save for audio processing
    audioProcDescriptor.sampleRate = resetInfo.sampleRate;
    audioProcDescriptor.bitDepth = resetInfo.bitDepth;



	wdfhipass[0].reset(getSampleRate());
	wdfhipass[1].reset(getSampleRate());

	wdfhishelf[0].reset(getSampleRate());
	wdfhishelf[1].reset(getSampleRate());

	rlchipass[0].reset(getSampleRate());
	rlchipass[1].reset(getSampleRate());

	WDFParameters par;
	par.fc = 5; //5HZ

	wdfhipass[0].setParameters(par);
	wdfhipass[1].setParameters(par);

	par.fc = 800;
	par.boostCut_dB = -10.;// inputgain_p;

	wdfhishelf[0].setParameters(par);
	wdfhishelf[1].setParameters(par);


	relmi = (0.693147 / (300.0 * 0.001*getSampleRate()));

	//rms env time window 400ms
	rmse = expf(-1. / (.001 * getSampleRate() * 400.));

    // --- other reset inits
    return PluginBase::reset(resetInfo);
}

/**
\brief one-time initialize function called after object creation and before the first reset( ) call

Operation:
- saves structure for the plugin to use; you can also load WAV files or state information here
*/
bool PluginCore::initialize(PluginInfo& pluginInfo)
{
	// --- add one-time init stuff here
	dwn[0].setupN(.47 / oversampling_p, 50.);
	
	dwnn[0].setparams(dwn[0].getCascadeStorage());
	dwnn[1].setparams(dwn[0].getCascadeStorage());
	upss[0].setparams(dwn[0].getCascadeStorage());
	upss[1].setparams(dwn[0].getCascadeStorage());

	for (int i = 0; i < 5000; i++) {  
		dwnn[0].process(0.);
		dwnn[1].process(0.);
		upss[0].process(0.);
		upss[1].process(0.);
	}

//	wdfhipass[0].createWDF(); 
//	wdfhipass[1].createWDF();

//	wdfhishelf[0].createWDF();
//	wdfhishelf[1].createWDF();

//	wdfimp[0].createWDF();

	relmi = (0.693147 / (300.0 * 0.001*getSampleRate()));

	return true;
}

/**
\brief do anything needed prior to arrival of audio buffers

Operation:
- syncInBoundVariables when preProcessAudioBuffers is called, it is *guaranteed* that all GUI control change information
  has been applied to plugin parameters; this binds parameter changes to your underlying variables
- NOTE: postUpdatePluginParameter( ) will be called for all bound variables that are acutally updated; if you need to process
  them individually, do so in that function
- use this function to bulk-transfer the bound variable data into your plugin's member object variables

\param processInfo structure of information about *buffer* processing

\return true if operation succeeds, false otherwise
*/
bool PluginCore::preProcessAudioBuffers(ProcessBufferInfo& processInfo)
{
    // --- sync internal variables to GUI parameters; you can also do this manually if you don't
    //     want to use the auto-variable-binding


	if (oversampling_p != oversampling_pOld) {

		for (int i = 0; i < 2; i++) {
			ups[i].setupN(.47 / oversampling_p, 70.);
			dwn[i].setupN(.47 / oversampling_p, 70.);

			dwnn[i].setparams(dwn[0].getCascadeStorage());
			upss[i].setparams(ups[0].getCascadeStorage());
		}

		oversampling_pOld = oversampling_p;
	}

/*
	auto bq = dwnn[0].upss.getCascadeStorage();

	auto arr = bq.stageArray; 
	int maxstages = bq.maxStages;


	std::ofstream myfile;
	myfile.open("zcoeffs.txt");

	for (int i = 0; i < maxstages; i++) {
		auto A0 = arr[ i ].getA0();
		auto A1 = arr[ i ].getA1();
		auto A2 = arr[ i ].getA2();
		auto B0 = arr[ i ].getB0();
		auto B1 = arr[ i ].getB1();
		auto B2 = arr[ i ].getB2();

		myfile << "\n";
		myfile << "A0 ";
		myfile << A0;
		myfile << " A1 ";
		myfile << A1;
		myfile << " A2 ";
		myfile << A2;
		myfile << " b0 ";
		myfile << B0;
		myfile << " b1 ";
		myfile << B1;
		myfile << " b2 ";
		myfile << B2;
	}

	myfile.close();
	*/

		ingainfactor  = pow(10.0, inputgain_p  * .05);
		outgainfactor = pow(10.0, outputgain_p * .05);
		ampgainfactor = pow(10.0,  ampgain_p  * .05);
		dthreshold    = pow(10.0, threshold_p  * .05);
		_threshold    = _mm_set1_pd( dthreshold );

		crossover = overbias_p; //pow(10.0, overbias_p  * .05) - 1;
		_overbias = _mm_set1_pd( crossover );

	WDFParameters para;
	para.fc = hipasscutoff_p; //5HZ
	
	wdfhipass[0].setParameters(para);
	wdfhipass[1].setParameters(para);

	rlchipass[0].setParameters(para);
	rlchipass[1].setParameters(para);

	para.fc = 8000.*alpha_p ;
	para.boostCut_dB = ( inputgain_p  ) ;


	wdfhishelf[0].setParameters(para);
	wdfhishelf[1].setParameters(para);


	Is = satcurrent_p * 0.000000000001;   // sat curr
	vt = thermalvoltage_p * 0.001;     //th volt
	C = capacitor_p * 0.000000001;      //cap

	c1 = 1. /  resistor_p / C ;
	c2 = 2. * Is / C;
	c3 = c2 / vt;
	c4 = 1. / vt;

	_c1 =  _mm_set1_pd(c1);
	_c2 =  _mm_set1_pd(c2);
	_c3 =  _mm_set1_pd(c3);
	_c4 =  _mm_set1_pd(c4);
	_vt =  _mm_set1_pd(vt);

	dccut = 1. - ((2.*kPi*hipasscutoff_p )/ (getSampleRate())); //
	dccutOS = 1. - ((2.*kPi*dccut_p) / (getSampleRate()*oversampling_p));

	oversampsampinv = 1./oversampling_p;

    syncInBoundVariables();

    return true;
}

/**
\brief frame-processing method

Operation:
- decode the plugin type - for synth plugins, fill in the rendering code; for FX plugins, delete the if(synth) portion and add your processing code
- note that MIDI events are fired for each sample interval so that MIDI is tightly sunk with audio
- doSampleAccurateParameterUpdates will perform per-sample interval smoothing

\param processFrameInfo structure of information about *frame* processing

\return true if operation succeeds, false otherwise
*/
bool PluginCore::processAudioFrame(ProcessFrameInfo& processFrameInfo)
{
    // --- fire any MIDI events for this sample interval
    processFrameInfo.midiEventQueue->fireMidiEvents(processFrameInfo.currentFrame);

	// --- do per-frame smoothing
//	doParameterSmoothing();

	// --- call your GUI update/cooking function here, now that smoothing has occurred
	//
	//     NOTE:
	//     updateParameters is the name used in Will Pirkle's books for the GUI update function
	//     you may name it what you like - this is where GUI control values are cooked
	//     for the DSP algorithm at hand
	// updateParameters();


    // --- decode the channelIOConfiguration and process accordingly
    //
	// --- Synth Plugin:
	// --- Synth Plugin --- remove for FX plugins
	if (getPluginType() == kSynthPlugin)
	{
		// --- output silence: change this with your signal render code
		processFrameInfo.audioOutputFrame[0] = 0.0;
		if (processFrameInfo.channelIOConfig.outputChannelFormat == kCFStereo)
			processFrameInfo.audioOutputFrame[1] = 0.0;

		return true;	/// processed
	}

    // --- FX Plugin:
    if(processFrameInfo.channelIOConfig.inputChannelFormat == kCFMono &&
       processFrameInfo.channelIOConfig.outputChannelFormat == kCFMono)
    {
		// --- pass through code: change this with your signal processing
        processFrameInfo.audioOutputFrame[0] = processFrameInfo.audioInputFrame[0];

        return true; /// processed
    }

    // --- Mono-In/Stereo-Out
    else if(processFrameInfo.channelIOConfig.inputChannelFormat == kCFMono &&
       processFrameInfo.channelIOConfig.outputChannelFormat == kCFStereo)
    {
		// --- pass through code: change this with your signal processing
        processFrameInfo.audioOutputFrame[0] = processFrameInfo.audioInputFrame[0];
        processFrameInfo.audioOutputFrame[1] = processFrameInfo.audioInputFrame[0];

        return true; /// processed
    }

    // --- Stereo-In/Stereo-Out
    else if(processFrameInfo.channelIOConfig.inputChannelFormat == kCFStereo &&
       processFrameInfo.channelIOConfig.outputChannelFormat == kCFStereo)
    {
		// --- pass through code: change this with your signal processing
        processFrameInfo.audioOutputFrame[0] = processFrameInfo.audioInputFrame[0];
        processFrameInfo.audioOutputFrame[1] = processFrameInfo.audioInputFrame[1];

        return true; /// processed
    }

    return false; /// NOT processed
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	--- BLOCK/BUFFER PRE-PROCESSING FUNCTION --- //
//      Only used when BLOCKS or BUFFERS are processed
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
\brief pre-process the audio block

Operation:
- fire MIDI events for the audio block; see processMIDIEvent( ) for the code that loads
  the vector on the ProcessBlockInfo structure

\param IMidiEventQueue ASPIK event queue of MIDI events for the entire buffer; this
       function only fires the MIDI events for this audio block

\return true if operation succeeds, false otherwise
*/
bool PluginCore::preProcessAudioBlock(IMidiEventQueue* midiEventQueue)
{
	// --- pre-process the block
	processBlockInfo.clearMidiEvents();

	// --- sample accurate parameter updates
	for (uint32_t sample = processBlockInfo.blockStartIndex;
		sample < processBlockInfo.blockStartIndex + processBlockInfo.blockSize;
		sample++)
	{
		// --- the MIDI handler will load up the vector in processBlockInfo
		if (midiEventQueue)
			midiEventQueue->fireMidiEvents(sample);
	}
	return true;
}



/**
\brief block or buffer-processing method

Operation:
- process one block of audio data; see example functions for template code
- renderSynthSilence: render a block of 0.0 values (synth, silence when no notes are rendered)
- renderFXPassThrough: pass audio from input to output (FX)

\param processBlockInfo structure of information about *block* processing

\return true if operation succeeds, false otherwise
*/
bool PluginCore::processAudioBlock(ProcessBlockInfo& processBlockInfo)
{
	// --- FX or Synth Render
	//     call your block processing function here
	// --- Synth
	if (getPluginType() == kSynthPlugin)
		renderSynthSilence(processBlockInfo);

	// --- or FX
	else if (getPluginType() == kFXPlugin)
		renderFXPassThrough(processBlockInfo);

	return true;
}


/**
\brief
renders a block of silence (all 0.0 values) as an example
your synth code would render the synth using the MIDI messages and output buffers

Operation:
- process all MIDI events for the block
- perform render into block's audio buffers

\param blockInfo structure of information about *block* processing
\return true if operation succeeds, false otherwise
*/
bool PluginCore::renderSynthSilence(ProcessBlockInfo& blockInfo)
{
	// --- process all MIDI events in this block (same as SynthLab)
	uint32_t midiEvents = blockInfo.getMidiEventCount();
	for (uint32_t i = 0; i < midiEvents; i++)
	{
		// --- get the event
		midiEvent event = *blockInfo.getMidiEvent(i);

		// --- do something with it...
		// myMIDIMessageHandler(event); // <-- you write this
	}

	// --- render a block of audio; here it is silence but in your synth
	//     it will likely be dependent on the MIDI processing you just did above
	for (uint32_t sample = blockInfo.blockStartIndex, i = 0;
		 sample < blockInfo.blockStartIndex + blockInfo.blockSize;
		 sample++, i++)
	{
		// --- write outputs
		for (uint32_t channel = 0; channel < blockInfo.numAudioOutChannels; channel++)
		{
			// --- silence (or, your synthesized block of samples)
			blockInfo.outputs[channel][sample] = 0.0;
		}
	}
	return true;
}

/**
\brief
Renders pass-through code as an example; replace with meaningful DSP for audio goodness

Operation:
- loop over samples in block
- write inputs to outputs, per channel basis

\param blockInfo structure of information about *block* processing
\return true if operation succeeds, false otherwise
*/
bool PluginCore::renderFXPassThrough(ProcessBlockInfo& blockInfo)
{
	RCH::Undenormal noDenormals;
	dctimer = fmin(dctimer + 0.01,1.);
	
	const double fs = getSampleRate() * oversampling_p;
	const double k = 1. / fs;
	const double halfk = k * .5;
	const double alphak = alpha_p * k;

	const __m128d _fs = _mm_set1_pd(fs);
	const __m128d _k = _mm_set1_pd(k);
	const __m128d _halfk = _mm_set1_pd(halfk);
	const __m128d _alphak = _mm_set1_pd(alphak);


	const double wk1 = 1.0 / (C * resistor_p);
	const double wB0 = 2 * getSampleRate();

	const double wk2 = (C * resistor_p ) / (wB0 * C * resistor_p + 1.0);
	const double wk3 = (Is * resistor_p ) / (wB0 * C * resistor_p + 1.0);
	const double wk5 = log((Is * resistor_p) / ((wB0 * C * resistor_p + 1.0) * vt));
	const double wk6 = -wB0 - wB0;

	const __m128d _wk1 = _mm_set1_pd(wk1);
	const __m128d _wk2 = _mm_set1_pd(wk2);
	const __m128d _wk3 = _mm_set1_pd(wk3);
	const __m128d _wk5 = _mm_set1_pd(wk5);
	const __m128d _wk6 = _mm_set1_pd(wk6);

	const double ebersVT = .001 * ampVt_p;  //   0.0026  threshold point //blows up below 0.00002V
	const double ebersIs = 1e-16; //   1e-16;  //weakly couple with Vt - best kept low
	const double ebersBf = 100.;					// not interesting //try mV
	const double ebersRe = 1000.; // 1e3;     //negligble effect
	const double ebersVplus = 1.;	//  Voltage

	const double ebersInvVT = 1.0 / ebersVT;
	const double ebersk1 = ebersIs * ebersRe;
	const double ebersk2 = 1.0 / ebersBf;
	const double ebersk = log(ebersInvVT * ebersk1 * (1.0 + ebersk2));

	const __m128d _ebersVplus = _mm_set1_pd(ebersVplus);
	const __m128d _ebersVT = _mm_set1_pd(ebersVT);
	const __m128d _ebersInvVT = _mm_set1_pd(ebersInvVT);
	const __m128d _ebersk1 = _mm_set1_pd(ebersk1);
	const __m128d _ebersk2 = _mm_set1_pd(ebersk2);
	const __m128d _ebersk = _mm_set1_pd( ebersk);


	// --- block processing -- write to outputs
	for (uint32_t sample = blockInfo.blockStartIndex, i = 0;
		sample < blockInfo.blockStartIndex + blockInfo.blockSize;
		sample++, i++)
	{

		double meterinl = blockInfo.inputs[0][sample];
			double meterinr = blockInfo.inputs[1][sample];

	if( sse_p == 1)
		{
			ingainsm  = ingainsm * .999 + 0.001 * ingainfactor;
			outgainsm = outgainsm * .999 + 0.001 * outgainfactor;
			ampgainsm = ampgainsm * .999 + 0.001 * ampgainfactor;
			dccutsm = dccutsm * .999 + 0.001 * dccutOS;

			assym_sm = assym_sm * .999 + 0.001 * ( assym_p * dctimer );

			const __m128d assym_d   = _mm_set1_pd( assym_sm);
			const __m128d _ingainsm = _mm_set1_pd(ingainsm);
			const __m128d _ampgainsm = _mm_set1_pd(ampgainsm);
			const __m128d _dccutOS = _mm_set1_pd(dccutsm);

			const double inp[2] = { ( blockInfo.inputs[0][sample] ) * oversampling_p ,
										(blockInfo.inputs[1][sample])* oversampling_p };

			__m128d _xin = _mm_load_pd(inp);


			__m128d _xinfll[32] = {_zerod };
			_xinfll[0] = _xin;

			for (int i = 0; i < oversampling_p; i++) {							 // SSE

				_xin = upss[0].processSSE(_xinfll[i]);

				_xin = _mm_mul_pd(_xin, _ingainsm);							//input gain

				_xin = _mm_add_pd(_xin, assym_d);				//operating bias


				_ebVx = _mm_mul_pd(_ebersk1, _mm_mul_pd(_ebersk2, BetterFastExpSsed(						//D'Angelos ebers moll Transistor
								_mm_mul_pd(_ebersInvVT, (_mm_sub_pd(_xin, _ebersVplus))))));
				_ebVout = _mm_sub_pd(_mm_mul_pd(_ebersVT, _omega3(_mm_add_pd(_mm_mul_pd(_ebersInvVT, _mm_add_pd(_xin, _ebVx)), _ebersk))), _ebVx);


					_dcstate = _dcblock;																// coupling
					_dcblock = _mm_add_pd( _ebVout , _mm_mul_pd( _dcblock , _dccutOS ) );
					_dcout = _mm_sub_pd( _dcblock , _dcstate );

					_dcout = _mm_mul_pd(_dcout, _ampgainsm);

		const __m128d thrshmask = _mm_mul_pd(_mm_set1_pd(-0.3241584), _mm_max_pd(_zerod, _mm_sub_pd(_dcout, _threshold)) );			// Pirkle Grid-Conduction stage
		const __m128d compressionfactor = _mm_add_pd(_mm_set1_pd(0.4548416), _mm_mul_pd(_mm_set1_pd(0.5451584), rexp( thrshmask))) ;	//original addition factor = 0.4473253

				_dcout = _mm_mul_pd( _dcout ,  compressionfactor  );
						
					{																									//Class B Stage
						const __m128d pos_in = _mm_add_pd(_overbias, _dcout);
						_cbposVx = _mm_mul_pd(_ebersk1, _mm_mul_pd(_ebersk2, BetterFastExpSsed(						//pos
							_mm_mul_pd(_ebersInvVT, (_mm_sub_pd(pos_in, _ebersVplus))))));
						_cbposVout = _mm_sub_pd(_mm_mul_pd(_ebersVT, _omega3(_mm_add_pd(_mm_mul_pd(_ebersInvVT, _mm_add_pd(pos_in, _cbposVx)), _ebersk))), _cbposVx);

						const __m128d neg_in = _mm_sub_pd(_overbias, _dcout);

						_cbnegVx = _mm_mul_pd(_ebersk1, _mm_mul_pd(_ebersk2, BetterFastExpSsed(						//neg
							_mm_mul_pd(_ebersInvVT, (_mm_sub_pd(neg_in, _ebersVplus))))));
						_cbnegVout = _mm_sub_pd(_mm_mul_pd(_ebersVT, _omega3(_mm_add_pd(_mm_mul_pd(_ebersInvVT, _mm_add_pd(neg_in, _cbnegVx)), _ebersk))), _cbnegVx);

						_dcout = _mm_sub_pd(_cbposVout, _cbnegVout);
					}


				{																	//// Lambert Wright-Omega approx3 - DIODE Pair
					const __m128d q = _mm_sub_pd(_mm_mul_pd(_wk1, _dcout), _z1);
					const __m128d r = _mm_div_pd(q, (_mm_sqrt_pd(_mm_max_pd(_mm_mul_pd(q, q), _sqfperror))));
					const __m128d w = _mm_add_pd(_mm_mul_pd(_wk2, q), _mm_mul_pd(_wk3, r));
					const __m128d OUT = _mm_sub_pd(w, _mm_mul_pd(_mm_mul_pd(_vt, r), _omega3(_mm_add_pd(_mm_mul_pd(_mm_mul_pd(_c4, r), w), _wk5))));
					_z1 = _mm_sub_pd(_mm_mul_pd(_wk6, OUT), _z1);
					_x[0] = OUT; // 
				}
			
/*
		{										
				_xin = _ebVout;																			//// Ducceschi
				_xin = _mm_mul_pd(_k, _mm_mul_pd(_c1, _xin));

				__m128d _s1 = _mm_mul_pd(_c4, _x[0]);
				_s1 = _mm_max_pd(_mm_min_pd(_s1, _mm_set1_pd(40.)), _mm_set1_pd(-40.));  // hard limit for std::sinh

//	const __m128d expo  = rexp(_s1);				//no good for division
//	const __m128d expoR = rexp( _mm_mul_pd(_nunityd, _s1) ) ;	
//	const __m128d _sh = _mm_mul_pd(_mm_sub_pd(expo, expoR), _halfd);
				const __m128d _sh = sinhd(_s1);

				const __m128d _f = _mm_mul_pd(_halfk, _mm_add_pd(_mm_mul_pd(_x[0], _c1), _mm_mul_pd(_c2, _sh)));

				const __m128d dvzmsk = _mm_cmpgt_pd(_mm_mul_pd(_x[0], _x[0]), _sqfperror);

				_fx = _mm_add_pd(_c1, _mm_mul_pd(_c2, _mm_div_pd(_sh, _x[0])));
				_fx = _mm_add_pd(_mm_and_pd(dvzmsk, _fx), _mm_andnot_pd(dvzmsk, _mm_add_pd(_c1, _c3)));

				const __m128d _fp = _mm_mul_pd(_c3, _mm_sqrt_pd(_mm_add_pd(_unityd, _mm_mul_pd(_sh, _sh))));
				const __m128d _sigma = _mm_add_pd(_unityd, _mm_mul_pd(_alphak, _fp));

				_x[0] = _mm_div_pd(_mm_add_pd(_mm_sub_pd(_mm_mul_pd(_sigma, _x[0]), _f), _xin),
					_mm_add_pd(_sigma, _mm_mul_pd(_halfk, _fx)));

				//		_x[0] = _mm_add_pd(_mm_add_pd(
				//			_mm_and_pd(_mm_cmpgt_pd(_x[0], _unityd), _mm_add_pd(_mm_mul_pd(_mm_sub_pd(_x[0], _unityd), _halfd), _unityd)),
				//			_mm_and_pd(_mm_cmplt_pd(_x[0], _nunityd), _mm_add_pd(_mm_mul_pd(_mm_sub_pd(_x[0], _nunityd), _halfd), _nunityd))),
				//			_mm_and_pd(_mm_cmple_pd(_mm_mul_pd(_x[0], _x[0]), _unityd), _x[0]));		// SLOW???  //  compares??

			//	_x[0] = _mm_max_pd(_mm_min_pd(_unityd, _x[0]), _nunityd);
			}
	*/

				_xoutput[i] = dwnn[0].processSSE(_x[0]);

			}

			_mm_store_pd(xoutput, _xoutput[0]);

			switch (hipass_p) {
			case 0: {
				xout = wdfhipass[0].processAudioSample(-xoutput[0]);
				xoutR = wdfhipass[1].processAudioSample(-xoutput[1]);
				break;
			}
			case 1: {
				dcstate[0] = dcblock[0];
				dcblock[0] = xoutput[0] + dcblock[0] * dccut;
				xout = dcblock[0] - dcstate[0];

				dcstate[1] = dcblock[1];
				dcblock[1] = xoutput[1] + dcblock[1] * dccut;
				xoutR = dcblock[1] - dcstate[1];
				break;
			}
			case 2: {
				xout = xoutput[0];
				xoutR = xoutput[1];
				break;
			}
			default:  break;
			}

//		xoutR = wdfhishelf->processAudioSample(-xoutR);

			blockInfo.outputs[0][sample] = (xout)  * outgainsm;
			blockInfo.outputs[1][sample] = (xoutR) * outgainsm;

	}
	else
	{

		// --- handles multiple channels, but up to you for bookkeeping
		for (uint32_t channel = 0; channel < blockInfo.numAudioOutChannels; channel++)
		{
			ingainsm = ingainsm * .999 + 0.001 * ingainfactor;
			outgainsm = outgainsm * .999 + 0.001 * outgainfactor;
			assym_sm = assym_sm * .999 + 0.001 * assym_p*dctimer;
			ampgainsm = ampgainsm * .999 + 0.001 * ampgainfactor;
			dccutsm = dccutsm * .999 + 0.001 * dccutOS;

			double xin = (blockInfo.inputs[channel][sample] * ingainsm ) * oversampling_p;
			//	xin = fmin(fmax(xin, -12), 12);



			double xinfll[32] = { 0. };
			xinfll[0] = xin;

			//todo check http://ijeais.org/wp-content/uploads/2018/07/IJAER180702.pdf exp

			for (int i = 0; i < oversampling_p; i++) {   // c+

						xin = ups[channel].filter( xin * ( i < 1));

/*
						xin = k * c1 * xin;
				const double s1 = fmin(fmax(c4 * x[channel], -35.), 35.);  //+-88 limit// or 36.7
		//		const double s1 = c4 * x[channel];

				const double sh = sinh(s1);

				const double f = halfk * ((x[channel] * c1) + c2 * sh);

				if ((x[channel] * x[channel]) > .000001) { fx = c1 + c2 * sh / x[channel]; }
				else fx = c1 + c3;

				const double fp = c3 * sqrt(1. + sh * sh) + c1;
				const double sigma = 1. + alphak * fp;

				//		if (FastAbs(sigma + halfk * fx) > 0.00000001) {
				x[channel] = (sigma * x[channel] - f + xin) / (sigma + halfk * fx);
				//		}

				*/

						double xinh = xin + assym_sm - x[channel] * .0; // fmax(xin, 0) + assym_p * 0;

						double ebVoutn = 0.; 
						{										//ebers-moll equations
							ebVx   =  ebersk1 * exp( FastMin(40.,FastMax(-40.,(ebersInvVT * (xinh - ebersVplus)) + ebersk2) ) );   //
							ebVout = ebersVT * omega3(ebersInvVT * (xinh + ebVx) + ebersk) - ebVx;
						}

			//			xin = ebVout - .5  ;
					

						dcstate1[channel] = dcblock1[channel];
						dcblock1[channel] = ebVout + dcblock1[channel] * dccutsm;
						xin = dcblock1[channel] - dcstate1[channel];

						if (xin > dthreshold) {															//// Pirkle's Grid Conduction stage
							xin = xin * (0.4548416 +										////<- original addition constant was 0.4473253 but f(0) != 1.
								0.5451584*expapproxf(-0.3241584* FastMax(xin - dthreshold, 0.)) );
						}

																																	////Class B 
						{	cbposVx = ebersk1 * exp(FastMin(40., FastMax(-40., (ebersInvVT * ((crossover + xin) - ebersVplus)) + ebersk2)));   //positive half
							cbposVout = ebersVT * omega3(ebersInvVT * ((crossover + xin) + cbposVx) + ebersk) - cbposVx;

							cbnegVx = ebersk1 * exp(FastMin(40., FastMax(-40., (ebersInvVT * (crossover-xin - ebersVplus)) + ebersk2)));   //negative half
							cbnegVout = ebersVT * omega3(ebersInvVT * ((crossover - xin) + cbnegVx) + ebersk) - cbnegVx;	}

						xin = cbposVout - cbnegVout;
						
						xin = xin * ampgainsm;

				{												// D'angelo's Lambert Wright-Omega Diode - https://www.dafx.de/paper-archive/2019/DAFx2019_paper_5.pdf
					const double q = wk1 * xin - p_z1[channel];
					const double r = signbit(q) * -2. + 1.;
					const double w = wk2 * q + wk3 * r;
					const double OUT = w - vt * r * omega3( c4 * r * w + wk5);
					 p_z1[channel] = wk6 * OUT - p_z1[channel];

					x[channel] = OUT;
				}


							xoutput[i] = dwn[channel].filter(x[channel]);
			//	xoutput[i] = dwnn[channel].processUnrolled8p(x[channel]);

			//	x[channel] = fmin(fmax(x[channel], -1.), 1.);	//safety shouldn't hit
			}

			switch (hipass_p) {
			case 0: {
				xout = wdfhipass[channel].processAudioSample(-xoutput[0]);
				break;
			}
			case 1: {
				dcstate[channel] = dcblock[channel];
				dcblock[channel] = xoutput[0] + dcblock[channel] * dccut;
				xout = dcblock[channel] - dcstate[channel];
				break;
			}
			case 2: {
				xout = xoutput[0];
				break;
			}
			default:  break;
			}

			blockInfo.outputs[channel][sample] = (xout)* outgainsm;

		} 
		xoutR = blockInfo.outputs[1][sample]; // quick fix for stereo meter
	}
		//meters
		{
			envinmi     = ( FastAbs( meterinl  * ingainsm * ampgainsm)) - zmiL;  //meter in
			envpolmi    = ((envinmi > 0.0)) + ((envinmi < 0.0) * relmi);
			zmiL = zmiL + (envinmi * envpolmi);

			envinmiR    = ( FastAbs( meterinr * ingainsm * ampgainsm)) - zmiR;			// 1 was gainsm
			envpolmiR   = ((envinmiR > 0.0)) + ((envinmiR < 0.0) * relmi);
			zmiR = zmiR + (envinmiR * envpolmiR);

			envinmi2L  = ( FastAbs( xout) * outgainsm) - zmi2L;
			envpolmi2L = ((envinmi2L > 0.0)) + ((envinmi2L < 0.0) * relmi);
			zmi2L = zmi2L + (envinmi2L * envpolmi2L);   //.00049

			envinmi2R  = ( FastAbs( xoutR ) * outgainsm) - zmi2R;
			envpolmi2R = ((envinmi2R > 0.0)) + ((envinmi2R < 0.0) * relmi);
			zmi2R = zmi2R + (envinmi2R * envpolmi2R);   //.00049				//meter out

			rmsil = rmsil * rmse + (1. - rmse) * ((meterinl *  ingainsm*ampgainsm) * ( meterinl *  ingainsm) );			//rms l	
			rmsir = rmsir * rmse + (1. - rmse) * ((meterinr *  ingainsm*ampgainsm) * (meterinr * ingainsm ) );			//rms r

			rmsol = rmsol * rmse + (1. - rmse) * ((xout* outgainsm)  * (xout  * outgainsm));		//rms lout
			rmsor = rmsor * rmse + (1. - rmse) * ((xoutR* outgainsm) * (xoutR * outgainsm));		//rms rout
		}

	//	if (isCustomViewDataQueueEnabled()) {	customViewDataQueue.enqueue(viewoutput);	}

	}
	return true;
}

	   


/**
\brief do anything needed prior to arrival of audio buffers

Operation:
- updateOutBoundVariables sends metering data to the GUI meters

\param processInfo structure of information about *buffer* processing

\return true if operation succeeds, false otherwise
*/
bool PluginCore::postProcessAudioBuffers(ProcessBufferInfo& processInfo)
{

	// --- update outbound variables; currently this is meter data only, but could be extended
	//     in the future
	updateOutBoundVariables();

    return true;
}

/**
\brief update the PluginParameter's value based on GUI control, preset, or data smoothing (thread-safe)

Operation:
- update the parameter's value (with smoothing this initiates another smoothing process)
- call postUpdatePluginParameter to do any further processing

\param controlID the control ID value of the parameter being updated
\param controlValue the new control value
\param paramInfo structure of information about why this value is being udpated (e.g as a result of a preset being loaded vs. the top of a buffer process cycle)

\return true if operation succeeds, false otherwise
*/
bool PluginCore::updatePluginParameter(int32_t controlID, double controlValue, ParameterUpdateInfo& paramInfo)
{
    // --- use base class helper
    setPIParamValue(controlID, controlValue);

    // --- do any post-processing
    postUpdatePluginParameter(controlID, controlValue, paramInfo);

    return true; /// handled
}

/**
\brief update the PluginParameter's value based on *normlaized* GUI control, preset, or data smoothing (thread-safe)

Operation:
- update the parameter's value (with smoothing this initiates another smoothing process)
- call postUpdatePluginParameter to do any further processing

\param controlID the control ID value of the parameter being updated
\param normalizedValue the new control value in normalized form
\param paramInfo structure of information about why this value is being udpated (e.g as a result of a preset being loaded vs. the top of a buffer process cycle)

\return true if operation succeeds, false otherwise
*/
bool PluginCore::updatePluginParameterNormalized(int32_t controlID, double normalizedValue, ParameterUpdateInfo& paramInfo)
{
	// --- use base class helper, returns actual value
	double controlValue = setPIParamValueNormalized(controlID, normalizedValue, paramInfo.applyTaper);

	// --- do any post-processing
	postUpdatePluginParameter(controlID, controlValue, paramInfo);

	return true; /// handled
}

/**
\brief perform any operations after the plugin parameter has been updated; this is one paradigm for
	   transferring control information into vital plugin variables or member objects. If you use this
	   method you can decode the control ID and then do any cooking that is needed. NOTE: do not
	   overwrite bound variables here - this is ONLY for any extra cooking that is required to convert
	   the GUI data to meaninful coefficients or other specific modifiers.

\param controlID the control ID value of the parameter being updated
\param controlValue the new control value
\param paramInfo structure of information about why this value is being udpated (e.g as a result of a preset being loaded vs. the top of a buffer process cycle)

\return true if operation succeeds, false otherwise
*/
bool PluginCore::postUpdatePluginParameter(int32_t controlID, double controlValue, ParameterUpdateInfo& paramInfo)
{
    // --- now do any post update cooking; be careful with VST Sample Accurate automation
    //     If enabled, then make sure the cooking functions are short and efficient otherwise disable it
    //     for the Parameter involved
    /*switch(controlID)
    {
        case 0:
        {
            return true;    /// handled
        }

        default:
            return false;   /// not handled
    }*/

    return false;
}

/**
\brief has nothing to do with actual variable or updated variable (binding)

CAUTION:
- DO NOT update underlying variables here - this is only for sending GUI updates or letting you
  know that a parameter was changed; it should not change the state of your plugin.

WARNING:
- THIS IS NOT THE PREFERRED WAY TO LINK OR COMBINE CONTROLS TOGETHER. THE PROPER METHOD IS
  TO USE A CUSTOM SUB-CONTROLLER THAT IS PART OF THE GUI OBJECT AND CODE.
  SEE http://www.willpirkle.com for more information

\param controlID the control ID value of the parameter being updated
\param actualValue the new control value

\return true if operation succeeds, false otherwise
*/
bool PluginCore::guiParameterChanged(int32_t controlID, double actualValue)
{
	/*
	switch (controlID)
	{
		case controlID::<your control here>
		{

			return true; // handled
		}

		default:
			break;
	}*/

	return false; /// not handled
}

/**
\brief For Custom View and Custom Sub-Controller Operations

NOTES:
- this is for advanced users only to implement custom view and custom sub-controllers
- see the SDK for examples of use

\param messageInfo a structure containing information about the incoming message

\return true if operation succeeds, false otherwise
*/
bool PluginCore::processMessage(MessageInfo& messageInfo)
{
	// --- decode message
	switch (messageInfo.message)
	{
		// --- add customization appearance here
	case PLUGINGUI_DIDOPEN:
	{
		enableCustomViewDataQueue(true);
		return false;
	}

	// --- NULL pointers so that we don't accidentally use them
	case PLUGINGUI_WILLCLOSE:
	{
		enableCustomViewDataQueue(false);
		return false;
	}

	// --- update view; this will only be called if the GUI is actually open
	case PLUGINGUI_TIMERPING:
	{

		if (isCustomViewDataQueueEnabled()) {

			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 2. + (-20.f* log10f(FastMin(zmiL *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 4. + (-20.f* log10f(FastMin(zmiR *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 10. + (-20.f* log10f(FastMin(zmi2L *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 12. + (-20.f* log10f(FastMin(zmi2R *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);

			viewoutput = 6. + (-20.* log10(FastMin(sqrt(rmsil) *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 8. + (-20.* log10(FastMin(sqrt(rmsir) *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 14. + (-20.* log10(FastMin(sqrt(rmsol) *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);  
			viewoutput = 16. + (-20.* log10(FastMin(sqrt(rmsor) *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);

			viewoutput = 20. + (-20.* log10(FastMin((  ( zmi2L / outgainsm)/( zmiL * ampgainsm )) , 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 22. + (-20.* log10(FastMin(( (zmi2R / outgainsm) /(zmiR * ampgainsm)) , 0.999)))*.001;;
			customViewDataQueue.enqueue(viewoutput);

			viewoutput = 24. + (-20.* log10(FastMin(((sqrt(rmsol) / outgainsm) / ( rmsil * ampgainsm) ), 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 26. + (-20.* log10(FastMin(((sqrt(rmsor) / outgainsm) / ( rmsir * ampgainsm)), 0.999)))*.001;;
			customViewDataQueue.enqueue(viewoutput);
	
			customViewDataQueue.enqueue(viewoutput);
			customViewDataQueue.enqueue(viewoutput);
			customViewDataQueue.enqueue(viewoutput);

		}


		if (SpectrumView2)
		{
			float audioSample = 0.0;

			// --- try to get a value from queue
			bool success = customViewDataQueue.try_dequeue(audioSample);

			// --- if succeeds:
			if (success)
			{
				// --- empty queue into views; the each handle this differently
				while (success)
				{
			
					if (SpectrumView2)
						SpectrumView2->pushDataValue(audioSample);

					// -- try to get next value until queue is empty
					success = customViewDataQueue.try_dequeue(audioSample);
				}
			}

			// --- update and mark view as dirty

			if (SpectrumView2)
				SpectrumView2->updateView();

			return true;
		}

		return false;
	}

	// --- register the custom view, grab the ICustomView interface
	case PLUGINGUI_REGISTER_CUSTOMVIEW:
	{
		// --- decode name string

		if (messageInfo.inMessageString.compare("View2") == 0)
		{
			// --- (1) get the custom view interface via incoming message data*
			if (SpectrumView2 != static_cast<ICustomView*>(messageInfo.inMessageData))
				SpectrumView2 = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!SpectrumView2) return false;

			// --- registered!
			return true;
		}
	}

	case PLUGINGUI_REGISTER_SUBCONTROLLER:
	case PLUGINGUI_QUERY_HASUSERCUSTOM:
	case PLUGINGUI_USER_CUSTOMOPEN:
	case PLUGINGUI_USER_CUSTOMCLOSE:
	case PLUGINGUI_EXTERNAL_SET_NORMVALUE:
	case PLUGINGUI_EXTERNAL_SET_ACTUALVALUE:
	{

		return false;
	}

	default:
		break;
	}

	return false; /// not handled
}


/**
\brief process a MIDI event

NOTES:
- MIDI events are 100% sample accurate; this function will be called repeatedly for every MIDI message
- see the SDK for examples of use

\param event a structure containing the MIDI event data

\return true if operation succeeds, false otherwise
*/
bool PluginCore::processMIDIEvent(midiEvent& event)
{
	// --- IF PROCESSING AUDIO BLOCKS: push into vector for block processing
	if (!pluginDescriptor.processFrames)
	{
		processBlockInfo.pushMidiEvent(event);
		return true;
	}

	// --- IF PROCESSING AUDIO FRAMES: decode AND service this MIDI event here
	//     for sample accurate MIDI
	// myMIDIMessageHandler(event); // <-- you write this

	return true;
}

/**
\brief (for future use)

NOTES:
- MIDI events are 100% sample accurate; this function will be called repeatedly for every MIDI message
- see the SDK for examples of use

\param vectorJoysickData a structure containing joystick data

\return true if operation succeeds, false otherwise
*/
bool PluginCore::setVectorJoystickParameters(const VectorJoystickData& vectorJoysickData)
{
	return true;
}

/**
\brief create all of your plugin parameters here

\return true if parameters were created, false if they already existed
*/
bool PluginCore::initPluginParameters()
{
	if (pluginParameterMap.size() > 0)
		return false;

	// --- Add your plugin parameter instantiation code bewtween these hex codes
	// **--0xDEA7--**

	PluginParameter* piParam = nullptr;

	piParam = new PluginParameter(controlID::inputgain, "input gain", "dB", controlVariableType::kDouble, -24.0, 24., 0.0, taper::kLinearTaper);
	piParam->setBoundVariable(&inputgain_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::outputgain, "output gain", "dB", controlVariableType::kDouble, -24.0, 24., 0.0, taper::kLinearTaper);
	piParam->setBoundVariable(&outputgain_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::resistor, "resistance", "Ohm", controlVariableType::kInt, 400000., 1., 500., taper::kLogTaper);
	piParam->setBoundVariable(&resistor_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::alpha, "ALPHA", "", controlVariableType::kDouble, 1., 6., 1., taper::kLinearTaper);
	piParam->setBoundVariable(&alpha_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::capacitor, "capacitor", "nF", controlVariableType::kDouble, 20, 70, 33., taper::kLinearTaper);
	piParam->setBoundVariable(&capacitor_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::satcurrent, "current", "pA", controlVariableType::kDouble, 2.52, 250., 25, taper::kAntiLogTaper);
	piParam->setBoundVariable(&satcurrent_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::thermalvoltage, "thermalvoltage", "mV", controlVariableType::kDouble, 15, 65, 33, taper::kLinearTaper);
	piParam->setBoundVariable(&thermalvoltage_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipasscutoff, "highpass freq", "Hz", controlVariableType::kDouble, 1, 200, 10, taper::kAntiLogTaper);
	piParam->setBoundVariable(&hipasscutoff_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipass, "highpass freq", "WDF, Biquad, off", "WDF");
	piParam->setBoundVariable(&hipass_p, boundVariableType::kInt);
	piParam->setIsDiscreteSwitch(true);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::assymetry, "dc bias", "V", controlVariableType::kDouble, 0, 1., 0.66, taper::kLinearTaper);
	piParam->setBoundVariable(&assym_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::oversampling, "oversampling", "x", controlVariableType::kInt, 1, 32, 8, taper::kLinearTaper);
	piParam->setBoundVariable(&oversampling_p, boundVariableType::kInt);
	piParam->setIsDiscreteSwitch(true);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ssecontrol, "sse", "x", controlVariableType::kInt, 0, 1, 1, taper::kLinearTaper);
	piParam->setBoundVariable(&sse_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ampgain, "drive", "dB", controlVariableType::kDouble, -12, 24., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&ampgain_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ampVt, "vt ", "mV", controlVariableType::kDouble, 0.1, 10., 0.5, taper::kLinearTaper);
	piParam->setBoundVariable(&ampVt_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::cathodeR, " ", "x", controlVariableType::kDouble, .00, 1., 0., taper::kAntiLogTaper);
	piParam->setBoundVariable(&cathodeR_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::threshold, "threshold", "dB", controlVariableType::kDouble, -48, 12., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&threshold_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::overbias, "overbias", "V", controlVariableType::kDouble, -1., 1., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&overbias_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::dccutoff, "dc cut", "Hz", controlVariableType::kDouble, 1., 200., 10., taper::kAntiLogTaper);
	piParam->setBoundVariable(&dccut_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	// **--0xEDA5--**

	// --- BONUS Parameter
	// --- SCALE_GUI_SIZE
	PluginParameter* piParamBonus = new PluginParameter(SCALE_GUI_SIZE, "Scale GUI", "tiny,small,medium,normal,large,giant", "normal");
	addPluginParameter(piParamBonus);

	// --- create the super fast access array
	initPluginParameterArray();

	return true;
}

/**
\brief use this method to add new presets to the list

NOTES:
- see the SDK for examples of use
- for non RackAFX users that have large paramter counts, there is a secret GUI control you
  can enable to write C++ code into text files, one per preset. See the SDK or http://www.willpirkle.com for details

\return true if operation succeeds, false otherwise
*/
bool PluginCore::initPluginPresets()
{
	// **--0xFF7A--**

	// **--0xA7FF--**

    return true;
}

/**
\brief setup the plugin description strings, flags and codes; this is ordinarily done through the ASPiKreator or CMake

\return true if operation succeeds, false otherwise
*/
bool PluginCore::initPluginDescriptors()
{
	// --- setup audio procssing style
	//
	// --- kProcessFrames and kBlockSize are set in plugindescription.h
	//
	// --- true:  process audio frames --- less efficient, but easier to understand when starting out
	//     false: process audio blocks --- most efficient, but somewhat more complex code
	pluginDescriptor.processFrames = kProcessFrames;

	// --- for block processing (if pluginDescriptor.processFrame == false),
	//     this is the block size
	processBlockInfo.blockSize = kBlockSize;

    pluginDescriptor.pluginName = PluginCore::getPluginName();
    pluginDescriptor.shortPluginName = PluginCore::getShortPluginName();
    pluginDescriptor.vendorName = PluginCore::getVendorName();
    pluginDescriptor.pluginTypeCode = PluginCore::getPluginType();

	// --- describe the plugin attributes; set according to your needs
	pluginDescriptor.hasSidechain = kWantSidechain;
	pluginDescriptor.latencyInSamples = kLatencyInSamples;
	pluginDescriptor.tailTimeInMSec = kTailTimeMsec;
	pluginDescriptor.infiniteTailVST3 = kVSTInfiniteTail;

    // --- AAX
    apiSpecificInfo.aaxManufacturerID = kManufacturerID;
    apiSpecificInfo.aaxProductID = kAAXProductID;
    apiSpecificInfo.aaxBundleID = kAAXBundleID;  /* MacOS only: this MUST match the bundle identifier in your info.plist file */
    apiSpecificInfo.aaxEffectID = "aaxDeveloper.";
    apiSpecificInfo.aaxEffectID.append(PluginCore::getPluginName());
    apiSpecificInfo.aaxPluginCategoryCode = kAAXCategory;

    // --- AU
    apiSpecificInfo.auBundleID = kAUBundleID;   /* MacOS only: this MUST match the bundle identifier in your info.plist file */
    apiSpecificInfo.auBundleName = kAUBundleName;

    // --- VST3
    apiSpecificInfo.vst3FUID = PluginCore::getVSTFUID(); // OLE string format
    apiSpecificInfo.vst3BundleID = kVST3BundleID;/* MacOS only: this MUST match the bundle identifier in your info.plist file */
	apiSpecificInfo.enableVST3SampleAccurateAutomation = kVSTSAA;
	apiSpecificInfo.vst3SampleAccurateGranularity = kVST3SAAGranularity;

    // --- AU and AAX
    apiSpecificInfo.fourCharCode = PluginCore::getFourCharCode();

    return true;
}

// --- static functions required for VST3/AU only --------------------------------------------- //
const char* PluginCore::getPluginBundleName() { return getPluginDescBundleName(); }
const char* PluginCore::getPluginName(){ return kPluginName; }
const char* PluginCore::getShortPluginName(){ return kShortPluginName; }
const char* PluginCore::getVendorName(){ return kVendorName; }
const char* PluginCore::getVendorURL(){ return kVendorURL; }
const char* PluginCore::getVendorEmail(){ return kVendorEmail; }
const char* PluginCore::getAUCocoaViewFactoryName(){ return AU_COCOA_VIEWFACTORY_STRING; }
pluginType PluginCore::getPluginType(){ return kPluginType; }
const char* PluginCore::getVSTFUID(){ return kVSTFUID; }
int32_t PluginCore::getFourCharCode(){ return kFourCharCode; }



