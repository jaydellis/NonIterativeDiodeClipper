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
#include "customviews.h" // for custom knob message only!!

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

	invSR = 1. / getSampleRate();


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

	invSR = 1. / getSampleRate();

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

	dyn[0].reset(getSampleRate());
	dyn[1].reset(getSampleRate());

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


	
	DynamicsProcessorParameters dpram = dyn[0].getParameters();
	dpram.ratio = .20;
	dpram.calculation = dynamicsProcessorType::kDownwardExpander;
	dpram.threshold_dB = fbthresh_p;
	dpram.releaseTime_mSec = release_p;
	dpram.attackTime_mSec = 5;
	dpram.kneeWidth_dB = 6;
	dpram.softKnee = true;

	dyn[0].setParameters(dpram);
	dyn[1].setParameters(dpram);

	//gain calculations
		ingainfactor  = pow(10.0, inputgain_p  * .05);
		outgainfactor = pow(10.0, outputgain_p * .05);
		ampgainfactor = pow(10.0,  ampgain_p  * .05);
		dthreshold    = pow(10.0, threshold_p  * .05);
		_threshold    = _mm_set1_ps( dthreshold );

		crossover = overbias_p; //pow(10.0, overbias_p  * .05) - 1;
		_overbias = _mm_set1_ps( crossover );

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

// todo 	//tune resistor via fc = 1/(2*pi*RC)
	Is = satcurrent_p * 0.000000000001;   // sat curr
	vt = thermalvoltage_p * 0.001;     //th volt
	C = capacitor_p * 0.000000001;      //cap


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

	//dcfilter not quick enough 
	dctimer = FastMin(dctimer + 0.005, 1.);

	zdf_low.setfilter(feedbacklow_p* kPi*oversampsampinv* invSR, 1.4);		// 1/2R this must be most unresonant
//	zdf_high.setfilter(feedbackhi_p* kPi*oversampsampinv* invSR, 2);
	

	DiodePair.setDiodePair(resistor_p, C, Is, vt, getSampleRate() * oversampling_p);

	doubleDiodePair[0].setDiodePair(resistor_p, C, Is, vt, getSampleRate() * oversampling_p);
	doubleDiodePair[1].setDiodePair(resistor_p, C, Is, vt, getSampleRate() * oversampling_p);

	const float wk1 = 1.0 / (C * resistor_p);
	const float wB0 = 2. * getSampleRate()*oversampling_p;

	const float wk2 = (C * resistor_p ) / (wB0 * C * resistor_p + 1.0);
	const float wk3 = (Is * resistor_p ) / (wB0 * C * resistor_p + 1.0);
	const float wk5 = log((Is * resistor_p) / ((wB0 * C * resistor_p + 1.0) * vt));

	const float wk5a = log((Is * resistor_p) / ((wB0 * C * resistor_p + 1.0) * .25 *vt));

	const float wk6 = -wB0 - wB0;


	const double ebersVT = .001 * ampVt_p;  //   0.0026  threshold point //blows up below 0.00002V
	const double ebersIs = 1e-16; //   1e-16;  //weakly couple with Vt - best kept low
	const double ebersBf = 100.;					// not interesting //try mV
	const double ebersRe = 1000.; // 1e3;     //negligble effect
	const double ebersVplus = 1.;	//  Voltage

	const double ebersInvVT = 1.0 / ebersVT;
	const double ebersk1 = ebersIs * ebersRe;
	const double ebersk2 = 1.0 / ebersBf;
	const double ebersk = log(ebersInvVT * ebersk1 * (1.0 + ebersk2));


	ClassA.setBJT_128(ampVt_p);
	ClassB.setBJT_128(ampVt_p);


	__m128 _upsmp[32] = { _zerod };

		//// Grid saturation from http://willpirkle.com/special/Addendum_A19_Pirkle_v1.0.pdf

	const __m128 _prka = _mm_set1_ps(-0.3241584);
	const __m128 _prkb = _mm_set1_ps(0.4548416);		//revised so that f(0) = 1;
	const __m128 _prkc = _mm_set1_ps(0.5451584);

//	const __m128 _prka = _mm_set1_ps((-0.3241584);
//	const __m128 _prkb = _mm_set1_ps(.4473253);		// from p94 
//	const __m128 _prkc = _mm_set1_ps(0.5451584);	//small error of ~ -42.48 dB

//	const __m128 _prka = _mm_set1_ps(-0.3241584);
//	const __m128 _prkb = _mm_set1_ps(0.447);			//from page 62
//	const __m128 _prkc = _mm_set1_ps(0.545);		// small error of ~ −41.9 dB


	// --- block processing -- write to outputs
	for (uint32_t sample = blockInfo.blockStartIndex, i = 0;
		sample < blockInfo.blockStartIndex + blockInfo.blockSize;
		sample++, i++)
	{

		double meterinl = blockInfo.inputs[0][sample];
			double meterinr = blockInfo.inputs[1][sample];

		//	if (sse_p == 1)
			{
				ingainsm = ingainsm * .999 + 0.001 * ingainfactor;
				outgainsm = outgainsm * .999 + 0.001 * outgainfactor;
				ampgainsm = ampgainsm * .999 + 0.001 * ampgainfactor;
				dccutsm = dccutsm * .999 + 0.001 * dccutOS;

				assym_sm = assym_sm * .999 + 0.001 * (assym_p) * dctimer;

				const __m128 assym_d = _mm_set1_ps(assym_sm);
				const __m128 _ingainsm = _mm_set1_ps(ingainsm * oversampling_p);
				const __m128 _ampgainsm = _mm_set1_ps(ampgainsm);
				const __m128 _dccutOS = _mm_set1_ps(dccutsm);

				DCblk.set(dccutsm);

				const float inp[4] = { (blockInfo.inputs[0][sample]),
											(blockInfo.inputs[1][sample]),
												(blockInfo.inputs[0][sample]) * 0,
													(blockInfo.inputs[1][sample]) * 0 };

		//compute feedback expansion level
				dyn[0].processAudioSample(xoutput[0]);
				dyn[1].processAudioSample(xoutput[1]);
				double expandL = 1/dyn[0].getParameters().gainReduction;
				double expandR = 1/dyn[1].getParameters().gainReduction;

		//Feedback expansion 
				const __m128 _feedback = _mm_set_ps(0,0, feedback_p*expandR, feedback_p*expandL); // whcky sse

		//Input Gain
			_upsmp[0] = _mm_mul_ps( _mm_load_ps(inp) , _ingainsm); 

			__m128 _xinfll[32];

			for (int i = 0; i < oversampling_p; i++) {			

				_xinfll[i] = upss[0].processSSEf( _upsmp[i] );				

		//Class A operating bias
				__m128 _xin = _mm_add_ps( _xinfll[i], assym_d);

				if (sse_p == 0)
				{
		//D'Angelos ebers moll Class A Transistor
					_xin = (ClassA.processBJT_128(_xin));

			//DC coupling		// still needs double precision
					__m128d _xind = _mm_cvtps_pd(_xin);
					_xind = DCblk.process_128d(_xind);
					_dcout = _mm_cvtpd_ps(_xind);

				}  
				else {		//// both need double precision

					__m128d _xind = _mm_cvtps_pd(_xin);
					_xind = ClassA.processBJT_128d(_xind);
		//DC coupling		
					_xind = DCblk.process_128d(_xind);

					_dcout = _mm_cvtpd_ps(_xind);

				}


		//gain
			_dcout = _mm_mul_ps(_dcout, _ampgainsm);

		// Grid-Conduction stage
			const __m128 thrshmask = _mm_mul_ps( _prka, _mm_max_ps(_zerod, _mm_sub_ps(_dcout, _threshold)) );			
			const __m128 compressionfactor = _mm_add_ps( _prkb, _mm_mul_ps( _prkc, rexpf( thrshmask))) ;

				_dcout = _mm_mul_ps( _dcout ,  compressionfactor  );


		// -feedback lowpass
				__m128	degen = zdf_low.zdf2pLP(_diodeout);
		//expansion
				degen = _mm_mul_ps(degen, _feedback);
				_dcout = _mm_sub_ps(_dcout, degen);


		//Combined SSE Class B Stage
				_dcout = ClassB.processClassBCombined_128( _overbias, _dcout);

			//Diodes
			_diodeout = DiodePair.processDiodePair_128(_dcout);
		
			//Downsampling filter
				_xoutput[i] = dwnn[0].processSSEf( _diodeout);
			}

			_mm_store_ps(xoutput, _xoutput[0]);


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

			blockInfo.outputs[0][sample] = (xout)  * outgainsm ;
			blockInfo.outputs[1][sample] = (xoutR) * outgainsm;
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

		if (knobView) {
			// --- send the view a message
			VSTGUI::CustomViewMessage knobMessage;
			knobMessage.message = VSTGUI::MESSAGE_QUERY_CONTROL;
			std::string s;
			s = std::to_string(resistor_p);
			auto y = s.c_str();
			knobMessage.queryString.assign(y);

			// --- send the message
			knobView->sendMessage(&knobMessage);
			knobView->updateView();
		}

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


		//	std::ofstream myfile;
		//	myfile.open("zcoeffs.txt");
		//	myfile << 0;
		//	myfile.close();
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

		if (messageInfo.inMessageString.compare("CustomKnobView2") == 0)
		{
			// --- (1) get the custom view interface via incoming message data*
			if (knobView2 != static_cast<ICustomView*>(messageInfo.inMessageData))
				knobView2 = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!knobView2) return false;


			// --- send the view a message
			VSTGUI::CustomViewMessage knobMessage;
			knobMessage.message = VSTGUI::MESSAGE_QUERY_CONTROL;
			knobMessage.queryString.assign("Hello There!");

			// --- send the message
			knobView2->sendMessage(&knobMessage);

			// --- check the reply string; the messgageData variable contains a pointer to the object (DANGEROUS)
			const char* reply = knobMessage.replyString.c_str();
			printf("%s", reply);

			VSTGUI::CKnob* customKnob = static_cast<VSTGUI::CKnob*>(knobMessage.messageData);
			
			// --- registered!
			return true;
		}


		// --- example of querying plugin for information and getting a pointer to the control
		   //     which is VERY risky - you should use the custom view data structure and messaging
		   //     to call functions on the control at the proper time
		if (messageInfo.inMessageString.compare("CustomKnobView") == 0)
		{
			// --- (1) get the custom view interface via incoming message data*
			if (knobView != static_cast<ICustomView*>(messageInfo.inMessageData))
				knobView = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!knobView) return false;

			// --- send the view a message
			VSTGUI::CustomViewMessage knobMessage;
			knobMessage.message = VSTGUI::MESSAGE_QUERY_CONTROL;
			knobMessage.queryString.assign("Hello There!");

			// --- send the message
			knobView->sendMessage(&knobMessage);

			// --- check the reply string; the messgageData variable contains a pointer to the object (DANGEROUS)
			const char* reply = knobMessage.replyString.c_str();
			printf("%s", reply);

			// --- DO NOT DO THIS!!! (but it is possible)
			//CAnimKnob* customKnob = static_cast<CAnimKnob*>(knobMessage.messageData);

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

	piParam = new PluginParameter(controlID::resistor, "resistance", "Ohm", controlVariableType::kInt, 60000., 1., 500., taper::kLogTaper);
	piParam->setBoundVariable(&resistor_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::alpha, "ALPHA", "", controlVariableType::kDouble, 1., 6., 1., taper::kLinearTaper);
	piParam->setBoundVariable(&alpha_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::capacitor, "capacitor", "nF", controlVariableType::kDouble, 30, 70, 33., taper::kLinearTaper);
	piParam->setBoundVariable(&capacitor_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::satcurrent, "current", "pA", controlVariableType::kDouble, .00252, 250., 25, taper::kAntiLogTaper);
	piParam->setBoundVariable(&satcurrent_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::thermalvoltage, "thermalvoltage", "mV", controlVariableType::kDouble, 15, 110, 33, taper::kLinearTaper);
	piParam->setBoundVariable(&thermalvoltage_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipass, "highpass freq", "WDF, Biquad, off", "WDF");
	piParam->setBoundVariable(&hipass_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipasscutoff, "highpass freq", "Hz", controlVariableType::kDouble, 1, 200, 10, taper::kAntiLogTaper);
	piParam->setBoundVariable(&hipasscutoff_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::assymetry, "ClassAbias", "V", controlVariableType::kDouble, 0, 1., 0.66, taper::kLinearTaper);
	piParam->setBoundVariable(&assym_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::oversampling, "oversampling", "x", controlVariableType::kInt, 1, 32, 8, taper::kLinearTaper);
	piParam->setBoundVariable(&oversampling_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ssecontrol, "sse", "x", controlVariableType::kInt, 0, 1, 1, taper::kLinearTaper);
	piParam->setBoundVariable(&sse_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ampgain, "drive", "dB", controlVariableType::kDouble, -12, 24., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&ampgain_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::ampVt, "vt", "mV", controlVariableType::kDouble, 0.01, 8., 0.023, taper::kAntiLogTaper);
	piParam->setBoundVariable(&ampVt_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::cathodeR, "r", "x", controlVariableType::kDouble, .00, 1., 0.1, taper::kAntiLogTaper);
	piParam->setBoundVariable(&cathodeR_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::threshold, "threshold", "dB", controlVariableType::kDouble, -48, 12., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&threshold_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::overbias, "overbias", "V", controlVariableType::kDouble, -1., 1., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&overbias_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::dccutoff, "dccut", "Hz", controlVariableType::kDouble, 2., 200., 10., taper::kAntiLogTaper);
	piParam->setBoundVariable(&dccut_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::feedback, "fbthreshold", "-x", controlVariableType::kDouble, -1., 1., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&feedback_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::fblowpass, "fbacklowpass", "Hz", controlVariableType::kDouble, 1, 20000., 5., taper::kAntiLogTaper);
	piParam->setBoundVariable(&feedbacklow_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::fbthresh, "fbackthresh", "Hz", controlVariableType::kDouble, -36., 12., -12., taper::kLinearTaper);
	piParam->setBoundVariable(&fbthresh_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::release, "release", "ms", controlVariableType::kDouble, 1., 500., 200., taper::kLinearTaper);
	piParam->setBoundVariable(&release_p, boundVariableType::kDouble);
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



