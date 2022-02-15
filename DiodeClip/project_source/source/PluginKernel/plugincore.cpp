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
	dpram.ratio = .50;
	dpram.calculation = dynamicsProcessorType::kDownwardExpander;
	dpram.threshold_dB = fbthresh_p;
	dpram.releaseTime_mSec = 150.;
	dpram.attackTime_mSec = 1;
	dpram.kneeWidth_dB = 3;
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
	dctimer = FastMin(dctimer + 0.01, 1.);
	
	float sagtime = 1. / (release_p * .001);

	twopolelowp.setfilter( sagtime  * kPi * invSR, 1.4);

	lowshelfL.setfilter(feedbacklow_p, getSampleRate());
	lowshelfR.setfilter(feedbacklow_p, getSampleRate());
	
	ClassA.setBJT_128(ampVt_p);
	ClassB.setBJT_128(ampVt_p);

	__m128 _upsmp[32] = { _zerod };


		//// Grid saturation from http://willpirkle.com/special/Addendum_A19_Pirkle_v1.0.pdf
	const __m128 _prka = _mm_set1_ps(-0.3241584);
	const __m128 _prkb = _mm_set1_ps(0.4548416);		//revised coeffs so that f(0) = 1;
	const __m128 _prkc = _mm_set1_ps(0.5451584);


		//vt
	vt = thermalvoltage_p * 0.001;

	float currentmakeup = 1. / (1.159148907293 * powf(satcurrent_p, -0.04811517431058769434));


	// --- block processing -- write to outputs
	for (uint32_t sample = blockInfo.blockStartIndex, i = 0;
		sample < blockInfo.blockStartIndex + blockInfo.blockSize;
		sample++, i++)
	{

		float sagfct[4] = { 0. };
		_mm_store_ps(sagfct, sagfactor);

		sagLPf = sagfct[0] * sagLP_p;
		sagInput = sagInf.m128_f32[0];

			if (sagVt_p < 0) {	sagVtf.m128_f32[0] = 1./sagVtf.m128_f32[0];	}

		float vtmakeup = 1. / (-0.000451063384466717 + 0.04847815336514556 * thermalvoltage_p * sagVtf.m128_f32[0]);
		vtmakeup = vtmakeup * currentmakeup;

		DiodePair.setDiodePair(resistor_p + sagLPf * 1000, C, Is, vt *sagVtf.m128_f32[0] , getSampleRate() * oversampling_p);

		double meterinl = blockInfo.inputs[0][sample];
		double meterinr = blockInfo.inputs[1][sample];

		//decibel conversion
	//	sagfct[0] = powf(10.f, sagfct[0] * .05f);
	//	expandL =  dyn[0].computeGain(sagfct[0]);
	//	expandR =  dyn[1].computeGain(sagfct[0]);

		double sagLowshelfL = 1. / ( 1.+ sagfactor.m128_f32[0]  * FastAbs(feedback_p) * 8.) ;
		double sagLowshelfR = 1. / (1. + sagfactor.m128_f32[1] * FastAbs(feedback_p) * 8.) ;

		//	if (sse_p == 1)
			{
				ingainsm = ingainsm * .999 + 0.001 * ingainfactor  ;
				outgainsm = outgainsm * .999 + 0.001 * outgainfactor ;
				ampgainsm = ampgainsm * .999 + 0.001 * ampgainfactor;
				dccutsm = dccutsm * .999 + 0.001 * dccutOS;

				assym_sm = assym_sm * .999 + 0.001 * (assym_p) * dctimer;

				const __m128 assym_d = _mm_set1_ps(assym_sm);
				const __m128 _ingainsm = _mm_set1_ps(ingainsm * oversampling_p * sagInput * 1. / vtmakeup);
				const __m128 _ampgainsm = _mm_set1_ps(ampgainsm);
				const __m128 _dccutOS = _mm_set1_ps(dccutsm);

				DCblk.set(dccutsm);

					 float inp[4] = { (blockInfo.inputs[0][sample]),
											(blockInfo.inputs[1][sample]),
												(blockInfo.inputs[0][sample]) * 0,
													(blockInfo.inputs[1][sample]) * 0 };

					 float shelfboostcut = signbit(feedback_p)*-2 + 1;

				inp[0] = inp[0] + shelfboostcut * ( sagLowshelfL - 1) * lowshelfL.process1poleLP(inp[0]);
				inp[1] = inp[1] + shelfboostcut * ( sagLowshelfR - 1 ) * lowshelfL.process1poleLP(inp[1]);

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

				_gridout = _mm_mul_ps( _dcout ,  compressionfactor  );

				_sumout = _gridout;

		//Combined SSE Class B Stage
				if (sse_p == 0) {
					_sumout = ClassB.processClassBCombined_128(_overbias, _sumout);
				}
				else {
					_sumout = _mm_cvtpd_ps( ClassB.processClassBCombined_128d(_mm_cvtps_pd(_overbias), _mm_cvtps_pd(_sumout)) );
				}

			//Diodes
				if (sse_p == 0) {
					_diodeout = DiodePair.processDiodePair_128(_sumout);
				}
				else {
				//	_diodeout = _mm_cvtpd_ps(DiodePair.processDucesshiDiodePair_128d(_mm_cvtps_pd(_sumout)));

					_diodeout = _mm_cvtpd_ps( DiodePair.processDiodePair_128d(_mm_cvtps_pd( _sumout )) );

				}
		
			//Downsampling filter
				_xoutput[i] = dwnn[0].processSSEf( _diodeout);
			}

			_xoutput[0] = _mm_mul_ps(_xoutput[0], _mm_set1_ps(vtmakeup ));

			//sag follower filter ~13.5 Hz slightly resonant after Kuehnel
			sagfactor = twopolelowp.zdf2pLP( _mm_and_ps(_xoutput[0], absmask) );

			sagfactor = _mm_and_ps(sagfactor, absmask);

			//appy expansion
		//	__m128 expsagfactor = _mm_mul_ps(sagfactor, _mm_set1_ps(expandL));

			//sag shaping after Kuehnel// 

			sagVtf = _mm_div_ps(_unityf, _mm_add_ps(_unityf, _mm_mul_ps( sagfactor, _mm_set1_ps(FastAbs(sagVt_p)))) );

			sagVolume = _mm_div_ps(_unityf, _mm_add_ps(_unityf, _mm_mul_ps( sagfactor, _mm_set1_ps(sagVol_p))));

			sagInf = _mm_div_ps(_unityf, _mm_add_ps(_unityf, _mm_mul_ps( sagfactor, _mm_set1_ps(sagintrim_p))));

			_xoutput[0] = _mm_mul_ps(_xoutput[0], _mm_set1_ps(outgainsm));

			_xoutput[0] = _mm_mul_ps(_xoutput[0], sagVolume);

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

			blockInfo.outputs[0][sample] = (xout)   ;
			blockInfo.outputs[1][sample] = (xoutR)  ;
	} 

		//meters
		{
			envinmi     = ( FastAbs( meterinl  * ingainsm )) - zmiL;  //meter in
			envpolmi    = ((envinmi > 0.0)) + ((envinmi < 0.0) * relmi);
			zmiL = zmiL + (envinmi * envpolmi);

			envinmiR    = ( FastAbs( meterinr * ingainsm )) - zmiR;			// 1 was gainsm
			envpolmiR   = ((envinmiR > 0.0)) + ((envinmiR < 0.0) * relmi);
			zmiR = zmiR + (envinmiR * envpolmiR);

			envinmi2L  = ( FastAbs( xout) ) - zmi2L;
			envpolmi2L = ((envinmi2L > 0.0)) + ((envinmi2L < 0.0) * relmi);
			zmi2L = zmi2L + (envinmi2L * envpolmi2L);   //.00049

			envinmi2R  = ( FastAbs( xoutR ) ) - zmi2R;
			envpolmi2R = ((envinmi2R > 0.0)) + ((envinmi2R < 0.0) * relmi);
			zmi2R = zmi2R + (envinmi2R * envpolmi2R);   //.00049				//meter out

			rmsil = rmsil * rmse + (1. - rmse) * ((meterinl *  ingainsm ) * ( meterinl *  ingainsm) );			//rms l	
			rmsir = rmsir * rmse + (1. - rmse) * ((meterinr *  ingainsm ) * (meterinr * ingainsm ) );			//rms r

			rmsol = rmsol * rmse + (1. - rmse) * ((xout )  * (xout  ));		//rms lout
			rmsor = rmsor * rmse + (1. - rmse) * ((xoutR ) * (xoutR ));		//rms rout

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
		VSTGUI::CustomViewMessage knobMessage;
		knobMessage.message = VSTGUI::MESSAGE_QUERY_CONTROL;
		std::string valstrng;
		const char* y;

		//fb thresh
		if (dynknob[0]) {
			valstrng = std::to_string(expandL *  sagfactor.m128_f32[0]);
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			// --- send the message
			dynknob[0]->sendMessage(&knobMessage);
			dynknob[0]->updateView();
		}

		//fb
		if (dynknob[1]) {

			float vval = FastMin(1., FastMax(-1., (FastAbs(degen.m128_f32[0])*feedback_p*expandL)));
					vval = FastMax(-0.f, FastMin(48.f, (20.f* log10f( vval ) + 12.f))) / 24.f;

			valstrng = std::to_string( FastMin(1., FastMax(-1.,(1.41f*(FastAbs(degen.m128_f32[0]))*feedback_p*expandL)) ) );
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[1]->sendMessage(&knobMessage);
			dynknob[1]->updateView();
		}

		//invol
		if (dynknob[2]) {

		//	float vval = FastMax(-0.f,FastMin(48.f ,(20.f* log10f( zmiL / ampgainsm) +24.f) ) )/48.f;

			float vval = FastMax(-0.,FastMin(48. ,(20.* log10f( ingainsm * sagInput) +24.) ) )/48.;

			valstrng = std::to_string( vval );
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[2]->sendMessage(&knobMessage);
			dynknob[2]->updateView();
		}

		//midstage gain
		if (dynknob[3]) {

			float vval = FastMax(-0.f, FastMin(36.f, (20.f* log10f( zmiL * ampgainsm ) + 12.f))) / 36.f;

			valstrng = std::to_string( vval );
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[3]->sendMessage(&knobMessage);
			dynknob[3]->updateView();
		}

		//output gain
		if (dynknob[4]) {

			float vval = FastMax(-0.f, FastMin(48.f, (20.f* log10f(outgainsm * ( sagVolume.m128_f32[0] )) + 24.f))) / 48.f;

			valstrng = std::to_string( vval );
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[4]->sendMessage(&knobMessage);
			dynknob[4]->updateView();
		}

		//vt
		if (dynknob[5]) {

			//from customparamameters antilog scaling// thanks Will

			float vall = (kCTCorrFactorAntiUnity)*(-pow(10.0, ( -(vt * 1000. * sagVtf.m128_f32[0]) / (200.f) / kCTCoefficient)) + 1.0);

			if (sagVt_p < 0) { vall = (kCTCorrFactorAntiUnity)*(-pow(10.0, (-(vt * 1000. / sagVtf.m128_f32[0]) / (200.f) / kCTCoefficient)) + 1.0); }

			valstrng = std::to_string(  vall );
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[5]->sendMessage(&knobMessage);
			dynknob[5]->updateView();
		}

		//res
		if (dynknob[6]) {

			float vall = 1. - (kCTCorrFactorAntiUnity)*(-pow(10.0, (-(resistor_p + sagLPf * 1000) / (60000.f) / kCTCoefficient)) + 1.0);

			valstrng = std::to_string(vall);
			y = valstrng.c_str();
			knobMessage.queryString.assign(y);

			dynknob[6]->sendMessage(&knobMessage);
			dynknob[6]->updateView();
		}

		if (isCustomViewDataQueueEnabled()) {

			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 2. + (-20.* log10f(FastMin(zmiL *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 4. + (-20.* log10f(FastMin(zmiR *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 10. + (-20.* log10f(FastMin(zmi2L *.0625, 0.999)))*.001;
			customViewDataQueue.enqueue(viewoutput);
			viewoutput = 12. + (-20.* log10f(FastMin(zmi2R *.0625, 0.999)))*.001;
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
		
		if (messageInfo.inMessageString.compare("DynamicKnobView") == 0)		//threshold
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[0] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[0] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[0]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKV2") == 0) ////feedback
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[1] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[1] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[1]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKVingain") == 0) ////ingain
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[2] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[2] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[2]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKVgain") == 0) ////gain
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[3] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[3] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[3]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKVoutgain") == 0) ////
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[4] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[4] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[4]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKVgrid") == 0) ////VT
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[5] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[5] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[5]) return false;
			// --- registered!
			return true;
		}

		if (messageInfo.inMessageString.compare("DKVlpf") == 0) ////res
		{
			// --- (1) get the custom view interface via incoming message data*
			if (dynknob[6] != static_cast<ICustomView*>(messageInfo.inMessageData))
				dynknob[6] = static_cast<ICustomView*>(messageInfo.inMessageData);

			if (!dynknob[6]) return false;
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

	piParam = new PluginParameter(controlID::satcurrent, "current", "pA", controlVariableType::kDouble, .00252, 2000., 25, taper::kAntiLogTaper);
	piParam->setBoundVariable(&satcurrent_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::thermalvoltage, "thermalvoltage", "mV", controlVariableType::kDouble, 1, 200, 33, taper::kAntiLogTaper);
	piParam->setBoundVariable(&thermalvoltage_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipass, "highpass freq", "WDF, Biquad, off", "WDF");
	piParam->setBoundVariable(&hipass_p, boundVariableType::kInt);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::hipasscutoff, "highpass freq", "Hz", controlVariableType::kDouble, 1, 100, 10, taper::kAntiLogTaper);
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

	piParam = new PluginParameter(controlID::dccutoff, "dccut", "Hz", controlVariableType::kDouble, 5., 100., 10., taper::kAntiLogTaper);
	piParam->setBoundVariable(&dccut_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::lowshelfsag, "sag->inputlowshelf", "x", controlVariableType::kDouble, -1., 1., 0., taper::kLinearTaper);
	piParam->setBoundVariable(&feedback_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::lowshelfreq, "lowshelffreq", "Hz", controlVariableType::kDouble, 30, 300., 80., taper::kAntiLogTaper);
	piParam->setBoundVariable(&feedbacklow_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::fbthresh, "fbackthresh", "db", controlVariableType::kFloat, -36., 12., -12., taper::kLinearTaper);
	piParam->setBoundVariable(&fbthresh_p, boundVariableType::kFloat);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::release, "release", "ms", controlVariableType::kDouble, 50., 400., 200., taper::kAntiLogTaper);
	piParam->setBoundVariable(&release_p, boundVariableType::kDouble);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::sagVol, "sagvol", "x", controlVariableType::kFloat, 0., 5., 0.1, taper::kAntiLogTaper);
	piParam->setBoundVariable(&sagVol_p, boundVariableType::kFloat);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::sagVt, "sagvt", "x", controlVariableType::kFloat, -5., 5., 0.1, taper::kLinearTaper);
	piParam->setBoundVariable(&sagVt_p, boundVariableType::kFloat);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::sagLP, "saglp", "x", controlVariableType::kFloat, 0., 5., 0.1, taper::kAntiLogTaper);
	piParam->setBoundVariable(&sagLP_p, boundVariableType::kFloat);
	addPluginParameter(piParam);

	piParam = new PluginParameter(controlID::sagIntrim, "sag_in", "x", controlVariableType::kFloat, 0., 5., 0.1, taper::kAntiLogTaper);
	piParam->setBoundVariable(&sagintrim_p, boundVariableType::kFloat);
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



