// -----------------------------------------------------------------------------
//    ASPiK Custom Views File:  customviews.cpp
//
/**
    \file   customviews.cpp
    \author Will Pirkle
    \date   17-September-2018
    \brief  implementation file for example ASPiK custom view and custom sub-controller
    		objects
    		- http://www.aspikplugins.com
    		- http://www.willpirkle.com
*/
// -----------------------------------------------------------------------------
#include "customviews.h"

#include <iomanip>

namespace VSTGUI {



	float HueToRgb2(float p, float q, float t) {
		if (t < 0.0f) t += 1.0f;
		else if (t > 1.0f) t -= 1.0f;

		if (t < .166666f) return p + (q - p) * 6.0f * t;
		if (t < .5f) return q;
		if (t < .666666f) return p + (q - p) * (.66666f - t) * 6.0f;
		return p;
	}

	static  CColor HslToRgba2(float h, float s, float l, float alpha) {
		float r, g, b;

		//	if (s == 0.0f)
		//		r = g = b = l;
		//	else
		{
			float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
			float p = 2.0f * l - q;
			r = HueToRgb2(p, q, h + .33333f);
			g = HueToRgb2(p, q, h);
			b = HueToRgb2(p, q, h - .33333f);
		}

		return CColor((int)(r * 255.f), (int)(g * 255.f), (int)(b * 255.f), (int)(alpha * 255.f));
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


	template <class T>			//by me
	inline T FastAbs(const T& x) {
		T a = (x > 0.);

		return a * x + (-1. + a) * x;
	}


	inline float log2f_app(float X) {     ////http://openaudio.blogspot.com/2017/02/faster-log10-and-pow.html
		float Y, F;
		int E;
		F = frexpf(fabsf(X), &E);
		Y = 1.23149591368684f;
		Y *= F;
		Y += -4.11852516267426f;
		Y *= F;
		Y += 6.02197014179219f;
		Y *= F;
		Y += -3.13396450166353f;
		Y += E;
		return(Y);
	}



/**
\brief WaveView constructor

\param size - the control rectangle
\param listener - the control's listener (usuall PluginGUI object)
\param tag - the control ID value
*/
WaveView::WaveView(const VSTGUI::CRect& size, IControlListener* listener, int32_t tag)
: CControl(size, listener, tag)
, ICustomView()
{
    // --- create circular buffer that is same size as the window is wide
	circularBuffer = new double[(int)size.getWidth()];

    // --- init
	writeIndex = 0;
	readIndex = 0;
	circularBufferLength = (int)size.getWidth();
    memset(circularBuffer, 0, circularBufferLength*sizeof(double));
	paintXAxis = true;
	currentRect = size;

    // --- ICustomView
    // --- create our incoming data-queue
    dataQueue = new moodycamel::ReaderWriterQueue<double, DATA_QUEUE_LEN>;
}

WaveView::~WaveView()
{
    if(circularBuffer)
        delete [] circularBuffer;

    if(dataQueue)
        delete dataQueue;
}

void WaveView::pushDataValue(double data)
{
    if(!dataQueue) return;

    // --- add data point, make room if needed
    dataQueue->enqueue(data);
}

void WaveView::updateView()
{
    // --- get the max value that was added to the queue during the last
    //     GUI timer ping interval
    double audioSample = 0.0;
    double max = 0.0;
    bool success = dataQueue->try_dequeue(audioSample);
    if(success)
    {
        max = audioSample;
        while(success)
        {
            success = dataQueue->try_dequeue(audioSample);
            if(success && audioSample > max)
                max = audioSample;
        }

        // --- add to circular buffer
        addWaveDataPoint((float)fabs(max));
    }

    // --- this will set the dirty flag to repaint the view
    invalid();
}

void WaveView::addWaveDataPoint(float fSample)
{
	if(!circularBuffer) return;
	circularBuffer[writeIndex] = fSample;
	writeIndex++;
	if(writeIndex > circularBufferLength - 1)
		writeIndex = 0;
}

void WaveView::clearBuffer()
{
	if(!circularBuffer) return;
	memset(circularBuffer, 0, circularBufferLength*sizeof(float));
	writeIndex = 0;
	readIndex = 0;
}

void WaveView::draw(CDrawContext* pContext)
{
    // --- setup the backround rectangle
    int frameWidth = 1;
    int plotLineWidth = 1;
    pContext->setLineWidth(frameWidth);
    pContext->setFillColor(CColor(200, 200, 200, 255)); // light grey
    pContext->setFrameColor(CColor(0, 0, 0, 255)); // black
	CRect size = getViewSize();

    // --- draw the rect filled (with grey) and stroked (line around rectangle)
    pContext->drawRect(size, kDrawFilledAndStroked);

    // --- this will be the line color when drawing lines
    //     alpha value is 200, so color is semi-transparent
    pContext->setFrameColor(CColor(32, 0, 255, 200));
    pContext->setLineWidth(plotLineWidth);

    if(!circularBuffer) return;

    // --- step through buffer
    int index = writeIndex - 1;
    if(index < 0)
        index = circularBufferLength - 1;

    for(int i=1; i<circularBufferLength; i++)
    {
        double sample = circularBuffer[index--];

        double normalized = sample*(double)size.getHeight();
        if(normalized > size.getHeight() - 2)
            normalized = (double)size.getHeight();

        // --- so there is an x-axis even if no data
        if(normalized == 0) normalized = 0.1f;

        if (paintXAxis)
        {
            const CPoint p1(size.left + i, size.bottom - size.getHeight() / 2.f);
            const CPoint p2(size.left + i, size.bottom - size.getHeight() / 2.f - 1.0);
            const CPoint p3(size.left + i, size.bottom - size.getHeight() / 2.f + 1.0);

            // --- move and draw lines
            pContext->drawLine(p1, p2);
#ifndef MAC
            pContext->drawLine(p1, p3); // MacOS render is a bit different, this just makes them look consistent
#endif
        }

        // --- halves
        normalized /= 2.f;

        // --- find the three points of interest
        const CPoint p1(size.left + i, size.bottom - size.getHeight()/2.f);
        const CPoint p2(size.left + i, size.bottom - size.getHeight()/2.f - normalized);
        const CPoint p3(size.left + i, size.bottom - size.getHeight()/2.f + normalized);

        // --- move and draw lines
        pContext->drawLine(p1, p2);
        pContext->drawLine(p1, p3);

        // --- wrap the index value if needed
        if(index < 0)
            index = circularBufferLength - 1;
    }
}



SpectrumView2::SpectrumView2(const VSTGUI::CRect& size, IControlListener* listener, int32_t tag)
	: CControl(size, listener, tag)
{

	circularBuffer3 = new double[512];//(int)size.getWidth()];

// --- init
	writeIndex3 = 0;
	readIndex3 = 0;
	circularBufferLength3 = 16; //(int)size.getWidth();  /								///// must be length + 1
	memset(circularBuffer3, 0, 16 * sizeof(double));  // circularBufferLength*sizeof(double));
	currentRect3 = size;

	// --- ICustomView
	// --- create our incoming data-queue
	dataQueue3 = new moodycamel::ReaderWriterQueue<double, 512>;


	gradient = CGradient::create(CGradient::ColorStopMap(gradmapMid));

	for (float i = 0; i < 1; i = i + 0.05) {
		gradient->addColorStop(i, HslToRgba2( i * i * .7 -.04, .90, .5, 1.));
	}

	gradientMid = CGradient::create(CGradient::ColorStopMap(gradmapMid));

	for (float i = 0; i < 1; i = i + 0.05) {
		gradientMid->addColorStop(i, HslToRgba2( (1. - i )* .1 - .04, .90, .5, 1.));
	}
}

SpectrumView2::~SpectrumView2()
{
	if (dataQueue3)
		delete dataQueue3;

	if (circularBuffer3)
		delete[] circularBuffer3;

	if (gradient)
		delete gradient;

	if (gradientMid)
		delete gradientMid;
}



void SpectrumView2::pushDataValue(double data)
{
	if (!dataQueue3) return;

	// --- add data point, make room if needed
	dataQueue3->enqueue(data);
}

void SpectrumView2::updateView()
{

	double audioSample = 0.0;
	double max = 0.0;
	bool success = dataQueue3->try_dequeue(audioSample);
	if (success)
	{

		while (success)
		{
			success = dataQueue3->try_dequeue(audioSample);
				if (success) // && audioSample > max)
					addWaveDataPoint(audioSample);
		}
		// --- add to circular buffer
	}
	// --- this will set the dirty flag to repaint the view
	invalid();
}

void SpectrumView2::addWaveDataPoint(double fSample)
{
	if (!circularBuffer3) return;
	circularBuffer3[writeIndex3] = fSample;

	//directly store our new values 
		double sample = circularBuffer3[writeIndex3];

		int id = int(sample);
		sample = sample - id;

		switch (id) {

		case 2: {		// max audio in l;
			inLeft = sample * 10.;
			break;
		}
		case 4: {		// max audio in;
			inRight = sample * 10.;    //rr
			break;
		}
		case 6: {		//  rms audio in;
			rmsil = sample * 10.;
			break;
		}
		case 8: {		// rms audio in;
			rmsir = sample * 10.;
			break;
		}
		case 10: {		// max audio out;
			loutp = sample * 10.;
			break;
		}
		case 12: {		// max audio out;
			routp = sample * 10.;
			break;
		}
		case 14: {		// rms audio outleft;
			rmsol = sample * 10.;
			break;
		}
		case 16: {		// rms audio outright;
			rmsor = sample * 10.;
			break;
		}
		case 20: {		// 
			diffL = sample * 10.;
			break;
		}
		case 22: {		// 
			diffR = sample * 10.;
			break;
		}
		case 24: {		// 
			diffrmsR = sample * 10.;
			break;
		}
		case 26: {		// 
			diffrmsL = sample * 10.;
			break;
		}

		default: { break; }
		}

	writeIndex3++;
	if (writeIndex3 > circularBufferLength3 - 1)
		writeIndex3 = 0;
}

void SpectrumView2::draw(CDrawContext* pContext)
{
	pContext->setDrawMode(kAntiAliasing);

	// --- setup the backround rectangle
	CRect rect = getViewSize();
	pContext->setFillColor(kBlackCColor);

	if (!circularBuffer3) return;
	// --- step through buffer
	int index = writeIndex3 - 1;

	SharedPointer<CGraphicsPath> pth = owned(pContext->createGraphicsPath());
	if (pth == nullptr)
		return;
//	CGraphicsPath* pth = pContext->createGraphicsPath();
	pth->addRoundRect(rect, 5);
	pContext->drawGraphicsPath(pth, CDrawContext::kPathFilled);
//	pth->forget();
	
	const int holdtime = 28;				// peak hold time in Frames

	if (countrr > holdtime) { countrr = 0; }

	harr[countrr] = ((1.f - (FastMin(inLeft, inRight))));
	harro[countrr] = ((1.f - (FastMin(loutp, routp))));

	for (int j = 0; j < holdtime; j++) {			
		if (clpm < harr[j])
		{	clpm = harr[j]; 	}

		if (clpmo < harro[j])
		{	clpmo = harro[j];	}
	}

	rect.setWidth(rect.getWidth()*.4);			// .4 , .2, .4 split for the three segments

	//leave vertical room for text
	rect.inset(0, 13);				
	rect.offset(0, -7);				
	
	//left rect
	CRect inrect = rect;			
	inrect.inset(2, 0);				//inset a horizontal space
	inrect.offset(0.5, 0);

	//right rectr
	CRect outrect = inrect;		
	outrect.offset(1.5 * inrect.getWidth() + 4, 0);

	SharedPointer<CGraphicsPath> gradpath = owned(pContext->createGraphicsPath());
	if (gradpath == nullptr) return;
	//CGraphicsPath* gradpath = pContext->createGraphicsPath();
	gradpath->addRect(inrect);
	gradpath->addRect(outrect);
	pContext->fillLinearGradient(gradpath, *gradient, inrect.getTopCenter(), inrect.getBottomCenter());
//	gradpath->forget();

	//midrect
	CRect midrr = inrect;
	midrr.offset(inrect.getWidth() + 2, .0);
	midrr.setWidth(inrect.getWidth() * .5);

	SharedPointer<CGraphicsPath> mgradpath = owned(pContext->createGraphicsPath());
	if (mgradpath == nullptr) return;
//	CGraphicsPath* mgradpath = pContext->createGraphicsPath();
	mgradpath->addRect(midrr);
	pContext->fillLinearGradient(mgradpath, *gradientMid, midrr.getTopCenter(), midrr.getBottomCenter());
//	mgradpath->forget();


	inrect.offset(1, 0);			///Gradient seems off by 1 pixel
	outrect.offset(1, 0);
	midrr.offset(1, 0);


	pContext->setFrameColor(kBlackCColor);
	pContext->setLineWidth(.5);

	const float roof =   12.;			// start db 0 = 24db
	const float segments = 60;
	const float invsegment = 1. / segments;
	const float scale    = 1.7;			// meter vertical axis zoom
	const float rmsscale = 1.7;

	CColor fillcolour = kBlackCColor;
	const float drkshade = .8;
	const float midshade = 0.3;
	const float lightshade = 0.1;
	const float ligthestshade = 0.0;

	for (int i = 0; i < segments; i++) {

		ifrac = i * invsegment;
		ifracp = (i + 1) * invsegment;

		////input l/r meters 
		if (i == int(segments + (  roof - (scale*(1. - clpm) * segments)))) {			//Peak Hold Left
			fillcolour.setNormAlpha(ligthestshade );
		}
		else if (i > (segments + ( roof - (scale * inLeft * segments))) ){			//Shadow above peak
			fillcolour.setNormAlpha(drkshade);	
		}
		else if ( ( i > (segments + ( roof - (scale*rmsil* segments))) ) ) {			//Lit
			fillcolour.setNormAlpha(midshade);
		}
		else  {																			//RMS
			fillcolour.setNormAlpha(lightshade);
		}	
		
		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(inrect.getLeftCenter().x - 2, inrect.getBottomLeft().y - inrect.getHeight()*(ifrac),
									inrect.getCenter().x + .5,  inrect.getBottomLeft().y - inrect.getHeight()*(ifracp)), kDrawFilledAndStroked);

			////input right
		if (i == int(segments + ( roof - (scale*(1. - clpm) * segments)))) {			//Peak Hold
			fillcolour.setNormAlpha(ligthestshade);
		}
		else if (i > (segments + ( roof - (scale * inRight * segments)))) {			//Shadow above peak
			fillcolour.setNormAlpha(drkshade);
		}
		else if ((i > (segments + ( roof - (scale*rmsir* segments))))) {			//Lit
			fillcolour.setNormAlpha(midshade);
		}
		else {																		//RMS
			fillcolour.setNormAlpha(lightshade);
		}

		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(inrect.getCenter().x -.5 , inrect.getBottomLeft().y - inrect.getHeight()*(ifrac),
			inrect.getTopRight().x + .5, inrect.getBottomLeft().y - inrect.getHeight()*(ifracp)), kDrawFilledAndStroked);


		////Output Meters
		if (i == int(segments + ( roof - (scale*(1. - clpmo) * segments)))) {			//Peak Hold Left
			fillcolour.setNormAlpha(ligthestshade);
		}
		else if (i > (segments + ( roof - (scale * loutp * segments)))) {			//Shadow above peak
			fillcolour.setNormAlpha(drkshade);
		}
		else if ((i > (segments + ( roof - (scale*rmsol* segments))))) {			//Lit
			fillcolour.setNormAlpha(midshade);
		}
		else {																			//RMS
			fillcolour.setNormAlpha(lightshade);
		}

		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(outrect.getLeftCenter().x - 2, outrect.getBottomLeft().y - outrect.getHeight()*(ifrac),
			outrect.getCenter().x + .5, outrect.getBottomLeft().y - outrect.getHeight()*(ifracp)), kDrawFilledAndStroked);


			//// Output Right
		if (i == int(segments + ( roof - (scale*(1. - clpmo) * segments)))) {			//Peak Hold Right
			fillcolour.setNormAlpha(ligthestshade);
		}
		else if (i > (segments + ( roof - (scale * routp * segments)))) {			//Shadow above peak
			fillcolour.setNormAlpha(drkshade);
		}
		else if ((i > (segments + ( roof - (scale* rmsor * segments))))) {			// rms lit
			fillcolour.setNormAlpha(midshade);
		}
		else {																		//RMS
			fillcolour.setNormAlpha(lightshade);
		}

		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(outrect.getCenter().x - .5, outrect.getBottomLeft().y - outrect.getHeight()*(ifrac),
			outrect.getTopRight().x + .5, outrect.getBottomLeft().y - outrect.getHeight()*(ifracp)), kDrawFilledAndStroked);
	}

	for (int i = 0; i <= segments; i++) {				// diff meters
		ifrac = i * invsegment;
		ifracp = (i + 1) * invsegment;

		//left difference meter
		if (i > (segments + (1 - (rmsscale*(diffrmsL) * segments)))) {											//RMS
			fillcolour.setNormAlpha(ligthestshade);
		}
		else if (i > (segments + (1 - (rmsscale*(diffL)* segments)))) {
			fillcolour.setNormAlpha(midshade);																			//Peak
		}	
		else {
			fillcolour.setNormAlpha(drkshade);
		}

		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(midrr.getBottomLeft().x-1, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac),
								midrr.getBottomCenter().x +.5, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp)), kDrawFilledAndStroked);

		//right difference meter
		if (i > (segments + (1 - (rmsscale*(diffrmsR)* segments)))) {											//RMS
			fillcolour.setNormAlpha(lightshade);
		}
		else if (i > (segments + (1 - (rmsscale*(diffR)* segments)))) {
			fillcolour.setNormAlpha(midshade);																			//Peak
		}
		else {
			fillcolour.setNormAlpha(drkshade);
		}

		pContext->setFillColor(fillcolour);
		pContext->drawRect(CRect(midrr.getBottomCenter().x -.5, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac),
							   	midrr.getBottomRight().x+1, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp)), kDrawFilledAndStroked);
	}


	//text output
	const float fontsize = 14;
	const CFontRef fntt = new CFontDesc("OCRStd", fontsize, 1);
	pContext->setFillColor(kWhiteCColor);

	UTF8String readout;
	std::stringstream strng;
	strng << std::fixed;
	strng << std::setprecision(1);

	if (clpm < .999) {										// in Peak
		strng << (-(1. - (clpm)) * 100. + 24.);
		readout = strng.str();
	}
	else readout = "24+";

	CPoint textpos(inrect.getCenter().x - (fontsize* readout.length() * .25f) + 2, getViewSize().getBottomCenter().y - (.5*fontsize) + 2);
	pContext->drawString(readout, textpos, true);


	std::stringstream strng2;
	strng2 << std::fixed;
	strng2 << std::setprecision(1);

	if (clpmo < .99) {									// out Peak
	strng2 << ((-(1. - (clpmo)) * 100 + 24));
	readout = strng2.str();
	}
	else readout = "24+";

	textpos(outrect.getCenter().x - (fontsize* readout.length() * .25f), getViewSize().getBottomCenter().y - (.5*fontsize) + 2);
	pContext->drawString(readout, textpos, true);

	delete fntt;

	// --- Peak Hold Fall Time
	clpm  = FastMin((clpm - .0065f), 1.f);			
	clpmo = FastMin((clpmo - .0065f), 1.f);

	countrr++;

	// --- wrap the index value if neede
	if (index < 0)
		index = circularBufferLength3 - 1;

}



#ifdef HAVE_FFTW
/**
\brief SpectrumView constructor

\param size - the control rectangle
\param listener - the control's listener (usuall PluginGUI object)
\param tag - the control ID value
*/
SpectrumView::SpectrumView(const VSTGUI::CRect& size, IControlListener* listener, int32_t tag)
: CControl(size, listener, tag)
{
    // --- ICustomView
    // --- create our incoming data-queue
    dataQueue = new moodycamel::ReaderWriterQueue<double,FFT_LEN>;

    // --- double buffers for mag FFTs
    fftMagBuffersReady = new moodycamel::ReaderWriterQueue<double*,2>;
    fftMagBuffersEmpty = new moodycamel::ReaderWriterQueue<double*,2>;

    // --- load up the empty queue with buffers
    fftMagBuffersEmpty->enqueue(fftMagnitudeArray_A);
    fftMagBuffersEmpty->enqueue(fftMagnitudeArray_B);

    // --- buffer being drawn, only ever used by draw code
    currentFFTMagBuffer = nullptr;

    // --- FFTW inits
    data        = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_LEN);
    fft_result  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_LEN);
    ifft_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_LEN);

    plan_forward  = fftw_plan_dft_1d(FFT_LEN, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_1d(FFT_LEN, fft_result, ifft_result, FFTW_BACKWARD, FFTW_ESTIMATE);

    // --- window
    setWindow(spectrumViewWindowType::kBlackmanHarrisWindow);
}

SpectrumView::~SpectrumView()
{
    fftw_destroy_plan( plan_forward );
    fftw_destroy_plan( plan_backward );

    fftw_free( data );
    fftw_free( fft_result );
    fftw_free( ifft_result );

	if (dataQueue)
		delete dataQueue;

	if (fftMagBuffersReady)
		delete fftMagBuffersReady;

	if (fftMagBuffersEmpty)
		delete fftMagBuffersEmpty;
}

void SpectrumView::setWindow(spectrumViewWindowType _window)
{
    window = _window;
    memset(&fftWindow[0], 0, FFT_LEN*sizeof(double));

    // --- rectangular has fftWindow[0] = 0, fftWindow[FFT_LEN-1] = 0, all other points = 1.0
    if(window == spectrumViewWindowType::kRectWindow)
    {
        for (int n=0;n<FFT_LEN-1;n++)
            fftWindow[n] = 1.0;
    }
    else if(window == spectrumViewWindowType::kHannWindow)
    {
        for (int n=0;n<FFT_LEN;n++)
            fftWindow[n] = (0.5 * (1-cos((n*2.0*M_PI)/FFT_LEN)));
    }
    else if(window == spectrumViewWindowType::kBlackmanHarrisWindow)
    {
        for (int n=0;n<FFT_LEN;n++)
            fftWindow[n] = (0.42323 - (0.49755*cos((n*2.0*M_PI)/FFT_LEN))+ 0.07922*cos((2*n*2.0*M_PI)/FFT_LEN));
    }
    else // --- default to rectangular
    {
        for (int n=1;n<FFT_LEN-1;n++)
            fftWindow[n] = 1.0;
    }
}

bool SpectrumView::addFFTInputData(double inputSample)
{
    if(fftInputCounter >= FFT_LEN)
        return false;

    data[fftInputCounter][0] = inputSample*fftWindow[fftInputCounter]; // stick your audio samples in here
    data[fftInputCounter][1] = 0.0; // use this if your data is complex valued

    fftInputCounter++;
    if(fftInputCounter == FFT_LEN)
        return true; // ready for FFT

    return false;
}


void SpectrumView::pushDataValue(double data)
{
    if(!dataQueue) return;

    // --- add data point, make room if needed
    dataQueue->enqueue(data);
}

void SpectrumView::updateView()
{
    // --- grab samples from incoming queue and add to FFT input
    double audioSample = 0.0;
    bool success = dataQueue->try_dequeue(audioSample);
    if(!success) return;
    bool fftReady = false;
    while(success)
    {
        // --- keep adding values into the array; it will stop when full
        //     and return TRUE if the FFT buffer is full and we are ready
        //     to do a FFT
        if(addFFTInputData(audioSample))
            fftReady = true; // sticky flag

        // --- for this wave view, we can only show the FFT of the last
        //     512 points anyway, so we just keep popping them from the queue
        success = dataQueue->try_dequeue(audioSample);
    }

    if(fftReady)
    {
        // do the FFT
        fftw_execute(plan_forward);

        double* bufferToFill = nullptr;
        fftMagBuffersEmpty->try_dequeue(bufferToFill);

        if(!bufferToFill)
        {
            fftReady = false;
            fftInputCounter = 0;
            return;
        }

        int maxIndex = 0;
        for(int i=0; i<FFT_LEN; i++)
        {
            bufferToFill[i] = (getMagnitude(fft_result[i][0], fft_result[i][1]));
        }

        // --- normalize the FFT buffer for max = 1.0 (note this is NOT dB!!)
        normalizeBufferGetFMax(bufferToFill, 512, &maxIndex);

        // 1) homework = do plot in dB
        // 2) homework = add other windows
        // 3) homework = plot frequency on log x-axis (HARD!)

        // --- add the new FFT buffer to the queue
        fftMagBuffersReady->enqueue(bufferToFill);

        // --- set flags (can reduce number of flags?)
        fftReady = false;
        fftInputCounter = 0;
    }

    // --- this will set the dirty flag to repaint the view
    invalid();
}

void SpectrumView::draw(CDrawContext* pContext)
{
    // --- setup the backround rectangle
    int frameWidth = 1;
    int plotLineWidth = 1;
    pContext->setLineWidth(frameWidth);
    pContext->setFillColor(CColor(200, 200, 200, 255)); // light grey
    pContext->setFrameColor(CColor(0, 0, 0, 255)); // black

    // --- draw the rect filled (with grey) and stroked (line around rectangle)
	CRect size = getViewSize();
    pContext->drawRect(size, kDrawFilledAndStroked);

    // --- this will be the line color when drawing lines
    //     alpha value is 200, so color is semi-transparent
    pContext->setFrameColor(CColor(32, 0, 255, 200));
    pContext->setLineWidth(plotLineWidth);

    // --- is there a new fftBuffer?
    if(fftMagBuffersReady->peek())
    {
        // --- put current buffer in empty queue
        if(currentFFTMagBuffer)
            fftMagBuffersEmpty->try_enqueue(currentFFTMagBuffer);

        // --- get next buffer to plot
        fftMagBuffersReady->try_dequeue(currentFFTMagBuffer);
    }

    if(!currentFFTMagBuffer)
        return;

    // --- plot the FFT data
    double step = 128.0/size.getWidth();
    double magIndex = 0.0;

    // --- plot first point
    double yn = currentFFTMagBuffer[0];
    double ypt = size.bottom - size.getHeight()*yn;

    // --- make sure we leave room for bottom of frame
	if (ypt > size.bottom - frameWidth)
		ypt = size.bottom - frameWidth;

    // --- setup first point, which is last point for loop below
    CPoint lastPoint(size.left, ypt);

    for (int x = 1; x < size.getWidth()-1; x++)
    {
        // --- increment stepper for mag array
        magIndex += step;

        // --- interpolate to find magnitude at this step
        yn = interpArrayValue(currentFFTMagBuffer, 128, magIndex);

        // --- calculate top (y) value of point
        ypt = size.bottom - size.getHeight()*yn;

        // --- make sure we leave room for bottom of frame
		if (ypt > size.bottom - frameWidth)
			ypt = size.bottom - frameWidth;

        // --- create a graphic point to this location
        const CPoint p2(size.left + x, ypt);

        // --- filled FFT is a set of vertical lines that touch
        if(filledFFT)
        {
            // --- find bottom point
			CPoint bottomPoint(size.left + x, size.bottom - frameWidth);

            // --- draw vertical line
            pContext->drawLine(bottomPoint, p2);
        }
        else // line-FFT
        {
            // --- move and draw line segment
            pContext->drawLine(lastPoint, p2);

            // --- save for next segment
            lastPoint.x = p2.x;
            lastPoint.y = p2.y;
        }
    }
}

#endif

/**
\brief CustomKnobView constructor

\param size - the control rectangle
\param listener - the control's listener (usuall PluginGUI object)
\param tag - the control ID value
\param subPixmaps - the number of frames in the strip animation
\param heightOfOneImage- the number of frames in the strip animation
\param background- the graphics file for the control
\param offset- positional offset value
\param bSwitchKnob - flag to enable switch knob
*/
CustomKnobView::CustomKnobView (const CRect& size, IControlListener* listener, int32_t tag, int32_t subPixmaps, CCoord heightOfOneImage, CBitmap* background, const CPoint &offset, bool bSwitchKnob)
: CAnimKnob (size, listener, tag, subPixmaps, heightOfOneImage, background, offset)
{
    // --- ICustomView
    // --- create our incoming data-queue
    dataQueue = new moodycamel::ReaderWriterQueue<CustomViewMessage, 32>;
}

CustomKnobView::~CustomKnobView(void)
{
    if(dataQueue) delete dataQueue;
}

void CustomKnobView::sendMessage(void* data)
{
    CustomViewMessage* viewMessage = (CustomViewMessage*)data;
    
    // --- example of messaging: plugin core send message, we acknowledge
    if (viewMessage->message == MESSAGE_QUERY_CONTROL)
    {
     //   if (viewMessage->queryString.compare("Hello There!") == 0)
   //     {
        //    viewMessage->replyString.assign("I'm Here!!");
         //   viewMessage->messageData = this; // <??? example of VERY risky thing to do; not recommended
    //    }
    }

    // --->> CustomViewMessage has =operator
	  dataQueue->enqueue(*viewMessage);
}

void CustomKnobView::updateView()
{
    CustomViewMessage viewMessage;
    bool success = dataQueue->try_dequeue(viewMessage);
    while(success)
    {
        // --- keep popping the queue in case there were multiple message insertions
		success = dataQueue->try_dequeue(viewMessage); 
		str = viewMessage.queryString;
    }

    // --- force redraw
    invalid();
}

void CustomKnobView::draw(CDrawContext* pContext) {

	const float fontsize = 22;
	const CFontRef fntt = new CFontDesc("Arial", fontsize, 1);
	pContext->setFillColor(kBlackCColor);

	pContext->drawEllipse(getViewSize(), kDrawStroked);

	CPoint point = getViewSize().getCenter();
	pContext->drawString(str, point, true);

	delete fntt;
	return;
}




CustomKnobView2::CustomKnobView2(const VSTGUI::CRect& size, IControlListener* listener, int32_t tag) : CKnob(size, nullptr, 0, nullptr, nullptr)
{
	// --- ICustomView
	// --- create our incoming data-queue
//	dataQueue = new moodycamel::ReaderWriterQueue<CustomViewMessage, 32>;
}

CustomKnobView2::~CustomKnobView2(void)
{
	//if (dataQueue) delete dataQueue;
}

/*
void CustomKnobView2::sendMessage(void* data)
{
	CustomViewMessage* viewMessage = (CustomViewMessage*)data;

	// --- example of messaging: plugin core send message, we acknowledge
	if (viewMessage->message == MESSAGE_QUERY_CONTROL)
	{
		//   if (viewMessage->queryString.compare("Hello There!") == 0)
	  //     {
		   //    viewMessage->replyString.assign("I'm Here!!");
			//   viewMessage->messageData = this; // <??? example of VERY risky thing to do; not recommended
	   //    }
	}

	// --->> CustomViewMessage has =operator
	dataQueue->enqueue(*viewMessage);
}

void CustomKnobView2::updateView()
{
	CustomViewMessage viewMessage;
	bool success = dataQueue->try_dequeue(viewMessage);
	while (success)
	{
		// --- keep popping the queue in case there were multiple message insertions
		success = dataQueue->try_dequeue(viewMessage);
		str = viewMessage.queryString;
	}

	// --- force redraw
	invalid();
}
*/

void CustomKnobView2::draw(CDrawContext* pContext) {

	pContext->setDrawMode(kAntiAliasing );
	
	CRect r = getViewSize();
	r.inset(3,3);
	CCoord w = r.getWidth();
	CCoord h = r.getHeight();	

	////init - generate radial marks
	if (getViewSize() != oldsize) 
	{	
		mrkpth = owned(pContext->createGraphicsPath());

	//	mrkpth->beginSubpath(r.getCenter());

		for (int i = 0; i < 11; i++) {
			float ifrac = float(i) / 10;

		//	mrkpth->addArc(r, -225, -225 + 270 * ifrac, 1);
			CPoint  mrk(r.getCenter().x + .5*r.getWidth()*sin(((1.5* ifrac) - .75) * Constants::pi),
						r.getCenter().y - .5*r.getHeight()*cos(((1.5* ifrac) - .75) * Constants::pi));

			mrkpth->addEllipse(CRect(mrk, CPoint(1, 1)));
	//		mrkpth->addLine(r.getCenter());
//			pContext->drawLine(mrk, r.getCenter());
		}

	//	mrkpth->addArc(r, -225, -225 + 270 * 1, 1);
	//	mrkpth->closeSubpath();
		oldsize = getViewSize();
	}
	
	pContext->setLineWidth(2);

	//outer arc
//	auto path = owned(pContext->createGraphicsPath());
//	auto path2 = owned(pContext->createGraphicsPath());
//	path->addArc(r, -225, -225 + 270 * getValueNormalized(), 1);
//	path2->addArc(r, -225 + 270 * getValueNormalized(), 45, 1);

//	pContext->setFrameColor(coronaColor); 
//	pContext->drawGraphicsPath(path, CDrawContext::kPathStroked);       // arc value shaded


	CColor transblck(kBlackCColor);
	transblck.setNormAlpha(.3);
	pContext->setFrameColor(transblck);
	pContext->drawGraphicsPath(mrkpth, CDrawContext::kPathStroked);

	pContext->setDrawMode(kAliasing);

	//back shadow
//		CRect rr = r;
//		rr.inset(5,5); rr.offset(0, 3);
//		auto pathsh = owned(pContext->createGraphicsPath());
//		pathsh->addEllipse(rr);
//		pContext->fillRadialGradient(pathsh, *shd, rr.getCenter(), .5*rr.getWidth());

	//cap
	CColor drkcap = getColorShadowHandle();
	double hue, sat, lum;
	drkcap.toHSV(hue, sat, lum);
	lum *= 0.85;
	drkcap.fromHSV(hue, sat, lum);


	CRect inner_r = r;
	inner_r.inset(9, 9);

	pContext->setDrawMode(kAntiAliasing);

	r.inset(7, 7);
	pContext->setFillColor(getColorShadowHandle());

	// cap
	pContext->drawEllipse(r, kDrawFilled);

	// grooves
	CLineStyle lineStyle(kLineOnOffDash);
	lineStyle.getDashLengths() = coronaLineStyle.getDashLengths();
	lineStyle.setLineCap(CLineStyle::kLineCapRound);
	pContext->setFrameColor(drkcap);
	pContext->setLineStyle(lineStyle);

	pContext->setLineWidth(6);
	float startoffset = getStartAngle();
	pContext->drawArc(inner_r, getStartAngle() + 270 * getValueNormalized(), getStartAngle() + 270 * getValueNormalized() - 1, kDrawStroked);

	pContext->setDrawMode(kAliasing);
	//cap gradient
//		auto pathsh3 = owned(pContext->createGraphicsPath());
//		pathsh3->addEllipse(r);
//		pContext->fillLinearGradient(pathsh3, *lightshde, r.getTopLeft(), r.getBottomRight());
//	
//		r.inset(-5, -5);
//		r.offset(2, 3);
//	
//		auto pathsh2 = owned(pContext->createGraphicsPath());	
//		pathsh2->addEllipse(r);
//		pContext->fillRadialGradient(pathsh2, *drkshd, r.getCenter(), r.getWidth()*.5f);

	pContext->setDrawMode(kAntiAliasing);

	//handle
	pContext->setLineWidth(handleLineWidth);
	pContext->setLineStyle(CLineStyle(CLineStyle::kLineCapRound));
	float hndlngth = getCoronaOutlineWidthAdd() ;

	float hndlx = cosf((getValueNormalized() + 0.5)*M_PI*1.5 ) * inner_r.getWidth();
	float hndly = sinf((getValueNormalized() + 0.5)*M_PI*1.5 ) * inner_r.getWidth();

	CPoint hndl  ( (0.5f * hndlx ) + inner_r.getCenter().x, inner_r.getCenter().y + (0.5f * hndly ));
	CPoint hndlin( (hndlngth * hndlx ) + inner_r.getCenter().x, inner_r.getCenter().y + (hndlngth * hndly));
	
	pContext->setFrameColor(colorHandle);
	pContext->drawLine(hndl, hndlin);		


	setDirty(false);

	return;
}


class drawKnob {
public:
	drawKnob() {};
	~drawKnob() {};

void drawknob(CDrawContext* pContext, CRect Size, float getValueNormalized, CColor coronaColor, CColor colorHandle,
	CColor getColorShadowHandle, CLineStyle coronaLineStyle, CCoord handleLineWidth, const CPoint valueToPoint, float str) {

	pContext->setDrawMode(kAntiAliasing | kNonIntegralMode);
	
	CRect r = Size;
	r.inset(3, 3);
	CCoord w = r.getWidth();
	CCoord h = r.getHeight();

	//	pContext->setFrameColor(kGreyCColor);
	//	pContext->drawGraphicsPath(patho, CDrawContext::kPathStroked);		// arc groove


		////init - generate radial marks
	if (Size != oldsize)
	{
		mrkpth = owned(pContext->createGraphicsPath());

		//	mrkpth->beginSubpath(r.getCenter());

		for (int i = 0; i < 11; i++) {
			float ifrac = float(i) / 10;

			//	mrkpth->addArc(r, -225, -225 + 270 * ifrac, 1);
			CPoint  mrk(r.getCenter().x + .5*r.getWidth()*sin(((1.5* ifrac) - .75) * Constants::pi),
				r.getCenter().y - .5*r.getHeight()*cos(((1.5* ifrac) - .75) * Constants::pi));

			mrkpth->addEllipse(CRect(mrk, CPoint(1, 1)));
			//		mrkpth->addLine(r.getCenter());
		//			pContext->drawLine(mrk, r.getCenter());
		}

		//	mrkpth->addArc(r, -225, -225 + 270 * 1, 1);
		//	mrkpth->closeSubpath();
		oldsize = Size;
	}

	pContext->setLineWidth(2);

	//outer arc
//	auto path = owned(pContext->createGraphicsPath());
//	auto path2 = owned(pContext->createGraphicsPath());

//	path->addArc(r, -225, -225 + 270 * getValueNormalized, 1);
//	path2->addArc(r, -225 + 270 * getValueNormalized, 45, 1);

//	pContext->setFrameColor(coronaColor);
//		pContext->drawGraphicsPath(path, CDrawContext::kPathStroked);       // arc value shaded
	

	CColor transblck(kBlackCColor);
	transblck.setNormAlpha(.3);
	pContext->setFrameColor(transblck);
	pContext->drawGraphicsPath(mrkpth, CDrawContext::kPathStroked);

	//back shadow
	CRect rr = r;
	rr.inset(5, 5); rr.offset(0, 3);
	auto pathsh = owned(pContext->createGraphicsPath());
	SharedPointer <CGradient> shd = owned(CGradient::create(0.8, 1, CColor(0, 00, 00, 70), CColor(0, 0, 0, 0)));
	pathsh->addEllipse(rr);
	pContext->fillRadialGradient(pathsh, *shd, rr.getCenter(), .5*rr.getWidth());

	//cap
	CColor drkcap = getColorShadowHandle;
	double hue, sat, lum;
	drkcap.toHSV(hue, sat, lum);
	lum *= 0.85;
	drkcap.fromHSV(hue, sat, lum);

	CLineStyle lineStyle(kLineOnOffDash);
	lineStyle.getDashLengths() = coronaLineStyle.getDashLengths();
	lineStyle.setLineCap(CLineStyle::kLineCapRound);
	pContext->setFrameColor(drkcap);
	pContext->setLineStyle(lineStyle);
	pContext->setLineWidth(6);
	CRect inner_r = r;
	inner_r.inset(9, 9);

	r.inset(7, 7);
	pContext->setFillColor(getColorShadowHandle);

	// cap
	pContext->drawEllipse(r, kDrawFilled);

	// grooves
	pContext->drawArc(inner_r, 270 * getValueNormalized, 270 * getValueNormalized - 1, kDrawStroked);


	//cap gradient
	auto pathsh3 = owned(pContext->createGraphicsPath());
	SharedPointer <CGradient> shd3 = owned(CGradient::create(.0, .9, CColor(250, 250, 250, 80), CColor(0, 0, 0, 0)));
	pathsh3->addEllipse(r);
	pContext->fillLinearGradient(pathsh3, *shd3, r.getTopLeft(), r.getBottomRight());

	r.inset(-5, -5);
	r.offset(2, 3);
	//	pContext->setFillColor(kGreyCColor);
	auto pathsh2 = owned(pContext->createGraphicsPath());
	SharedPointer <CGradient> shd2 = owned(CGradient::create(.7, 1., CColor(0, 00, 00, 80), CColor(0, 0, 0, 0)));
	pathsh2->addEllipse(r);
	pContext->fillRadialGradient(pathsh2, *shd2, r.getCenter(), r.getWidth()*.5f);

	//handle
	CPoint where(valueToPoint);
	//valueToPoint(where);

	CPoint origin(Size.getWidth() * .5,Size.getHeight() * .5);
	where.offset(Size.left - 1, Size.top);
	origin.offset(Size.left - 1, Size.top);

	pContext->setLineWidth(handleLineWidth);
	pContext->setLineStyle(CLineStyle(CLineStyle::kLineCapRound));
	//	pContext->drawLine(where, origin);

	where.offset(1, -1);
	origin.offset(1, -1);

	CPoint wh2(((where.x * .8 + origin.x * .2)), ((where.y * .8 + origin.y * .2)));

	pContext->setFrameColor(colorHandle);
	pContext->drawLine(where, wh2);


//	pContext->setFrameColor(kRedCColor);
//	pContext->drawString(str, Size.getCenter(), true);

//	setDirty(false);

	return;
}

private:
	CRect oldsize;
	SharedPointer<VSTGUI::CGraphicsPath> mrkpth = nullptr;
};




DynamicKnobView::DynamicKnobView(const VSTGUI::CRect& size, IControlListener* listener, int32_t tag) : CKnob(size, nullptr, 0, nullptr, nullptr)
{
	// --- ICustomView
	// --- create our incoming data-queue
	dataQueue = new moodycamel::ReaderWriterQueue<CustomViewMessage, 2>;
}

DynamicKnobView::~DynamicKnobView(void)
{
	if (dataQueue) delete dataQueue;
}

void DynamicKnobView::sendMessage(void* data)
{
	CustomViewMessage* viewMessage = (CustomViewMessage*)data;

	// --->> CustomViewMessage has =operator
	dataQueue->enqueue(*viewMessage);
}

void DynamicKnobView::updateView()
{
	CustomViewMessage viewMessage;
	bool success = dataQueue->try_dequeue(viewMessage);
	while (success)
	{
		// --- keep popping the queue in case there were multiple message insertions
		success = dataQueue->try_dequeue(viewMessage);
		str = viewMessage.queryString;
	}
	// --- force redraw

//	DynRng.updateDynRng(str);

//	invalidRect(oldsize);
	invalid();
}

void DynamicKnobView::draw(CDrawContext* pContext) {

	CPoint valtopoint;
	valueToPoint(valtopoint);

		float strngv = strtof(str, NULL );
		
		
	//	DynRng.drawDynRing(pContext, str, drawStyle, getViewSize());
	
		int cntr = 0;
		if (drawStyle & kCoronaFromCenter) { 
					cntr += 135; 
					strngv *= .5;	
		}
	
		// gui rate smoother 1 frame = ~30ms
			//	smoothval = .9 * smoothval + .1 * s;
			//	s = smoothval;
	
		int clockwise = 1 - signbit(strngv);
		auto path = owned(pContext->createGraphicsPath());
		CRect dynvalr = getViewSize();
		dynvalr.inset(6, 6);
		path->addArc(dynvalr, -225 + cntr, -225 + cntr + 270 * strngv, clockwise);
	
		CColor dyncol(200, 100, 100, 100);
		pContext->setFrameColor(dyncol);
		pContext->setLineWidth(3.f);
		pContext->drawGraphicsPath(path, CDrawContext::kPathStroked);

//	drawKnob drw;
//		drw.drawknob(pContext, getViewSize(), getValueNormalized(), coronaColor,
//			colorHandle, getColorShadowHandle(), coronaLineStyle, handleLineWidth, valtopoint, strngv);


	setDirty(false);

	return;
}



}
