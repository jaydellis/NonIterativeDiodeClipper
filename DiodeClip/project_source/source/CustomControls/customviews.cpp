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

}

SpectrumView2::~SpectrumView2()
{
	if (dataQueue3)
		delete dataQueue3;

	if (circularBuffer3)
		delete[] circularBuffer3;
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
			//	if (success) // && audioSample > max)
				//	addWaveDataPoint((max));

			if (!circularBuffer3) return;
			circularBuffer3[writeIndex3] = audioSample;
			writeIndex3++;
			if (writeIndex3 > circularBufferLength3 - 1)
				writeIndex3 = 0;

		}

		// --- add to circular buffer

	}

	// --- this will set the dirty flag to repaint the view
	invalid();

}

void SpectrumView2::draw(CDrawContext* pContext)
{

	pContext->setDrawMode(kAntiAliasing);

	// --- setup the backround rectangle

	CRect size = getViewSize();

	float sample = 0.0f;
	float colaudio = 0.0f;

	if (!circularBuffer3) return;
	// --- step through buffer
	int index = writeIndex3 - 1;

	if (index < 0)
		index = circularBufferLength3 - 1;

	pContext->setFillColor(CColor(0, 00, 0, 250)); // not grey was 200,200,200
	//pContext->drawRect(size, kDrawFilled); 
	auto pth = pContext->createGraphicsPath();
	pth->addRoundRect(size, 5);
	pContext->drawGraphicsPath(pth, CDrawContext::kPathFilledEvenOdd);
	pth->forget();

	for (int j = 0; j < circularBufferLength3; j++) {

		sample = circularBuffer3[j];

		int id = int(sample);
		sample = sample - id;

		switch (id) {

		case 2: {		// max audio in l;
			val2 = sample * 10.;
			break;
		}
		case 4: {		// max audio in;
			valold = sample * 10.;    //rr
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
		case 10: {		// max audio in;
			loutp = sample * 10. ;
			break;
		}
		case 12: {		// max audio in;
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

	}

	countrr++;

	if (countrr > 30) { countrr = 0; }

	pContext->setLineWidth(.5);  // .5

	harr[countrr] = ((1.f - (FastMin(valold, val2))));
	harro[countrr] = ((1.f - (FastMin(loutp, routp))));


	for (int j = 0; j < 30; j++) {			// max j is hold time
		if (clpm < harr[j])
		{
			clpm = harr[j];
		}

		if (clpmo < harro[j])
		{
			clpmo = harro[j];
		}
	}

	rr = getViewSize();

	rr.setWidth(rr.getWidth()*.4);			// .4 , .2, .4 split

	//rr.inset(rr.getWidth()*.5, 0);

	rr.inset(0, 15);
	rr.offset(0, -14);

	srr = rr;
	srr.inset(3, 0);
	srr.offset(1, 9);

	const float roof =  + 18.;

	for (int i = 0; i <= 40; i++) {			//input l/r meters 

		ifrac = i * (1. / 40.f);
		ifracp = (i + 1) * (1. / 40.f);		//	float randv = float(rand()) / RAND_MAX;  float randb = float(rand()) / RAND_MAX;


		if (i < (40. + (roof - (2.5f*val2 * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .45f, .5f);
		}
		else {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .15f, 1.f);
		}

		if (i == int(40. + (roof - (2.5f*(1.f - clpm) * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .75f);
		}

		if (i == int(40. + (roof - (2.5f*rmsil* 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .7f);
		}

		pContext->setFillColor(cc);

		b(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifrac));
		a(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifracp));

		pContext->drawRect(CRect(srr.getLeftCenter().x, b.y, srr.getCenter().x, a.y), kDrawFilled);		//L


		if (i < (40. + (roof - (2.5 *valold * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .45f, .5f);
		}
		else {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .15f, 1.f);
		}
		if (i == int(40. + (roof - (2.5f*(1.f - clpm) * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .75f);
		}

		if (i == int(40. + (roof - (2.5f*rmsir * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .7f);
		}

		pContext->setFillColor(cc);

		b(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifrac));
		a(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifracp));

		pContext->drawRect(CRect(srr.getCenter().x, b.y, srr.getBottomRight().x, a.y), kDrawFilled);		//R
	}

	CRect midrr = srr;
	midrr.offset(srr.getWidth() + 2.5, .0);

	srr.offset(srr.getWidth() + 8 + srr.getWidth()*.5, 0);

	for (int i = 0; i <= 40; i++) {			//output l/r meters

		ifrac = i * (1. / 40.f);
		ifracp = (i + 1) * (1. / 40.f);		//	float randv = float(rand()) / RAND_MAX;  float randb = float(rand()) / RAND_MAX;


		if (i < (40. + (roof - (2.5f*loutp * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .45f, .5f);
		}
		else {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .15f, 1.f);
		}

		if (i == int(40. + (roof - (2.5f*(1.f - clpmo) * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .75f);
		}

		if (i == int(40. + (roof - (2.5f*rmsol* 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .7f);
		}

		pContext->setFillColor(cc);

		b(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifrac));
		a(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifracp));

		pContext->drawRect(CRect(srr.getLeftCenter().x, b.y, srr.getCenter().x, a.y), kDrawFilled);


		if (i < (40. + (roof - (2.5f*routp * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .45f, .5f);
		}
		else {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .15f, 1.f);
		}
		if (i == int(40. + (roof - (2.5f*(1.f - clpmo) * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .7f);
		}
		if (i == int(40. + (roof - (2.5f*rmsor * 40.f)))) {
			cc = HslToRgba2((.9f - ifrac)*.60f, .99f, .5f, .75f);
		}

		pContext->setFillColor(cc);

		b(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifrac));
		a(srr.getBottomLeft().x, srr.getBottomLeft().y - srr.getHeight()*(ifracp));

		pContext->drawRect(CRect(srr.getCenter().x, b.y, srr.getBottomRight().x, a.y), kDrawFilled);
	}

	midrr.setWidth(size.getWidth() * .2);
//	midrr.offset(0, 0);

	for (int i = 0; i <= 40; i++) {				// diff meters
		ifrac = i * (1. / 40.f);
		ifracp = (i + 1) * (1. / 40.f);

	

		if (i > (40. + (1 - (2.5f*(diffrmsL) * 40.f)))) {						//L
			cc = HslToRgba2((1.f - ifrac)*.05f, .99f, .45f  * ifrac + .05, (1.f - ifrac)*.55f + .4);
		}
		else {
			cc = HslToRgba2((1.f - ifrac)*.05f + .01f, .99f, .15f * ifrac , (1.f - ifrac)*.50f + .4);
		}

		pContext->setFillColor(cc);

		b(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac));
		a(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp));

		pContext->drawRect(CRect(midrr.getBottomLeft().x, b.y, midrr.getBottomCenter().x, a.y), kDrawFilled);

		if (i > (40. + (1 - (2.5f*(diffL) * 40.f)))) {
			cc = HslToRgba2((1.f - ifrac)*.05f, .99f, .45, (1.f - ifrac)*.55f+.4);

			pContext->setFillColor(cc);

			b(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac));
			a(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp));

			pContext->drawRect(CRect(midrr.getBottomLeft().x, b.y, midrr.getBottomCenter().x, a.y), kDrawFilled);
		}



		if (i > (40. + (1 - (2.5f*(diffrmsR) * 40.f)))) {				//R
			cc = HslToRgba2((1.f - ifrac)*.05f, .99f, .45f  * ifrac + .05, .55f);
		}
		else {
			cc = HslToRgba2((1.f - ifrac)*.05f + .01f, .99f, .15f * ifrac, 0.5f);
		}

		pContext->setFillColor(cc);

		b(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac));
		a(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp));

		pContext->drawRect(CRect(midrr.getBottomCenter().x, b.y, midrr.getBottomRight().x, a.y), kDrawFilled);

		if (i > (40. + (1 - (2.5f*(diffR) * 40.f)))) {
			cc = HslToRgba2((1.f - ifrac)*.05f, .99f, .45f, .55f);

			pContext->setFillColor(cc);

			b(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifrac));
			a(midrr.getBottomLeft().x, midrr.getBottomLeft().y - midrr.getHeight()*(ifracp));

			pContext->drawRect(CRect(midrr.getBottomCenter().x, b.y, midrr.getBottomRight().x, a.y), kDrawFilled);
		}

	}



	if (clpm < .999) {
		g = toString(int(-(1. - (clpm)) * 100 + 24));
	}
	else g = "24+";

	const CFontRef fntt = new CFontDesc("OCRStd", 14, 1);

	h = 14;

	CPoint cp(rr.getCenter().x - (h* g.length() * .25f), getViewSize().getBottomCenter().y - (.5*h) + 2);
	pContext->setFillColor(kWhiteCColor);
	pContext->drawString(g, cp, true);

	if (clpmo < .99) {
	g = toString(int(-(1. - (clpmo)) * 100 + 24));
	}
	else g = "12+";

	cp(srr.getCenter().x - (h* g.length() * .25f), getViewSize().getBottomCenter().y - (.5*h) + 2);
	pContext->drawString(g, cp, true);

	delete fntt;

	clpm = FastMin((clpm - .0065f), 1.f);			//FallTimes
	clpmo = FastMin((clpmo - .0065f), 1.f);

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
        if (viewMessage->queryString.compare("Hello There!") == 0)
        {
            viewMessage->replyString.assign("I'm Here!!");
            viewMessage->messageData = this; // <– example of VERY risky thing to do; not recommended
        }
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
        // --- not connected, but example of setting control's appearance via message
        //     you could use this to also show/hide the control, or move it to a new location
        if(viewMessage.message == MESSAGE_SET_CONTROL_ALPHA)
        {
            setAlphaValue((float)viewMessage.controlAlpha);
        }

        // --- keep popping the queue in case there were multiple message insertions
        success = dataQueue->try_dequeue(viewMessage);
    }

    // --- force redraw
    invalid();
}



}
