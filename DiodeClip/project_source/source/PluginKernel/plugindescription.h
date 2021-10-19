// --- CMAKE generated variables for your plugin

#include "pluginstructures.h"

#ifndef _plugindescription_h
#define _plugindescription_h

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define AU_COCOA_VIEWFACTORY_STRING STR(AU_COCOA_VIEWFACTORY_NAME)
#define AU_COCOA_VIEW_STRING STR(AU_COCOA_VIEW_NAME)

// --- AU Plugin Cocoa View Names (flat namespace) 
#define AU_COCOA_VIEWFACTORY_NAME AUCocoaViewFactory_BA464D50DA2B375EAC5AB3D6B0F06B60
#define AU_COCOA_VIEW_NAME AUCocoaView_BA464D50DA2B375EAC5AB3D6B0F06B60

// --- BUNDLE IDs (MacOS Only) 
const char* kAAXBundleID = "developer.aax.diodeclip.bundleID";
const char* kAUBundleID = "developer.au.diodeclip.bundleID";
const char* kVST3BundleID = "developer.vst3.diodeclip.bundleID";

// --- Plugin Names 
const char* kPluginName = "DiodeClip";
const char* kShortPluginName = "DiodeClip";
const char* kAUBundleName = "DiodeClip_AU";
const char* kAAXBundleName = "DiodeClip_AAX";
const char* kVSTBundleName = "DiodeClip_VST";

// --- bundle name helper 
inline static const char* getPluginDescBundleName() 
{ 
#ifdef AUPLUGIN 
	return kAUBundleName; 
#endif 

#ifdef AAXPLUGIN 
	return kAAXBundleName; 
#endif 

#ifdef VSTPLUGIN 
	return kVSTBundleName; 
#endif 

	// --- should never get here 
	return kPluginName; 
} 

// --- Plugin Type 
const pluginType kPluginType = pluginType::kFXPlugin;

// --- VST3 UUID 
const char* kVSTFUID = "{ba464d50-da2b-375e-ac5a-b3d6b0f06b60}";

// --- 4-char codes 
const int32_t kFourCharCode = 'dclp';
const int32_t kAAXProductID = 'dclp';
const int32_t kManufacturerID = 'modb';

// --- Vendor information 
const char* kVendorName = "modb";
const char* kVendorURL = "www.modb.com";
const char* kVendorEmail = "modb@myplugins.com";

// --- Plugin Options 
const bool kProcessFrames = false;
const uint32_t kBlockSize = 0;
const bool kWantSidechain = false;
const uint32_t kLatencyInSamples = 0;
const double kTailTimeMsec = 0.000000;
const bool kVSTInfiniteTail = false;
const bool kVSTSAA = false;
const uint32_t kVST3SAAGranularity = 1;
const uint32_t kAAXCategory = 0;

#endif
