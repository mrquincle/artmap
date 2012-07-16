/*
 * art.h is the interface of the base Art class
 * This Art class implements the Adaptive Resonance Theory network
 * for on-going plastic and stable associative learning (S. Grossberg and G. Carpenter)
 *
 * This ART network has match-track capabilities for ARTMAP incorporation.
 * The network is also able to return distributed output as well as WTA
 *
 *  Created on: Sep 2, 2009
 *      Author: ted
 */

#ifndef ART_H_
#define ART_H_

#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <fstream>


namespace almendeSensorFusion
{

/**************************************************************************************************************
 * Type definitions that make it easier to understand the code
 *
 * They are not always necessary and sometimes different typedefs are used for the same data structure to
 * tell what something really does
 *************************************************************************************************************/

//! We just use floats here as individual inputs to our nodes
typedef float ART_TYPE;

//! The "prototype" corresponding to a node in F2 (long-term memory) is stored as a vector of
//! weights from all F1 nodes to the given node.
typedef std::vector<ART_TYPE> PROTOTYPE;

//! An "aspect" is a mono-modal view of a perceivable "object" using one (sub)modality
typedef std::vector< ART_TYPE> ART_ASPECT;

//! Multiple features can also be seen as a "view" of an object using "aspects" from multiple modalities
typedef std::vector< ART_ASPECT*> ART_VIEW;

//! A class in ART is not just "1" value, it is also represented by a (distributed) vector of values
//! For all practical purposes, it is also fine to use a vector (1,0) for one class and (0,1) for another.
typedef std::vector< ART_TYPE> ART_DISTRIBUTED_CLASS;

//! Multiple classes (each represented in a distributed manner)
typedef std::vector< ART_DISTRIBUTED_CLASS*> ART_DISTRIBUTED_CLASSES;

//! There can be multiple "views" of the same object (over time or from different directions)
typedef std::vector< ART_VIEW*> ART_VIEWS;

enum ART_COMPUTATION_TYPE
{
	DEFAULT_ARTMAP,
	FUZZY_ARTMAP
};

/**
 * The ART network by Grossberg et al. has by default two layers, called F1 and F2. The input pattern
 * has a fixed number of inputs "m". There is a vigilance parameter which specifies the resemblance
 * needed between F1 and F2 to be a match. The field F2 can be seen as represented the corresponding
 * class, prototype, or codebook vector. In ART both fields F1 and F2 are considered short-term
 * memory. The long-term memory are the connections between F1 and F2 in the form of the weights. The
 * weights are adjusted automatically when classifyInput() is called.
 */
class Art
{
public:
	/**
	 * Create a default ART network. The "matchTrack" parameter defines if the ART network
	 * will be used only to "match and track" or as a "supervisor" network. So, if "matchTrack"
	 * is true this means either stand-alone use, or "slave" network in a supervisor mode.
	 */
	Art(bool matchTrack, bool useInputComplement = true, bool useWTA = true);

	/**
	 * The only function you will need to use to actually work with ART. All the other functions
	 * are setters and getters for parameters.
	 */
	ART_DISTRIBUTED_CLASS* classifyInput(ART_ASPECT &input);

	void addToVigilanceHistory(ART_TYPE vig);
	void setVigilanceHistorySize(int size);
	void saveArtNetwork(std::string fileName);
	void loadArtNetWork(std::string fileName);

	ART_TYPE getAVGVigilance();
	inline int getCompressionCount() const { return d_compressionCount;}

	inline int getVigilanceHistorySize() const { return d_vigilanceHistorySize; }

	inline bool getMatchTrack() const { return d_matchTrack; }
	inline void setMatchTrack(bool d_matchTrack) { this->d_matchTrack = d_matchTrack; }

	//! Return all incoming weights of F2 node
	inline PROTOTYPE* getPrototype(int id){ return d_F2[id];}

	inline float getAlpha() const 						{ return d_alpha; }

	//! Alpha defines activity per node, the larger alpha, the less active the node
	//! By having a different alpha for an ART network it is possible to scale overall activity
	inline void setAlpha(float d_alpha) 				{ this->d_alpha = d_alpha;  }

	inline float getTrackingValue() const 				{ return d_trackingValue; }
	inline void setTrackingValue(float d_trackingValue) { this->d_trackingValue = d_trackingValue; }
	inline float getLearningFraction() const 			{ return d_learningFraction; }

	//! The learning fraction at "1" means fast learning, this is used to update the weights
	//! with d_learning_fraction set to "1" the weights are basically set to AND with the input pattern
	inline void setLearningFraction(float d_learningFraction) { this->d_learningFraction = d_learningFraction; }
	inline float getVigilance() const 					{ return d_vigilance; }
	inline void setVigilance(float vigilance)			{ d_vigilance = vigilance; }

	//! Return all weights of all prototypes (you can see this as the actual network)
	inline std::vector<PROTOTYPE*>* getF2()  			{ return &d_F2; }
	inline float getNetworkReliability() const 			{ return d_networkReliability; }
	inline void setNetworkReliability(float d_networkReliability)
	{ this->d_networkReliability = d_networkReliability; }
	inline void setTestMatch(bool test){d_testMatch = test;}
	inline bool getTestMatch(){return d_testMatch;}

	//! Match the input vector
	std::vector<ART_TYPE>* matchTrack(bool finished = false, bool raiseVigilance = false);

protected:
	//! Actually update the weights
	void updateWeights();
private:
	/**
	 * The value "T" or actually T_j (per node) indicates the similarity between the input vector I and
	 * the weight vector of the j'th F2 memory node. A "PROTOTYPE" is a set of weights from all input
	 * nodes on F1 to a node on F2.
	 */
	struct PROTOTYPE_Activation
	{
		// The index in F2
		int id;
		// The T_j value (the activity of the node)
		// Used for adjusting the weights subsequently if needed
		float T;
		// The resonance value (the actual non mismatch from that node)
		// The most active node can still be very much off (e.g. it is the only one)
		float resonance;
	};

	/**
	 * Function to sort the prototypes based on their "T" activity levels (depends on the last input).
	 * As you can see the value with the highest "T" value "wins" (in the priority queue will be at
	 * the front). With equal "T" values the highest index always wins.
	 */
	struct ComparePrototype {
		bool operator() (const PROTOTYPE_Activation* pt1, const PROTOTYPE_Activation* pt2) const
		{
			if(pt1->T == pt2->T)
				return pt1->id < pt2->id;

			return pt1->T < pt2->T;
		}
	};

	float	d_vigilance;

	//! The "signal rule parameter" denotes how quickly a weight vector of an F2 node shifts towards
	//! an input pattern
	float	d_alpha;
	float 	d_inputSize;
	float 	d_trackingValue;
	float	d_learningFraction;
	float	d_networkReliability;
	int 	d_vigilanceHistorySize;
	int 	d_currVHist;
	int 	d_compressionCount;

	//! Short-term memory input pattern
	std::vector<ART_TYPE>	d_F1;
	//! Long-term memory (which is not a series of nodes, but the weights to each high-level nodes)
	std::vector<PROTOTYPE*>	d_F2;

	std::vector<ART_TYPE>	d_vigilanceHist;
	//! A queue with the prototypes ordered on activity ("T" value)
	std::priority_queue<PROTOTYPE_Activation*, std::vector<PROTOTYPE_Activation*>, ComparePrototype> d_curPTAct;
	bool d_matchTrack, d_useInputComplement, d_useWTA, d_testMatch;
	ART_COMPUTATION_TYPE d_ACT;

	//! Creates values for F1 (and uses two-complementary representation if needed)
	void createF1(std::vector<ART_TYPE> &input);

	//! Calculates activity and resonance values for each prototype in F2
	void signalToProtoType();
};
}

#endif /* ART_H_ */
