/*
 * artMap.h
 *
 *  Created on: Sep 3, 2009
 *      Author: ted
 *
 * This is the interface of the multi-directional (un-)supervised ARTMAP
 * This is a modifdng but also
 * one-to-many, many-to-one, and many-to-many mappings to multiple ART categories are possible.
 *
 * The ARTMAP is standard un-supervised, by using a sort of Hebbian learning the weights of all the
 * ART categories to a single map are learned. All the single ART networks are suppose to present
 * the right output wrong output categories are matched but by using a slow learning rate, noise
 * doesn't effect the system to much. When one ART network output is missing it will be predicted
 * by the system.
 *
 * When an ART network has the status supervisor, this network can initiate match-tracking, for
 * the classes that gave the wrong output by using the map vigilance.
 *
 * This version only handles WTA for the moment
 *
 */

#ifndef ARTMAP_H_
#define ARTMAP_H_

#include "art.h"
#include <vector>
#include <utility>
#include <iostream>
#include <sstream>
#include <iomanip>


namespace almendeSensorFusion
{

/**************************************************************************************************************
 * Type definitions that make it easier to understand the code
 *
 * They are not always necessary and sometimes different typedefs are used for the same data structure to
 * tell what something really does
 *************************************************************************************************************/

//! A "map field" is used to "synchronize" between two or more ART networks (see artMap class explanation)
typedef std::vector< ART_TYPE> ART_MAPFIELD;

//! The activity of the "map field" calculated for all ART networks
typedef std::vector<ART_MAPFIELD*> ART_MAPFIELDS;

//! Normal ART_MAPFIELD is about activity, this is also about how many ART networks "vote" for it
typedef std::pair<int,ART_TYPE> ART_MAPFIELD_NODE_POPULARITY;

//! Stores the popularity of each node (in a vector)
typedef std::vector<ART_MAPFIELD_NODE_POPULARITY*> ART_MAPFIELD_POPULARITY;

//! One weight from a given F2 node to a map field node
typedef std::pair<int, ART_TYPE> F2_NODE_TO_MAPFIELD_NODE;

//! Weights from all F2 nodes (from one ART network) to one map field node
//! A "map field node" is also called often a class, so F2_TO_CLASS would be fine too
typedef std::vector<F2_NODE_TO_MAPFIELD_NODE*> F2_TO_MAPFIELD_NODE;

//! Weights from all F2 nodes (from one ART network) to all map field nodes
//! These are considered "classes"
//! The size of F2_TO_MAPFIELD should be the number of neurons in long-term memory (F2)
typedef std::vector<F2_TO_MAPFIELD_NODE*> F2_TO_MAPFIELD;

//! One weight from map field node to an F2 node
typedef std::pair<int, ART_TYPE> MAPFIELD_NODE_TO_F2_NODE;
typedef std::vector<MAPFIELD_NODE_TO_F2_NODE*> MAPFIELD_TO_F2_NODE;
typedef std::vector<MAPFIELD_TO_F2_NODE*> MAPFIELD_TO_F2;

//! For referencing we use integers, we might use char's for size later.
typedef int ART_INDEX;

//! The set of ART networks referenced by index
typedef std::vector< ART_INDEX> ART_NETWORK_INDICES;

//! The set of map field nodes referenced by index
typedef std::vector< ART_INDEX> ART_MAPFIELD_INDICES;

/**
 * The most normal ARTMAP exists out of two ART networks that are coupled to each other by a
 * so-called "map field". To one of the ART networks, say ART_a, is an input vector fed, which needs
 * to be classified. We happen to have another ART network which we already trained before, ART_b,
 * and now we want to use that network to supervise ART_a. This means we have to synchronize the
 * long-term memory nodes (in F2) of each network. This synchronization is done via map field F_ab.
 * A first thought would synchronize in such way that in ART_a a specific node F2_a[i] synchronizes
 * with the same specific node in ART_b, F2_b[i]. However, we do not expect in all cases to have
 * something represented by just one node in F2, a prototype can be a distributed set of activations.
 * Moreover, we also do not care about having the same representation of this prototype in ART_a
 * versus ART_b. What we only care about that if these prototypes synchronize! That is why there is
 * this "complicated" map field concept. The synchronization happens when a node in the map field
 * is activated by both ART networks.
 */
class ArtMap
{

public:
	/**
	 * We need a set of ART networks.
	 */
	ArtMap(std::vector<Art*>* networks, float learnFraction = 0.5);

	/**
	 * The classification routine. Of course the input vectors need to be ordered corresponding to
	 * the existing ART networks. Either given at construction, or later on by addArtNetwork.
	 */
	ART_DISTRIBUTED_CLASSES* classify(ART_VIEW& multipleInputVectors);

	/**
	 * Return the distributed activation of the "map field" in between two (or more) ART networks.
	 * Consider that the input is multiple "ART_VIEWs". So, e.g. in case multiple views from
	 * different locations are taken of the same object. The result has to be communicated as a
	 * vector of winners over the different views.
	 */
	ART_DISTRIBUTED_CLASS* distMapNodeClassification(ART_VIEWS* multipleInputVectors,
			int* foundCount, std::vector<int>* winnerCount, F2_TO_MAPFIELD* distOutput = NULL);

	void saveArtMap(std::string fileName);
	void loadArtMap(std::string fileName);
	void printArtMap();


	inline int getNrMapNodes() const { return d_nrMapNodes; }
	inline bool getUseVigilance() const { return d_useVigilance; }
	inline void setUseVigilance(bool d_useVigilance) { this->d_useVigilance = d_useVigilance; }

	inline float getLearningFraction() const { return d_learningFraction; }
	inline void setLearningFraction(float d_learningFraction) { this->d_learningFraction = d_learningFraction; }
	inline float getVigilance() const { return d_vigilance; }
	inline void setVigilance(float d_vigilance) { this->d_vigilance = d_vigilance; }
	inline void addArtNetwork(Art* artNetwork) { d_artNetworks->push_back(artNetwork); }
	inline Art* getArtNetwork(int networkNr) { return (*d_artNetworks)[networkNr] ;}

protected:
	//! Winner-take-all of all field map nodes
	int getMapNodeWTA(ART_INDEX artNetworkNr, ART_INDEX classId);
	//! Winner-take-all of all F2 nodes
	int getArtClassWTA(int artNetworkNr, int mapNode);
private:
	float 		d_learningFraction;
	float		d_vigilance;
	int			d_nrMapNodes;
	int			d_nrOfInputClasses;
	bool		d_useVigilance;

	std::vector<Art*>*	d_artNetworks;

	//! From all ART networks to "map field" structure
	//! The size of d_artF2 is equal to the number of ART networks
	std::vector<F2_TO_MAPFIELD*> d_artF2;

	//! From "map field" to all ART networks
	//! The size of d_mapNodes is equal to the number of map field nodes
	std::vector<MAPFIELD_TO_F2*> d_mapNodes;

	//! Calculate map field activity for each ART network
	ART_MAPFIELDS* calcMapNodeActivation(ART_VIEW* inputVectors);

	//! Create a new map field node
	void createNewMapNode(ART_VIEW* inputVectors);

	void addToMapNode(int mapNodeNr, ART_ASPECT* inputVectors);

	//! Get the ART networks
	ART_NETWORK_INDICES* getSupervisors(ART_VIEW* inputVectors);

	void updateConnections(int winningMapNode, ART_VIEW* inputVectors);

	//! The "popularity" of each map field node
	ART_MAPFIELD_POPULARITY* calcWinningNode(ART_MAPFIELDS* mnActList, ART_ASPECT* artN_new_nodes,
			ART_MAPFIELD_INDICES* input_map_nodes, ART_NETWORK_INDICES* nrSv, int *maxNodeNr, ART_TYPE *maxNodeCount);

	bool mapClasses(ART_VIEW* inputVectors);
};
}

#endif /* ARTMAP_H_ */
