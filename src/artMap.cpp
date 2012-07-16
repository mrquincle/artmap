/*
 * artMap.cpp
 *
 *  Created on: Sep 3, 2009
 *      Author: ted
 */

#include "artMap.h"
#define DEBUG_INFO 1
using namespace std;

namespace almendeSensorFusion
{

/**
 * An ARTMAP basically exists out of ART networks and an association field in between the
 * networks.
 */
ArtMap::ArtMap(std::vector<Art*>* networks, float learnFraction): d_artF2(0),
		d_mapNodes(0)
{
	d_learningFraction 	= learnFraction;
	d_artNetworks		= networks;
	d_nrMapNodes		= 0;
	d_nrOfInputClasses  = 0;
	//d_useVigilance		= false;
}

/**
 * This methods classifies input from different vectors (for example multiple views or a sequence
 * of features over time).
 * it gives as output the art classes to which the input vectors belong.
 * The classes for the input vector that are NULL are searched for and also returned
 * @return 		For each input, a class will be returned
 */
ART_DISTRIBUTED_CLASSES* ArtMap::classify(ART_VIEW& multipleInputVectors)
{
	ART_DISTRIBUTED_CLASSES* artClasses = new ART_DISTRIBUTED_CLASSES(0);
	// Maybe have a different vigilance for match track and when no supervision is available
	ART_NETWORK_INDICES* nrOfSuperv = getSupervisors(&multipleInputVectors);
	d_nrOfInputClasses = 0;
	for (int x = 0; x < multipleInputVectors.size(); ++x)
		if(multipleInputVectors[x] != NULL)
			++d_nrOfInputClasses;

	for (int artNr = 0; artNr < multipleInputVectors.size(); ++artNr)
	{
		if(multipleInputVectors[artNr] != NULL)
		{
			bool mt = (*d_artNetworks)[artNr]->getMatchTrack();
			// When there is no supervisor, do not match track
			// but when there is no other input to associate with
			// take the best match -> match tracking
			// unless there the use of vigilance is forced
			if(nrOfSuperv->size() == 0 && (d_nrOfInputClasses > 1 || d_useVigilance))
			{
				(*d_artNetworks)[artNr]->setMatchTrack(false);
			}
			std::vector<ART_TYPE>* out = (*d_artNetworks)[artNr]->classifyInput(*(multipleInputVectors[artNr]));
			artClasses->push_back(out);
			(*d_artNetworks)[artNr]->setMatchTrack(mt);
			//cout << "Art: " << artNr << " winner:" << ((*artClasses)[artNr])->at(0) << endl;
		}
		else
		{
			artClasses->push_back(NULL);
		}
	}
	// The missing classes are given in the ArtClasses
	bool result = mapClasses(artClasses);
	for (int artNr = 0; artNr < multipleInputVectors.size(); ++artNr)
	{
		if(multipleInputVectors[artNr] != NULL)
		{
			(*d_artNetworks)[artNr]->matchTrack(true);
		}
	}
	delete nrOfSuperv;
	return artClasses;
}

/**
 * This method classifies input and returns the distributed map node activation. The inputs are meant
 * for all the involved ART networks and can even be about multiple views (over time or distance e.g.).
 * The
 */
ART_DISTRIBUTED_CLASS* ArtMap::distMapNodeClassification(ART_VIEWS* multipleInputVectors,
		int* foundCount, std::vector<int>* winnerCount, F2_TO_MAPFIELD* distOutput)
{
	vector<pair<int,ART_TYPE>*>* total_node_values = new vector<pair<int,ART_TYPE>*>(0);
	*foundCount = 0;

	// iterate over all views
	for (int inputNr = 0; inputNr < multipleInputVectors->size(); ++inputNr) {

		// shortcut for this view
		ART_VIEW* inputVectors = (*multipleInputVectors)[inputNr];

		// this is NOT the artClasses vector which will be returned
		ART_DISTRIBUTED_CLASSES* artClasses = new ART_DISTRIBUTED_CLASSES(0);
		d_nrOfInputClasses = 0;
		for (int x = 0; x < inputVectors->size(); ++x)
			if(((*inputVectors)[x]) != NULL)
				++d_nrOfInputClasses;

		// classify "aspect" with the corresponding ART network
		for (int artNr = 0; artNr < inputVectors->size(); ++artNr)
			if((*inputVectors)[artNr] != NULL)
				artClasses->push_back((*d_artNetworks)[artNr]->classifyInput(*((*inputVectors)[artNr])));
			else
				artClasses->push_back(NULL);

		// calculate map field
		ART_MAPFIELDS* result = calcMapNodeActivation(artClasses);

		// Temp values //
		ART_NETWORK_INDICES* nrSv = getSupervisors(inputVectors);
		vector<ART_TYPE> artN_new_nodes(0);
		ART_MAPFIELD_INDICES input_map_nodes(inputVectors->size());

		int countExistingNodes 	= 0;
		int maxNodeNr 			= -1;					// winning node
		ART_TYPE maxNodeCount 	= 0;					// nr of times a winner
		// Temp values //

		// calculate winning Node and activations
		vector<pair<int,ART_TYPE>*>* node_values = calcWinningNode(result, &artN_new_nodes, &input_map_nodes,
				nrSv, &maxNodeNr, &maxNodeCount);

		// finalize the matchtrack process (means: update the weights)
		for (int artNr = 0; artNr < inputVectors->size(); ++artNr)
			if((*inputVectors)[artNr] != NULL)
				(*d_artNetworks)[artNr]->matchTrack(true);

		//Cleanup
		for (int artNr = 0; artNr < artClasses->size(); ++artNr)
			if((*artClasses)[artNr] != NULL)
				delete (*artClasses)[artNr];
		delete artClasses;
		delete nrSv;

		//Sum the node values
		if(node_values != NULL)
		{
			bool nodeFound = false;
			for (int nodeNr = 0; nodeNr < node_values->size(); ++nodeNr)
			{
				while(total_node_values->size() < node_values->size())
					total_node_values->push_back(new pair<int, ART_TYPE>(0,0.0));
				(*total_node_values)[nodeNr]->first += (*node_values)[nodeNr]->first;
				(*total_node_values)[nodeNr]->second += (*node_values)[nodeNr]->second;
				if((*total_node_values)[nodeNr]->first > 0)
					nodeFound = true;
				delete (*node_values)[nodeNr];
			}
			delete node_values;
			if(nodeFound) ++(*foundCount);
		}
	}

	// Debugging info
#if DEBUG_INFO > 0
//	cout << "MapField Activation:" << endl;
	for (int nodeNr = 0; nodeNr < total_node_values->size(); ++nodeNr)
	{
		cout << "	Mapfield ID:" << nodeNr << " Connections:" << (*total_node_values)[nodeNr]->first << " Total weight:" << (*total_node_values)[nodeNr]->second << endl;
	}
#endif

	// First search if there is a node
	for (int nodeNr = 0; nodeNr < total_node_values->size(); ++nodeNr){ 	}

	vector<ART_TYPE>* artClasses = NULL;

	// For all ART networks that where empty find the associated classes
	if(multipleInputVectors != NULL && multipleInputVectors->size() > 0)
	{
		artClasses = new vector<ART_TYPE>(0);
		vector<vector<ART_TYPE>*>* inputVectors = (*multipleInputVectors)[0];
		for (int artNr = 0; artNr < inputVectors->size(); ++artNr)
		{
			int winningClass 	= -1;
			int winnCount		= 0;
			if((*inputVectors)[artNr] == NULL)
			{
				//std::cout << "find empty class art: " << artNr << std::endl;
				vector<pair<int,ART_TYPE>*>* artClassValues = new vector<pair<int,ART_TYPE>*>(0);
				for (int nodeNr = 0; nodeNr < total_node_values->size(); ++nodeNr)
				{
					if((*total_node_values)[nodeNr]->first > 0)
					{
						int artClass = getArtClassWTA(artNr, nodeNr);
						if(artClass != -1)
						{
							while(artClassValues->size() <= artClass)
								artClassValues->push_back(new pair<int, ART_TYPE>(0,0.0));

							// Edited!!
							// now using the mapnode active connection count instead of 1 time activation
							(*artClassValues)[artClass]->first += (*total_node_values)[nodeNr]->first;
							(*artClassValues)[artClass]->second += (*total_node_values)[nodeNr]->second;
							// Edited!!
							// the winning class is the one connected to the mapfield with the heights number of connection
							// Only this way we can overcome the plasticity stability dilemma
							if((*total_node_values)[nodeNr]->first > winnCount)
							{
								winnCount = (*total_node_values)[nodeNr]->first;
								winningClass = artClass;
								//cout << "winnclass " << winningClass << " winncount " << winnCount << endl;
							}
						}
					}
				}
				// Winning class found for this ART network based on count,
				// check for multiple winners and use activation
				if(winningClass != -1)
				{
					// Check if there is another mapfield with the same number of connections
					// if so than if the associated class has a higher amount of active connections, then it will be chosen
					ART_TYPE winnValue = (*artClassValues)[winningClass]->first;
					for (int nodeNr = 0; nodeNr < total_node_values->size(); ++nodeNr)
					{
						int artClass = getArtClassWTA(artNr, nodeNr);
						if(artClass != -1)
						{
							if((*total_node_values)[nodeNr]->first == winnCount && winnValue < (*artClassValues)[artClass]->second)
							{
								winnValue 		= (*artClassValues)[artClass]->second;
								winningClass	 = artClass;
							}
						}
					}
				}
				if(distOutput == NULL)
					for (int artClass = 0; artClass < (*artClassValues).size(); ++artClass)
					{
						delete (*artClassValues)[artClass];
					}

				if(distOutput == NULL)
					delete artClassValues;
				else
					distOutput->push_back(artClassValues);
				// Store winners and occurrences
				winnerCount->push_back(winnCount);
				artClasses->push_back(winningClass);
			}
			else
			{
				artClasses->push_back(-1);
				winnerCount->push_back(0);
			}
		}
	}
	// clean up stuff
	for (int nodeNr = 0; nodeNr < total_node_values->size(); ++nodeNr)
		delete (*total_node_values)[nodeNr];
	delete total_node_values;
	return artClasses;
}

/**
 * Calculate the winning "node". This node is one in the map field. So, not a node in one of the
 * F2 layers of one of the ART networks.
 * @param mnActList			in: a pointer to the set of activity values for the map field sorted on ART network
 * @param artN_new_nodes
 * @param input_map_nodes	out: the winning map field node per ART network, -1 if there was no node winning
 * @param nrSv				in: indices to the ART networks that are supervising
 * @param maxNodeNr			out: the index to the map field node with maximum overall activity
 * @param maxNodeCount		out: the number of ART networks that do activate this map field node
 * @return					out: "popularity" of each map field node
 */
ART_MAPFIELD_POPULARITY* ArtMap::calcWinningNode(ART_MAPFIELDS* mnActList,
		vector<ART_TYPE>* artN_new_nodes, ART_MAPFIELD_INDICES* input_map_nodes, ART_NETWORK_INDICES* nrSv,
		int *maxNodeNr , ART_TYPE *maxNodeCount)
{
	/****************WARNING SUPERBAD CODE***************/
	if(mnActList == NULL)
		return NULL;

	*maxNodeNr  = -1;

	// the "int" in this pair defines the number of ART networks that activate this node positively
	// the "ART_TYPE" value in this pair denotes the total activation of this node
	// then in the end we store the popularity for each map field node
	ART_MAPFIELD_POPULARITY* node_valuePair = new ART_MAPFIELD_POPULARITY(0);

	for (int networkNr = 0; networkNr < mnActList->size(); ++networkNr)
	{
		// So the individual values in "nodes" will be the activity values of the map field nodes
		ART_MAPFIELD* nodes = (*mnActList)[networkNr];
		if(nodes != NULL)
		{
			//cout << "Calc winning node art:" << networkNr << " : node size: " << nodes->size();
			ART_TYPE winningNodeValue = 0;
			int 	winningNode = -2;

			// iterate over all map field nodes for this ART network
			for (int nodeNr = 0; nodeNr < nodes->size(); ++nodeNr)
			{
				// ugly code indeed, basically just adds a pair only at loop networkNr==0
				while(node_valuePair->size() <= nodeNr)
					(*node_valuePair).push_back(new ART_MAPFIELD_NODE_POPULARITY(0,0.0));

				// now retrieve the previously created pair
				ART_MAPFIELD_NODE_POPULARITY* node_pair = (*node_valuePair)[nodeNr];

				// the second field is increased with the activity value of the node
				// it will contain total aggregated activity for all ART networks for this node in the end
				node_pair->second += (*nodes)[nodeNr];

				// the first field counts the number of strictly positive activity values
				// it will contain a number w.r.t. how many ART network positively contributed to this node
				if((*nodes)[nodeNr] > 0) {
					node_pair->first += 1;
				}

				// update maximum node "count" number if node_pair is larger
				// stores node with the most ART networks referencing it
				if(*maxNodeCount < node_pair->first)
				{
					*maxNodeNr 		= nodeNr;
					*maxNodeCount 	= node_pair->first;
				}

				// update activity of (currently) winning node
				// stores node with highest activity caused by 1 of the ART networks
				if(winningNodeValue < (*nodes)[nodeNr])
				{
					winningNodeValue = (*nodes)[nodeNr];
					winningNode = nodeNr;
				}
			}

			// If a new value has entered but old connections already exist
			// then add this node to the existing node
			if(winningNode == -2)
			{
				bool superVisorNew = nrSv->size() > 0;
				for (int spv = 0; spv < nrSv->size(); ++spv)
				{
					superVisorNew *= ((*nrSv)[spv] == networkNr);
				}
				if(!superVisorNew)
					artN_new_nodes->push_back(networkNr);
			}

			// here we set the winning node per ART network
			(*input_map_nodes)[networkNr] = winningNode;
			delete nodes;
		}
		else
		{
			(*input_map_nodes)[networkNr] = -1;
		}
	}
	// The node with the most occurrence is found, now check if others don't
	// have the same occurrence and if so use activation to choose the winner
	if(*maxNodeNr != -1)
	{
		// so this only is useful if there is another node with just as many ART networks voting for it
		// only then the activity of the node is considered
		pair<int, ART_TYPE>* maxNode_pair = (*node_valuePair)[*maxNodeNr];
		ART_TYPE maxActivation = maxNode_pair->second;
		for (int nodeNr = 0; nodeNr < node_valuePair->size(); ++nodeNr)
		{
			pair<int, ART_TYPE>* node_pair = (*node_valuePair)[nodeNr];
			if(*maxNodeCount == node_pair->first && maxActivation < node_pair->second)
			{
				*maxNodeNr = nodeNr;
				// Have to check this!!!
				//std::cout << "a node was found with the same occurances" << std::endl;
			}
		}
	}
	return node_valuePair;
}

/* This method associates input by assigning it to a known or newly created cluster
 * The input vector has as many ART category vectors as ART networks.
 * Every category vector contains a list of category activations.
 * With WTA ART networks the list contains only one class.
 * The order of the ART network vector is maintained for the input vector
 *
 * If an ART category input vector is NULL then the associated category is looked for,
 * in the ART corresponding ART network, using the other input vectors.
 * When only one ART category vector is given as input, then only if a cluster is found,
 * the missing parts will be returned.
 *
 * By using the activation of the category inputs, together with their weight strength,
 * a Map_Node is activated. The weight strength is updated using Hebbian learning.
 *
 * A Map_node is created when:
 * - an ART network has a new class
 * 	 if there are more then 2 ART Networks then
 * 		if there already is an association between the other networks without a connection
 * 		to this network, then this map_node had never associated a value from this art network
 * - there is inconsistent output
 *
 * The value of an Map_Node is measured as follows (WTA):
 *
 * - first calculate a list of all the Map_nodes to which there is a connection for every class
 * - if a map node is present in all the lists then
 * 		that node is the winner
 * 		TODO: when there are multiple winners then choose the one
 * 			with the highest activation (for #ART networks > 2)
 * - else
 * 		// inconsistent map_node output
 *      if there are multiple supervisors then
 *      	if all supervisors have the same class as output then
 *      		use that class as desirable output for match tracking
 *      	else
 *      		create a new Map_node
 *      else if there is one supervisor
 *      	a search is initiated by the supervisor for all inconsistent networks
 *      	if the correct output is found then
 *      		update the weights
 *     	 	else
 *      		a new class is created by an ART network, create new Map_node
 *      else
 *      	create a new Map_node
 *
 * - exception if a new supervisor enters, matchtrack others until new node
 *
 * A supervisor has to have values that are unique for a class, when two supervisors
 * are used, a "composite (more unique) key" is created
 *
 * if one modality is missing and a new map_node is created then no connection to the missing
 * modality will be created
 *
 * TODO: implement increased gradient CAM Rule
 * To be done: Chop up this dragon in normal pieces
 *
 */
bool ArtMap::mapClasses(vector<vector<ART_TYPE>*>* inputVectors)
{
	//cout << "Calc act" << endl;
	vector<vector<ART_TYPE>*>* mnActList = calcMapNodeActivation(inputVectors);
	if(mnActList == NULL)
		return false;
	int nrOfMapsNodes = d_nrMapNodes;
	ART_NETWORK_INDICES* nrSv = getSupervisors(inputVectors);
	vector<ART_TYPE> artN_new_nodes(0);
	ART_MAPFIELD_INDICES input_map_nodes(inputVectors->size());

	int maxNodeNr 			= -1;					// winning node
	ART_TYPE maxNodeCount 	= 0;					// nr of times a winner

	// calculate winning Node and activations
	vector<pair<int,ART_TYPE>*>* node_values = calcWinningNode(mnActList, &artN_new_nodes, &input_map_nodes, nrSv,  &maxNodeNr, &maxNodeCount);
	for (int x = 0; x < node_values->size(); ++x)
		delete (*node_values)[x];
	delete node_values;
	delete mnActList;

	//cout << "Winning node:" << maxNodeNr << endl;
	//cout << "Nodes win time: " << maxNodeCount<< endl;
	//cout << "New nodes:" << artN_new_nodes.size() << endl;

	// All classes have the same map node
	if(maxNodeCount == d_nrOfInputClasses)
	{
		//cout << "Update nodes Max node Count" << endl;
		if(d_nrOfInputClasses > 1)
			updateConnections(maxNodeNr, inputVectors);
		// else
		// don't update only give missing values
	}
	// All inputs don't have map nodes
	else if(artN_new_nodes.size() == d_nrOfInputClasses && d_nrOfInputClasses > 1)
	{
		createNewMapNode(inputVectors);
	}
	// there are new nodes, and the rest have the same map node
	else if(artN_new_nodes.size() == (d_nrOfInputClasses - maxNodeCount) && d_nrOfInputClasses > 1)
	{
		// check if the winning map node has this ART network else connect it
		bool newInfo = true;
		for (int netNr = 0; netNr < artN_new_nodes.size(); ++netNr)
			newInfo *= ((*(d_mapNodes[maxNodeNr])).size() <= artN_new_nodes[netNr]) || ((*(d_mapNodes[maxNodeNr]))[artN_new_nodes[netNr]]->size()==0);

		// No Art network available
		// connect and update weights
		if(newInfo)
		{
			//cout << "Update nodes there are new nodes but also existing shizle" << endl;
			vector<ART_TYPE> newClasses = vector<ART_TYPE>(inputVectors->size());
			for (int inputNr = 0; inputNr < newClasses.size(); ++inputNr)
				newClasses[inputNr] = -1;

			for (int inputNr = 0; inputNr < artN_new_nodes.size(); ++inputNr)
				newClasses[artN_new_nodes[inputNr]] = (*(*inputVectors)[artN_new_nodes[inputNr]])[0];

			addToMapNode(maxNodeNr, &newClasses);
			updateConnections(maxNodeNr, inputVectors);
		}
		// create new map node can't match track when new node is created
		else
			createNewMapNode(inputVectors);
	}
	// the output is inconsistent match track or new map node
	else
	{
		// there is a new class with inconsistent map nodes
		// create new map node
		if(artN_new_nodes.size() > 0 && d_nrOfInputClasses > 1)
		{
			createNewMapNode(inputVectors);
		}
		else if(nrSv->size() > 0)
		{
			// check if all supervisors have the same map node as output
			bool consistentSupervisors = true;
			int spvWinningNode = 0;
			for (int superVNr = 0; superVNr < nrSv->size(); ++superVNr)
			{
				int artNNr = (*nrSv)[superVNr];
				if(superVNr == 0)
					spvWinningNode = input_map_nodes[artNNr];
				else
					consistentSupervisors *= (spvWinningNode == input_map_nodes[artNNr]) ;
			}
			maxNodeNr = spvWinningNode;
			// Match track (MT) others
			// NB. is a switch between vigilance possible?
			// does an un-supervised network with a fixed vigilance
			// still work when classes are created with a different vigilance?
			// Test and update!
			if(consistentSupervisors)
			{
				// if MT is initiated then:
				// 1. the right classes can be found
				// 2. a wrong class can be found
				// 3. a new class can be created
				//
				// for each ART network with wrong output: Case:
				// 1. communicate true back to MT
				// 2. communicate false back to MT and run MT again
				// 3. communicate true back to MT mark new class
				//
				// TODO add new nodes if the map node did not encountered
				// the art network before
				// if 1 or more new classes are found:
				// 		add new nodes to existing classes
				// else
				// 		update weights
				vector<vector<ART_TYPE>*>* newInputVectors = new vector<vector<ART_TYPE>*>(0);
				vector<ART_TYPE> addtoMapNodeList(0);

				for (int nodeNR = 0; nodeNR < input_map_nodes.size(); ++nodeNR)
				{
					//cout << "for every input map node" << endl;
					if(input_map_nodes[nodeNR] != -1)
					{
						//	cout << "input map node has:" << input_map_nodes[nodeNR] << endl;
						int map_node = input_map_nodes[nodeNR];
						int classId	= (*(*inputVectors)[nodeNR])[0]; //WTA
						while(map_node != spvWinningNode)
						{
							// Match track
							//cout << "Match Track" << endl;
							vector<ART_TYPE>* artOut = (*d_artNetworks)[nodeNR]->matchTrack(false, true);
							// New class found
							classId = (*artOut)[0];
							delete artOut;
							// Find WTA map node for this class
							map_node = getMapNodeWTA(nodeNR, classId);
							// If class is new, add to map node
							if(map_node == -1)
							{
								F2_TO_MAPFIELD* F2 = d_artF2[nodeNR];
								// If the class index is not known in the F2 network then create it
								while(F2->size() <= classId)
									F2->push_back(new F2_TO_MAPFIELD_NODE(0));
								break;
							}
						}
						addtoMapNodeList.push_back(classId);
						vector<ART_TYPE>* artValues = new vector<ART_TYPE>(0);
						artValues->push_back(classId);
						newInputVectors->push_back(artValues);
					}
					else
					{
						//	cout << "input mapnode is -1" << endl;
						addtoMapNodeList.push_back(-1);
						newInputVectors->push_back(NULL);
					}
				}
				//	cout << "Update nodes consistent supervisor" << endl;
				if(spvWinningNode != -2)
				{
					addToMapNode(spvWinningNode, &addtoMapNodeList);
					updateConnections(spvWinningNode, newInputVectors);
				}
				else
					// new supervisor node
				{
					createNewMapNode(newInputVectors);
					//	cout << "Done" << endl;
				}
				//delete newInputVector
				for (int x = 0; x < newInputVectors->size(); ++x)
				{
					if ((*newInputVectors)[x] != NULL)
						delete (*newInputVectors)[x];
				}
				delete newInputVectors;
			}
			// create new
			else
			{
				createNewMapNode(inputVectors);
			}
		}
		// Create new node
		else if (d_nrOfInputClasses > 1)
		{
			createNewMapNode(inputVectors);
		}
	}
	delete nrSv;
	// find missing values
	if(d_nrOfInputClasses < d_artNetworks->size())
	{
		//cout << "Find missing" << endl;
		for (int artN = 0; artN < input_map_nodes.size(); ++artN)
		{
			//cout << "Art network to search for";
			// No value given: (*inputVectors)[artN] == NULL
			if(input_map_nodes[artN] == -1)
			{
				//cout << artN << endl;
				int fClassId = -1;
				if(nrOfMapsNodes == d_nrMapNodes)
				{
					//cout << "get ARTWTA class" << endl;
					fClassId = getArtClassWTA(artN, maxNodeNr);
					if(fClassId != -1)
					{
						vector<ART_TYPE>* artNOutput = new vector<ART_TYPE>(0);
						artNOutput->push_back(fClassId);
						(*inputVectors)[artN] = artNOutput;
					}
				}
				//else
				//{
				//	cout << "New node created, none found" << endl;
				//}
				// else a new class was created no output can be given
			}
		}
	}
	return true;
}

/**
 * Consider the indicated ART network and return the map node that has the highest activation
 * value.
 * @param artNetworkNr		in: the index of the ART network
 * @param classId			in: the index of the F2 node
 * @return					out: the index of a map field node
 */
int ArtMap::getMapNodeWTA(ART_INDEX artNetworkNr, ART_INDEX classId)
{
	if(d_artF2.size() <= artNetworkNr)
		return -1;

	F2_TO_MAPFIELD* F2 = d_artF2[artNetworkNr];
	if(F2->size() <= classId)
		return -1;

	// the f2 nodes are also called "classes"
	F2_TO_MAPFIELD_NODE* mnl = (*F2)[classId];
	int maxNodeNr 			= -1;		// winning map node
	ART_TYPE maxNodeValue 	= -1;		// max value

	// iterate over all map field nodes and get the one with largest activity/weight
	for (int i = 0; i < mnl->size(); ++i)
	{
		F2_NODE_TO_MAPFIELD_NODE* ml = (*mnl)[i];
		if(maxNodeValue <  ml->second)
		{
			maxNodeValue 	= ml->second;
			maxNodeNr		= ml->first;
		}
	}
	return maxNodeNr;
}

/**
 * See getMapNodeWTA, but this time the F2 node will be chosen in the network
 * to which this map node is connected with the highest weight
 */
int ArtMap::getArtClassWTA(int artNetworkNr, int mapNode)
{
	if(d_mapNodes.size() <= mapNode)
		return -1;

	MAPFIELD_TO_F2* artN = d_mapNodes[mapNode];
	if(artN->size() <= artNetworkNr)
		return -1;

	MAPFIELD_TO_F2_NODE* acl = (*artN)[artNetworkNr];
	int maxClassNr			= -1;			// winning class
	ART_TYPE maxClassValue 	= -1;			// nr of times a winner

	// iterate over all classes
	for (int i = 0; i < acl->size(); ++i)
	{
		MAPFIELD_NODE_TO_F2_NODE* ac = (*acl)[i];
		if(maxClassValue <  ac->second)
		{
			maxClassValue 	= ac->second;
			maxClassNr		= ac->first;
		}
	}
	return maxClassNr;
}

void ArtMap::updateConnections(int winningMapNode, vector<vector<ART_TYPE>*>* inputVectors)
{
	//cout << "Update connections" << endl;
	for (int x = 0; x < inputVectors->size(); ++x)
	{
		// For all the input vectors that have values
		if((*inputVectors)[x] != NULL)
		{
			// For F2 to Map_Node
			vector<ART_TYPE> *outputClasses = (*inputVectors)[x];
			int	classIndex = (*outputClasses)[0];
			F2_TO_MAPFIELD* F2 = d_artF2[x];
			F2_TO_MAPFIELD_NODE* mnl = (*F2)[classIndex];
			// Maybe better to have each map_node a position but
			// maybe slower when there are a lot of map nodes and only 2 per class
			for (int i = 0; i < mnl->size(); ++i)
			{
				F2_NODE_TO_MAPFIELD_NODE* ml = (*mnl)[i];
				if(ml->first == winningMapNode)
					ml->second += d_learningFraction;
			}

			// For Map_Node to F2
			MAPFIELD_TO_F2* artNs 	= d_mapNodes[winningMapNode];
			MAPFIELD_TO_F2_NODE* artCl 	= (*artNs)[x];
			for (int i = 0; i < artCl->size(); ++i)
			{
				MAPFIELD_NODE_TO_F2_NODE* artC 	= (*artCl)[i];
				if(artC->first == classIndex)
					artC->second += d_learningFraction;
			}
		}
	}
}

/**
 * Get all the supervising networks (recognizable by a network reliability of 1.0)
 */
ART_NETWORK_INDICES* ArtMap::getSupervisors(ART_VIEW* inputVectors)
{
	ART_NETWORK_INDICES* supers = new ART_NETWORK_INDICES(0);
	for (ART_INDEX x = 0; x < d_artNetworks->size(); ++x)
	{
		if((*inputVectors)[x] != NULL)
			if((*d_artNetworks)[x]->getNetworkReliability() == 1.0)
				supers->push_back(x);
	}
	return supers;
}

/**
 * Create an entirely new map field node. We use the terminology of "aspect" and "view". An aspect
 * is the basic input vector for an ART network and can reflect one feature set of one sensor from
 * a given object. A view is a set of aspects from for example multiple sensors. Hence, it is
 * possible to observe different aspects of an object, but where one of the sensors does not observe
 * anything.
 */
void ArtMap::createNewMapNode(ART_VIEW* inputVectors)
{
	// we iterate over all "views", per view we have a network
	for (int x = 0; x < inputVectors->size(); ++x)
	{
		// the view shouldn't be empty of course
		if((*inputVectors)[x] == NULL) continue;

		vector<ART_TYPE> *outputClasses = (*inputVectors)[x];
		int	classIndex = (*outputClasses)[0];

		// F2 to Map node
		F2_NODE_TO_MAPFIELD_NODE* mn = new F2_NODE_TO_MAPFIELD_NODE(d_nrMapNodes, d_learningFraction);
		F2_TO_MAPFIELD* F2 = d_artF2[x];
		F2_TO_MAPFIELD_NODE* mnl = (*F2)[classIndex];
		mnl->push_back(mn);

		// Map node to F2
		if(d_mapNodes.size() <= d_nrMapNodes)
			d_mapNodes.push_back(new MAPFIELD_TO_F2(0));

		MAPFIELD_TO_F2* artN = d_mapNodes[d_nrMapNodes];
		while(artN->size() <= x)
			artN->push_back(new MAPFIELD_TO_F2_NODE(0));

		MAPFIELD_NODE_TO_F2_NODE* artC = new MAPFIELD_NODE_TO_F2_NODE(classIndex, d_learningFraction);
		(*artN)[x]->push_back(artC);
	}
	// increment the counter
	++d_nrMapNodes;
}

/**
 * This function adds new incoming/outgoing weights to an existing "map field node" or "class".
 * All connections will be created, so: only WTA.
 * @param mapNodeNr		The map field node the weights need to be added at
 * @param inputVectors	A bundle of weights from all F2 to given map field node
 * @result				d_artF2 gets outgoing weights, d_mapNodes gets outgoing weights too
 */
void ArtMap::addToMapNode(int mapNodeNr, vector<ART_TYPE>* inputVectors)
{
	for (int x = 0; x < inputVectors->size(); ++x)
	{
		// For all the inputs that have values
		if((*inputVectors)[x] != -1)
		{
			int	classIndex = (*inputVectors)[x];

			// F2 to Map node
			F2_NODE_TO_MAPFIELD_NODE* mn = new F2_NODE_TO_MAPFIELD_NODE(mapNodeNr, 0);
			F2_TO_MAPFIELD* F2 = d_artF2[x];
			F2_TO_MAPFIELD_NODE* mnl = (*F2)[classIndex];
			mnl->push_back(mn);

			// Map node to F2
			MAPFIELD_TO_F2* artN = d_mapNodes[mapNodeNr];
			while(artN->size() <= x)
				artN->push_back(new MAPFIELD_TO_F2_NODE(0));

			MAPFIELD_NODE_TO_F2_NODE* artC = new MAPFIELD_NODE_TO_F2_NODE(classIndex, 0);
			(*artN)[x]->push_back(artC);
		}
	}
}

/**
 * This function calculates the activation of the map nodes (WTA) for every ART Network
 */
ART_MAPFIELDS* ArtMap::calcMapNodeActivation(ART_VIEW* inputVectors)
{
	vector<vector<ART_TYPE>*> *mapNodeActList = new vector<vector<ART_TYPE>*>(0);
	int nrOfInputVectors = 0;

	for (int x = 0; x < d_artNetworks->size(); ++x)
	{
		// For all the input vectors that have values
		if((*inputVectors)[x] != NULL)
		{
			++nrOfInputVectors;
			vector<ART_TYPE> *outputClasses = (*inputVectors)[x];
			if ((*outputClasses).size() > 1)
			{
				cout << "Distributed ART network not supported yet." << endl;
				return NULL;
			}
			int	classIndex = (*outputClasses)[0];

			// Create structure if this is the first time
			while (d_artF2.size() <= x)
			{
				F2_TO_MAPFIELD* cl = new F2_TO_MAPFIELD(0);
				d_artF2.push_back(cl);
			}

			F2_TO_MAPFIELD* F2 = d_artF2[x];
			// If the class index is not known in the F2 network then create it
			while(F2->size() <= classIndex)
				F2->push_back(new F2_TO_MAPFIELD_NODE(0));

			F2_TO_MAPFIELD_NODE* mnl = (*F2)[classIndex];

			// Calculate how many times a Map node is activated for this ART network x
			// map_node first has the number of the map_node, and second the weight value
			// because of WTA 1.0 times the weight is used
			mapNodeActList->push_back(new vector<ART_TYPE>(d_nrMapNodes));
			for (int mnNr = 0; mnNr < mnl->size(); ++mnNr)
			{
				F2_NODE_TO_MAPFIELD_NODE * mn = (*mnl)[mnNr];
				(*(*mapNodeActList)[x])[mn->first] = 1.0*(mn->second);
			}
		}
		else
			mapNodeActList->push_back(NULL);
	}

	if(nrOfInputVectors == 0)
		return NULL;
	return mapNodeActList;
}

void ArtMap::saveArtMap(std::string fileName)
{
	ofstream outputFile(fileName.c_str(), ios::out | ios::binary);
	if(!outputFile)
		printf( "Cannot open ARTMAP output file.\n");
	else
	{
		outputFile.write((char *) &d_learningFraction, sizeof(float));
		outputFile.write((char *) &d_vigilance, sizeof(float));
		outputFile.write((char *) &d_nrMapNodes, sizeof(int));
		outputFile.write((char *) &d_nrOfInputClasses, sizeof(int));
		outputFile.write((char *) &d_useVigilance, sizeof(bool));

		// Write d_artF2
		int f2Size = d_artF2.size();
		outputFile.write((char *) &f2Size, sizeof(int));
		for (int x = 0; x < d_artF2.size(); ++x)
		{
			F2_TO_MAPFIELD *classes = d_artF2[x];
			int classesSize = classes->size();
			outputFile.write((char *) &classesSize, sizeof(int));
			for (int y = 0; y < classes->size(); ++y)
			{
				MAPFIELD_TO_F2_NODE * artClassList = (*classes)[y];
				int artClassListSize = artClassList->size();

				outputFile.write((char *) &artClassListSize, sizeof(int));
				for (int z = 0; z < artClassList->size(); ++z)
				{
					MAPFIELD_NODE_TO_F2_NODE * artClass = (*artClassList)[z];
					outputFile.write((char *) &artClass->first, sizeof(int));
					outputFile.write((char *) &artClass->second, sizeof(ART_TYPE));
				}
			}
		}

		// Write d_mapNodes
		int size = d_mapNodes.size();
		outputFile.write((char *) &size, sizeof(int));
		for (int x = 0; x < d_mapNodes.size(); ++x)
		{
			MAPFIELD_TO_F2 * artNetworks = d_mapNodes[x];
			size = artNetworks->size();
			outputFile.write((char *) &size, sizeof(int));
			for (int y = 0; y < artNetworks->size(); ++y)
			{
				MAPFIELD_TO_F2_NODE * artClassList = (*artNetworks)[y];
				size =  artClassList->size();
				outputFile.write((char *) &size, sizeof(int));
				for (int z = 0; z < artClassList->size(); ++z)
				{
					MAPFIELD_NODE_TO_F2_NODE * artClass = (*artClassList)[z];
					outputFile.write((char *) &artClass->first, sizeof(int));
					outputFile.write((char *) &artClass->second, sizeof(ART_TYPE));
				}

			}
		}
		outputFile.close();
	}
}

void ArtMap::loadArtMap(std::string fileName)
{
	ifstream inputFile(fileName.c_str(),std::ios::in | std::ios::binary);

	// Load from file
	if(!inputFile.fail())
	{
		printf("Loading ARTMAP from file\n");
		inputFile.read((char *) &d_learningFraction, sizeof(float));
		inputFile.read((char *) &d_vigilance, sizeof(float));
		inputFile.read((char *) &d_nrMapNodes, sizeof(int));
		inputFile.read((char *) &d_nrOfInputClasses, sizeof(int));
		inputFile.read((char *) &d_useVigilance, sizeof(bool));

		// load d_artF2
		int f2Size = d_artF2.size();
		inputFile.read((char *) &f2Size, sizeof(int));
		for (int x = 0; x < f2Size; ++x)
		{
			F2_TO_MAPFIELD *classes = new F2_TO_MAPFIELD(0);
			d_artF2.push_back(classes);
			int classesSize = classes->size();
			inputFile.read((char *) &classesSize, sizeof(int));
			for (int y = 0; y < classesSize; ++y)
			{
				MAPFIELD_TO_F2_NODE * artClassList =  new MAPFIELD_TO_F2_NODE(0);
				classes->push_back(artClassList);
				int artClassListSize = artClassList->size();
				inputFile.read((char *) &artClassListSize, sizeof(int));
				for (int z = 0; z < artClassListSize; ++z)
				{
					int first = 0;
					ART_TYPE second = 0;
					inputFile.read((char *) &first, sizeof(int));
					inputFile.read((char *) &second, sizeof(ART_TYPE));
					MAPFIELD_NODE_TO_F2_NODE * artClass = new MAPFIELD_NODE_TO_F2_NODE(first, second);
					artClassList->push_back(artClass);
				}
			}
		}

		// load d_mapNodes
		int mapNodeSize = d_mapNodes.size();
		inputFile.read((char *) &mapNodeSize, sizeof(int));
		for (int x = 0; x < mapNodeSize; ++x)
		{
			MAPFIELD_TO_F2 * artNetworks =  new MAPFIELD_TO_F2(0);
			int artNetworkSize = artNetworks->size();
			d_mapNodes.push_back(artNetworks);
			inputFile.read((char *) &artNetworkSize, sizeof(int));
			for (int y = 0; y < artNetworkSize; ++y)
			{
				MAPFIELD_TO_F2_NODE * artClassList = new MAPFIELD_TO_F2_NODE(0);
				int artClassListSize =  artClassList->size();
				artNetworks->push_back(artClassList);
				inputFile.read((char *) &artClassListSize, sizeof(int));
				for (int z = 0; z < artClassListSize; ++z)
				{
					int first = 0;
					ART_TYPE second = 0;
					inputFile.read((char *) &first, sizeof(int));
					inputFile.read((char *) &second, sizeof(ART_TYPE));
					MAPFIELD_NODE_TO_F2_NODE * artClass = new MAPFIELD_NODE_TO_F2_NODE(first, second);
					artClassList->push_back(artClass);
				}
			}
		}
		inputFile.close();
	}
}

void ArtMap::printArtMap()
{
	// Loop through all the mapfield nodes and print the associate class
	int mapNodeSize = d_mapNodes.size();
	cout << "ARTMAP" << endl;
	int maxPatternSize = 20;
	for (int x = 0; x < mapNodeSize; ++x)
	{
		cout << "Mapfield: " << x << endl; 														// Mapfield nodeID
		vector<MAPFIELD_TO_F2_NODE*>* artNetworks = d_mapNodes[x];
		for (int ARTnetworkId = 0; ARTnetworkId < artNetworks->size(); ++ARTnetworkId)
		{
			vector<MAPFIELD_NODE_TO_F2_NODE*>* artClassList = (*artNetworks)[ARTnetworkId];
			cout << "	ART: " << ARTnetworkId << endl; 											// ART networkID
			for (int classID = 0; classID < artClassList->size(); ++classID)
			{
				int classId = ((MAPFIELD_NODE_TO_F2_NODE)*(*artClassList)[classID]).first; 			// ART classID
				ART_TYPE strength = ((MAPFIELD_NODE_TO_F2_NODE)*(*artClassList)[classID]).second;		// Connection strength
				stringstream value;

				vector<ART_TYPE>* pattern = ((Art*)(*d_artNetworks)[ARTnetworkId])->getF2()->at(classId);
				for (int patternPos = 0; patternPos < pattern->size()/2; ++patternPos)
				{
					if(patternPos != 0)
						value << ", ";
					value <<  setprecision(2) << (*pattern)[patternPos];
				}
				if(pattern->size() == 1)
					value << (*pattern)[0];

				//if(value.str().length() > maxPatternSize)
				//	maxPatternSize = value.str().length();

				cout << "		Class:" << classId << " Pattern:" << left << setw(maxPatternSize) << value.str() << " Weight: " << strength;
			}
			cout << endl;
		}
	}
}

/*
 * Clean-up Stuff
 */

}
