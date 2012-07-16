/*
 * art.cpp
 *
 * This class implements an ART network suitable for
 * the default ARTMAP
 *
 *  Created on: Sep 2, 2009
 *      Author: ted
 */

#include "art.h"

using namespace std;

namespace almendeSensorFusion
{

/**
 * This constructor initializes the network to be used for a single ART network or an ART network which
 * is part of an ARTMAP.
 *
 * If complement coding is used, then the input values need to be pre-scaled to [0,1]
 * TODO: fix for without complement coding
 */
Art::Art(bool matchTrack, bool useInputComplement, bool useWTA ): d_F1(0),
		d_F2(0),
		d_vigilanceHist(0)
{
	d_matchTrack 			= matchTrack;			// Match-tracking for the use in an ARTMAP
	d_useInputComplement 	= useInputComplement;	// Complement input coding or single input vector
	d_useWTA				= true;					// Distributed / Winner Take All output class
	d_vigilance  			= 0.65;					// Baseline vigilance for the default ARTMAP = 0
	d_ACT 					= DEFAULT_ARTMAP;		// Type of signal computation
	d_alpha					= 0.01;					// Signal rule parameter
	d_trackingValue			= -0.001;				// Match tracking value to raise vigilance with (-1,1) < 0 = MT-
	d_learningFraction		= 1;					// Learning fraction 1 = fast learning
	d_testMatch				= false;				// A best match is returned without updating the vigilance
	if(matchTrack)
		d_networkReliability= 0.9;					// cannot be a supervisor and matchTrack
	else
		d_networkReliability= 1.0;					// Reliability of the network [0,1] 1 = supervisor
	d_vigilanceHistorySize	= 0;					// The vigilance can be based on experience in supervised learning
	d_currVHist				= 0;					// value used for overwriting the correct history item
	d_compressionCount		= 0;
}

/**
 * This method implements the default ART network classification structure if match-tracking is not
 * used. If match tracking is enabled it implements the first run for the classification process
 * in an ARTMAP network. The output is the prototype or "distributed" class representation. This means
 * that it is represented as a vector, but not necessarily that not some kind of WTA mechanism did
 * happen. Only one of the values might be non-zero.
 */
ART_DISTRIBUTED_CLASS* Art::classifyInput(ART_ASPECT &input)
{
	createF1(input);
	d_inputSize = input.size();
	signalToProtoType();
	ART_DISTRIBUTED_CLASS* output = NULL;
	output = matchTrack();
	if(!d_matchTrack)
		matchTrack(true);
	return output;
}

/**
 * Basically just copies the input vector to F1. However, in the case of complement
 * encoding, F1 is made twice as large in this way:
 *  input: 0.9 0.2 0.3 0.4
 *  F1: 0.9 0.2 0.3 0.4 0.1 0.2 0.7 0.6
 * This means that the individual input value will not influence the overall weight
 * and is a manner of normalizing the input. See ART papers for more details.
 */
void Art::createF1(std::vector<ART_TYPE> &input)
{
	d_F1.clear();

	for (int x = 0; x < input.size(); ++x)
		d_F1.push_back(input[x]);

	if(d_useInputComplement)
		for (int x = 0; x < input.size(); ++x)
			d_F1.push_back(1-input[x]);
}

/* The Signals to the prototype (or committed coding node)
 * are calculated with the DefaultARTMAP type by:
 * Tj = |A n Wj| + (1-alpha)(M-|Wj|) for every prototype j
 *
 * and with the FUZZY_ART type:
 * Tj = |A n Wj|/alpha + |Wj|
 *
 * where A is the (complement coded) input vector
 * n the component-wise minimum (fuzzy intersection)
 * Wj the prototype feature vector
 * alpha the signal rule parameter (0,infinity) default 0.01
 * M the number of input features |A| = M
 * |.| the vector size (L1-norm)
 * if complement coding is used d_useInputComplement = true
 * then F1 and F2 have to be aligned when they differ in size
 * With F2 of size 2 the vigilance can be set higher then for larger F2
 * this has to be taken into consideration when calculating the resonance
 */
void Art::signalToProtoType()
{
	// Clear previous activations
	while (!d_curPTAct.empty())
	{
		delete d_curPTAct.top();
		d_curPTAct.pop();
	}

	// iterate over all high-level nodes in F2
	for (int x = 0; x < d_F2.size(); ++x)
	{
		// monkey out of the sleeve: a node d_F2[i] IS its weight vector
		std::vector<ART_TYPE>* Wj = d_F2[x];
		ART_TYPE Tj 	= 0;
		ART_TYPE diff	= 0;
		ART_TYPE sumWj 	= 0;

		// Align for different size with complement coding
		if(d_useInputComplement)
		{
			if(Wj->size() <= d_F1.size())
			{
				int sizeDiff = (d_F1.size() - Wj->size())/2;
				for (int i = 0; i < d_F1.size()-sizeDiff; ++i)
				{
					int indexF1 = i;
					if(i >= Wj->size()/2)
						indexF1 = (d_F1.size()/2) + (i-(Wj->size()/2));

					if(i < Wj->size())
						diff 	+= fabs(min(d_F1[indexF1],(*Wj)[i]));
					// Last half of complement is for the shortest always the highest
					else
						diff 	+= fabs(d_F1[indexF1]);
				}
			}
			else
			{
				int sizeDiff = (Wj->size() - d_F1.size())/2;
				for (int i = 0; i < Wj->size()-sizeDiff; ++i)
				{
					// The network can have different input sizes
					int indexF2 = i;
					if(i >= d_F1.size()/2)
						indexF2 = (Wj->size()/2) + (i-(d_F1.size()/2));

					if(i < d_F1.size())
						diff 	+= fabs(min(d_F1[i],(*Wj)[indexF2]));
					else
						diff 	+=  fabs((*Wj)[indexF2]);

					if(d_inputSize <= i)
						d_inputSize = i+1;
				}
			}
		}
		// without complement coding
		else
		{
			// if the network is too large for the inputs, weights will be neglected
			// if the input is larger than the network, inputs will be disregarded (but counted: diff is smaller)
			for (int i = 0; i < Wj->size(); ++i)
			{
				// The network can have different input sizes
				if(i < d_F1.size())
					diff 	+= fabs((d_F1[i]-(*Wj)[i]));
				else if(i > d_inputSize)
					d_inputSize = i; // only set/increase d_inputSize, diff becomes smaller!
			}
			diff = d_inputSize/(diff+1.0);
		}

		for (int i = 0; i < Wj->size(); ++i)
			sumWj 	+= fabs((*Wj)[i]);

		if(d_ACT == DEFAULT_ARTMAP)
			Tj = diff + (1 - d_alpha) * (d_inputSize - sumWj);

		if(d_ACT == FUZZY_ARTMAP)
			Tj = diff / (d_alpha + sumWj);

		if((d_ACT == DEFAULT_ARTMAP && Tj > d_alpha*d_inputSize) || d_ACT == FUZZY_ARTMAP || !d_useInputComplement)
		{
			PROTOTYPE_Activation *pr = new PROTOTYPE_Activation;
			pr->id = x;
			pr->T = Tj;
			pr->resonance = diff/d_inputSize;
			d_curPTAct.push(pr);
		}
		//else
		//	cout << "Prototype activation Tj lower then " << d_alpha*d_inputSize << endl;
	}
}

void Art::setVigilanceHistorySize(int vigilanceHistorySize)
{
	if(d_vigilanceHistorySize > vigilanceHistorySize)
		d_vigilanceHist.resize(vigilanceHistorySize);

	d_vigilanceHistorySize = vigilanceHistorySize;
}

void Art::addToVigilanceHistory(ART_TYPE vig)
{
	if(d_vigilanceHist.size() <= d_currVHist)
		d_vigilanceHist.push_back(vig);
	else
		d_vigilanceHist[d_currVHist] = vig;
	++d_currVHist;
	if(d_currVHist == d_vigilanceHistorySize)
		d_currVHist = 0;
}

ART_TYPE Art::getAVGVigilance()
{
	ART_TYPE avg = 0;
	for (int x = 0; x < d_vigilanceHist.size(); ++x)
		avg += d_vigilanceHist[x];
	if(d_vigilanceHist.size() > 0)
		avg /= float(d_vigilanceHist.size());
	if(avg == 0)
		return d_vigilance;
	return avg;
}

/**
 * Does the actual updating.
 */
void Art::updateWeights() {
	if(d_curPTAct.empty()) return;

	// Last node is send as winning node, update
	if(!d_testMatch)
	{
		PROTOTYPE_Activation *protA = d_curPTAct.top();
		PROTOTYPE *prot 			= (d_F2[protA->id]);

		// align prototype to input, do not change prototype size
		if(d_useInputComplement)
			if(prot->size() > d_F1.size())
			{
				for (int x = 0; x < prot->size(); ++x)
				{
					if(x < d_F1.size()/2)
						(*prot)[x] = d_learningFraction*( min(d_F1[x],(*prot)[x]) )+(1 - d_learningFraction)*(*prot)[x];
					else if(x >= d_F1.size()/2 &&  x < prot->size()/2)
						(*prot)[x] = d_learningFraction*( 0 )+(1 - d_learningFraction)*(*prot)[x];
					else if(x >= prot->size()/2 && ((d_F1.size()/2) + (x-(prot->size()/2))) < d_F1.size())
					{
						int indexF1 = (d_F1.size()/2) + (x-(prot->size()/2));
						(*prot)[x] = d_learningFraction*( min(d_F1[indexF1],(*prot)[x]) )+(1 - d_learningFraction)*(*prot)[x];
					}
					else
					{
						(*prot)[x] = d_learningFraction*( (*prot)[x] )+(1 - d_learningFraction)*(*prot)[x];
					}
				}
			}
			else
			{
				for (int x = 0; x < prot->size(); ++x)
				{
					int indexF1 = x;
					if(x >= prot->size()/2)
						indexF1 = (d_F1.size()/2) + (x-(prot->size()/2));
					(*prot)[x] = d_learningFraction*( min(d_F1[indexF1],(*prot)[x]) )+(1 - d_learningFraction)*(*prot)[x];
				}
			}
		else
		{
			for (int x = 0; x < prot->size(); ++x)
			{
				if(x < d_F1.size())
					(*prot)[x] = d_learningFraction*( min(d_F1[x],(*prot)[x]) )+(1 - d_learningFraction)*(*prot)[x];
				else
					(*prot)[x] = d_learningFraction*( 0 )+(1 - d_learningFraction)*(*prot)[x];

			}
		}

		if(d_vigilanceHistorySize > 0 && d_matchTrack)
			addToVigilanceHistory(protA->resonance-(d_alpha*10));
		++d_compressionCount;
	}
	//			// Print prototype x
	//			cout << "After: ";
	//			for (int x = 0; x < prot->size(); ++x)
	//			{
	//				cout << (*prot)[x] << " ";
	//			}
	//			cout << endl;

	// set resonance in history

	// Clear previous activations
	while (!d_curPTAct.empty())
	{
		delete d_curPTAct.top();
		d_curPTAct.pop();
	}
}

/**
 * The "body" of the ART class. First we have calculate the activity and vigilance levels in
 * signalToProtoType(). Now we want to use these to adjust our ART network. As you might
 * understand first this function is tend to called with finished=false. This means that the
 * this function iterates through the series of nodes to find
 * a
 */
std::vector<ART_TYPE>* Art::matchTrack(bool finished, bool raiseVigilance)
{
	std::vector<ART_TYPE> * output = new std::vector<ART_TYPE>(0);

	// Update weights
	if (finished) {
		updateWeights();
		return NULL;
	}

	float vigilance = d_vigilance;
	if(d_vigilanceHistorySize > 0)
		vigilance = getAVGVigilance();

	//cout << "AVigilance: " << getAVGVigilance() << " vig: " <<  vigilance << " candidates: " << d_curPTAct.size() << endl;

	if(d_matchTrack)
		vigilance = 0;

	// Find the node that matches the criterion
	// resonance >= vigilance
	// Only WTA implemented
	// TODO: Distributed output
	if(d_useWTA)
	{
		while (!d_curPTAct.empty())
		{
			PROTOTYPE_Activation *prot = d_curPTAct.top();

			// Last prototype was not correct
			// find new prototype with new vigilance
			if(raiseVigilance)
			{
				vigilance = prot->resonance+d_trackingValue;
				delete prot;
				d_curPTAct.pop();
				if (d_curPTAct.empty())
					continue;
				prot = d_curPTAct.top();
				raiseVigilance = false;
			}
			// Return winning node
			//cout << "prot: " << prot->id << " res: " << prot->resonance << endl;
			if(prot->resonance >= vigilance)
			{
				//cout << "prot: " << prot->id << " res: " << prot->resonance << " Tj:" << prot->T << " inpsize: " << d_inputSize << endl ;

				output->push_back(prot->id);
				return output;
			}
			// Remove node from list
			// Check if the nodes can still be chosen
			// when the vigilance changed
			else
			{
				//cout << "No resonance id:" << prot->id << " resonance:" << prot->resonance << " vig:" << vigilance << endl;
				delete prot;
				d_curPTAct.pop();
			}
		}

		// Return nothing if only testing for a match is used
		if(d_testMatch)
			return NULL;

		// If empty create new prototype
		PROTOTYPE *pr = new PROTOTYPE(0);
		for (int x = 0; x < d_F1.size(); ++x)
			(*pr).push_back(d_F1[x]);

		d_F2.push_back(pr);
		output->push_back(d_F2.size()-1);
		return output;
	}
	else
		return NULL;
}

/**
 * Just storing the network to the given file.
 */
void Art::saveArtNetwork(std::string fileName)
{
	ofstream outputFile(fileName.c_str(), ios::out | ios::binary);
	if(!outputFile)
		printf( "Cannot open ART output file.\n");
	else
	{
		outputFile.write((char *) &d_vigilance, sizeof(float));
		outputFile.write((char *) &d_alpha, sizeof(float));
		outputFile.write((char *) &d_inputSize, sizeof(float));
		outputFile.write((char *) &d_trackingValue, sizeof(float));
		outputFile.write((char *) &d_learningFraction, sizeof(float));
		outputFile.write((char *) &d_networkReliability, sizeof(float));
		outputFile.write((char *) &d_vigilanceHistorySize, sizeof(int));
		outputFile.write((char *) &d_currVHist, sizeof(int));
		outputFile.write((char *) &d_compressionCount, sizeof(int));
		outputFile.write((char *) &d_matchTrack, sizeof(bool));
		outputFile.write((char *) &d_useInputComplement, sizeof(bool));
		outputFile.write((char *) &d_useWTA, sizeof(bool));
		outputFile.write((char *) &d_testMatch, sizeof(bool));

		int size = d_F1.size();
		outputFile.write((char *) &size, sizeof(int));

		for (int x = 0; x < d_F1.size(); ++x)
			outputFile.write((char *) &(d_F1[x]), sizeof(ART_TYPE));

		size = d_F2.size();
		outputFile.write((char *) &size, sizeof(int));
		for (int x = 0; x < d_F2.size(); ++x)
		{
			size = d_F2[x]->size();
			outputFile.write((char *) &size, sizeof(int));
			for (int y = 0; y < d_F2[x]->size(); ++y)
				outputFile.write((char *) &(*(d_F2[x]))[y], sizeof(ART_TYPE));
		}
		size = d_vigilanceHist.size();
		outputFile.write((char *) &size, sizeof(int));
		for (int x = 0; x < d_vigilanceHist.size(); ++x)
			outputFile.write((char *) &(d_vigilanceHist[x]), sizeof(ART_TYPE));

		outputFile.close();
	}
}

/**
 * Loading the ART network from the given file.
 */
void Art::loadArtNetWork(std::string fileName)
{
	ifstream inputFile(fileName.c_str(),std::ios::in | std::ios::binary);

	// Load from file
	if(!inputFile.fail())
	{
		printf("Loading Art Network from file\n");
		inputFile.read((char *) &d_vigilance, sizeof(float));
		inputFile.read((char *) &d_alpha, sizeof(float));
		inputFile.read((char *) &d_inputSize, sizeof(float));
		inputFile.read((char *) &d_trackingValue, sizeof(float));
		inputFile.read((char *) &d_learningFraction, sizeof(float));
		inputFile.read((char *) &d_networkReliability, sizeof(float));
		inputFile.read((char *) &d_vigilanceHistorySize, sizeof(int));
		inputFile.read((char *) &d_currVHist, sizeof(int));
		inputFile.read((char *) &d_compressionCount, sizeof(int));
		inputFile.read((char *) &d_matchTrack, sizeof(bool));
		inputFile.read((char *) &d_useInputComplement, sizeof(bool));
		inputFile.read((char *) &d_useWTA, sizeof(bool));
		inputFile.read((char *) &d_testMatch, sizeof(bool));
		int size = 0;
		inputFile.read((char *) &size, sizeof(int));

		d_F1.clear();
		for (int x = 0; x < size; ++x)
		{
			ART_TYPE value = 0;
			inputFile.read((char *) &value, sizeof(ART_TYPE));
			d_F1.push_back(value);
		}

		inputFile.read((char *) &size, sizeof(int));
		for (int x = 0; x < size; ++x)
		{
			int size2 = 0;
			inputFile.read((char *) &size2, sizeof(int));

			d_F2.push_back(new vector<ART_TYPE>);

			for (int y = 0; y < size2; ++y)
			{
				ART_TYPE value = 0;
				inputFile.read((char *) &value, sizeof(ART_TYPE));
				(*(d_F2[x])).push_back(value);
			}
		}

		inputFile.read((char *) &size, sizeof(int));
		for (int x = 0; x < size; ++x)
		{
			ART_TYPE value = 0;
			inputFile.read((char *) &value, sizeof(ART_TYPE));
			d_vigilanceHist.push_back(value);
		}
	}
	else
		printf("Failed loading Art input file\n");

	inputFile.close();
}
}

