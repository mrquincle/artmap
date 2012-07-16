<!-- Uses markdown syntax for neat display at github -->

# ARTMAP
ARTMAP is a supervised variant (denoted by MAP) of Adaptive Resonance Theory (ART).

## What does it do?
ARTMAP can be used for classification. There are many classification methods. ARTMAP maps vectors from an input space unto nodes in a higher layer. It stores long-term memory as the weights on the edges from the input nodes, layer L1, to the higher layer L2. ARTMAP can store classes relatively cheap, because an L2 node can correspond with multiple L1 nodes. ARTMAP works by a matching operator, if a new input is close enough to the weights towards one L2 node, it is recognized and these weights are adapted. If not, another L2 node is selected. If none of the nodes are close enough, a new node is generated.

## Is it good?
Every classification method has its own merits. ARTMAP has the following properties: a.) it is incremental (not all training examples need to be there, useful on a robot), b.) the number of clusters do not need to be specific beforehand (as for example with ordinary k-means clustering), and c.) outliers are not removed (no condensing scheme for denoising).

## What are the alternatives?
Almende and DO bots have been using ARTMAP (the unsupervised version of Adaptive Resonance Theory) in the European robotics project Replicator. Other alternatives can be: iLVQ, neural gas, and maybe extensions of principle component analysis or k-means clustering. However, because they are not prototype-based and not incremental, this would stretch the imagination.

Note, that we are not interested in proving that ARTMAP is better than anything else. Sometimes any classification method will do, as long as it indeed fits the requirements to run on a robot.

## An example
The classification task is simple because of illustrative purposes. A so-called circle-in-a-square problem. There are two classes, one inside a circle, one outside of the circle. The expected result on visualisation would be a perfect circle. However, because of the dynamics in ARTMAP, the circle is not perfect, but it's pretty fine. 

![alt text](https://github.com/mrquincle/artmap/raw/master/doc/artmap_circle.jpg "ARTMAP circle")

## Where can I read more?
* [Wikipedia](http://en.wikipedia.org/wiki/Adaptive_resonance_theory)

## Copyrights
The copyrights (2012) belong to:

- Author: Ted Schmidt
- Author: Anne van Rossum
- Almende B.V., http://www.almende.com and DO bots B.V., http://www.dobots.nl
- Rotterdam, The Netherlands
